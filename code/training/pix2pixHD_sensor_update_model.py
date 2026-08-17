# -*- coding: utf-8 -*-
"""
Corrected pix2pixHD-style seismic model for 7-channel sensor update, closer to the architecture described in:
《数字孪生框架下土石坝地震动力响应时空分析系统研究》

Important scientific note:
- This is NOT the authors' source code.
- The paper does not disclose every implementation detail.
- The corrections below make the implementation closer to the written description:
  1) G1 now has actual U-Net skip connections.
  2) G2 now has parallel multi-scale local residual branches.
  3) 153 x 1000 inputs are reflection-padded to a multiple of 32 and cropped back,
     avoiding repeated 144->153 and 992->1000 output interpolation.
  4) The two multiscale discriminators are retained.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image


@dataclass
class Config:
    input_nc: int = 3
    output_nc: int = 3
    ngf: int = 64
    ndf: int = 64
    n_residual_blocks: int = 9
    n_downsamples_g1: int = 4
    num_d: int = 2
    low_res_factor: int = 2
    pad_multiple: int = 32


class ResidualBlock(nn.Module):
    """Convolution + normalization + LeakyReLU residual module."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        effective = dilation * (kernel_size - 1) + 1
        padding = effective // 2
        self.block = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(padding),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetUpsampleBlock(nn.Module):
    """Transposed convolution followed by an actual U-Net skip fusion."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
    ) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                out_channels + skip_channels,
                out_channels,
                kernel_size=3,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            raise RuntimeError(
                f"U-Net skip shape mismatch: up={tuple(x.shape)}, "
                f"skip={tuple(skip.shape)}. Input padding is incorrect."
            )
        return self.fusion(torch.cat([x, skip], dim=1))


class GlobalGeneratorG1(nn.Module):
    """G1: 4 downsampling + 9 residual + 4 upsampling with U-Net skips."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.n_downsamples = config.n_downsamples_g1
        self.n_residual_blocks = config.n_residual_blocks
        self.n_upsamples = config.n_downsamples_g1

        ngf = config.ngf
        self.input_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(config.input_nc, ngf, kernel_size=7, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
        )

        encoder_channels = [ngf]
        down_blocks: list[nn.Module] = []
        in_channels = ngf
        for index in range(config.n_downsamples_g1):
            out_channels = ngf * min(2 ** (index + 1), 16)
            down_blocks.append(DownsampleBlock(in_channels, out_channels))
            encoder_channels.append(out_channels)
            in_channels = out_channels
        self.down_blocks = nn.ModuleList(down_blocks)

        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(in_channels)
                for _ in range(config.n_residual_blocks)
            ]
        )

        # Skip features: input feature, down1, down2, down3.
        skip_channels = list(reversed(encoder_channels[:-1]))
        up_blocks: list[nn.Module] = []
        for skip_channel in skip_channels:
            out_channels = skip_channel
            up_blocks.append(
                UNetUpsampleBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    skip_channels=skip_channel,
                )
            )
            in_channels = out_channels
        self.up_blocks = nn.ModuleList(up_blocks)

        self.output_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, config.output_nc, kernel_size=7),
            nn.Tanh(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        original_size = x.shape[-2:]

        first = self.input_layer(x)
        encoder_features = [first]
        features = first
        for block in self.down_blocks:
            features = block(features)
            encoder_features.append(features)

        bottleneck = self.residual_blocks(features)
        decoded = bottleneck

        # Exclude bottleneck feature itself; use down3, down2, down1, input.
        for block, skip in zip(
            self.up_blocks,
            reversed(encoder_features[:-1]),
        ):
            decoded = block(decoded, skip)

        output = self.output_layer(decoded)
        if output.shape[-2:] != original_size:
            raise RuntimeError(
                f"G1 output shape {tuple(output.shape[-2:])} does not equal "
                f"input shape {tuple(original_size)}."
            )
        return output, bottleneck


class LocalMultiScaleBranch(nn.Module):
    """One local residual branch operating at a specific receptive field."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.block = ResidualBlock(
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LocalEnhancerG2(nn.Module):
    """
    G2 local enhancer with parallel multi-scale residual branches.

    Branches:
    - 3 x 3 local branch
    - 3 x 3 dilated branch
    - 5 x 5 wider branch
    """

    def __init__(
        self,
        config: Config,
        global_channels: int,
    ) -> None:
        super().__init__()
        ngf = config.ngf
        local_channels = ngf * 2

        self.local_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(config.input_nc, ngf, kernel_size=7, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            DownsampleBlock(ngf, local_channels),
        )

        self.local_branches = nn.ModuleList(
            [
                LocalMultiScaleBranch(local_channels, 3, 1),
                LocalMultiScaleBranch(local_channels, 3, 2),
                LocalMultiScaleBranch(local_channels, 5, 1),
            ]
        )
        self.branch_fusion = nn.Sequential(
            nn.Conv2d(
                local_channels * len(self.local_branches),
                local_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(local_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.global_projection = nn.Sequential(
            nn.Conv2d(
                global_channels,
                local_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(local_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.global_local_fusion = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                local_channels * 2,
                local_channels,
                kernel_size=3,
                bias=False,
            ),
            nn.BatchNorm2d(local_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Retains the paper-described transposed-convolution enhancement layer.
        self.local_deconv = nn.Sequential(
            nn.ConvTranspose2d(
                local_channels,
                ngf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.output_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, config.output_nc, kernel_size=7),
            nn.Tanh(),
        )

    def forward(
        self,
        x: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        original_size = x.shape[-2:]
        local = self.local_conv(x)

        branches = [branch(local) for branch in self.local_branches]
        local = self.branch_fusion(torch.cat(branches, dim=1))

        projected_global = self.global_projection(global_features)
        projected_global = F.interpolate(
            projected_global,
            size=local.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        fused = self.global_local_fusion(
            torch.cat([local, projected_global], dim=1)
        )
        decoded = self.local_deconv(fused)
        output = self.output_layer(decoded)

        if output.shape[-2:] != original_size:
            raise RuntimeError(
                f"G2 output shape {tuple(output.shape[-2:])} does not equal "
                f"input shape {tuple(original_size)}."
            )
        return output


def pad_to_multiple(
    x: torch.Tensor,
    multiple: int,
) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = x.shape[-2:]
    pad_height = (-height) % multiple
    pad_width = (-width) % multiple

    if pad_height == 0 and pad_width == 0:
        return x, (height, width)

    padded = F.pad(
        x,
        (0, pad_width, 0, pad_height),
        mode="reflect",
    )
    return padded, (height, width)


class Pix2PixHDSeismicGenerator(nn.Module):
    """Two-level G={G1,G2}, with divisible-size padding and exact cropping."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.g1 = GlobalGeneratorG1(config)
        global_channels = config.ngf * min(
            2 ** config.n_downsamples_g1,
            16,
        )
        self.g2 = LocalEnhancerG2(config, global_channels)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padded, original_size = pad_to_multiple(
            x,
            self.config.pad_multiple,
        )

        padded_height, padded_width = padded.shape[-2:]
        low_size = (
            padded_height // self.config.low_res_factor,
            padded_width // self.config.low_res_factor,
        )
        low_input = F.interpolate(
            padded,
            size=low_size,
            mode="bilinear",
            align_corners=False,
        )

        g1_low_output, global_features = self.g1(low_input)
        g2_padded_output = self.g2(padded, global_features)

        height, width = original_size
        g2_output = g2_padded_output[..., :height, :width]
        return g2_output, g1_low_output


class PatchDiscriminator(nn.Module):
    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        ndf: int,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        channels = input_nc + output_nc

        sequence: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(
                    channels,
                    ndf,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.LeakyReLU(0.2, inplace=True),
            )
        ]

        in_channels = ndf
        for layer_index in range(1, n_layers):
            out_channels = ndf * min(2 ** layer_index, 8)
            stride = 1 if layer_index == n_layers - 1 else 2
            sequence.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=4,
                        stride=stride,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            in_channels = out_channels

        sequence.append(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
            )
        )
        self.layers = nn.ModuleList(sequence)

    def forward(
        self,
        condition: torch.Tensor,
        response: torch.Tensor,
    ) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        x = torch.cat([condition, response], dim=1)
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.num_d = config.num_d
        self.discriminators = nn.ModuleList(
            [
                PatchDiscriminator(
                    config.input_nc,
                    config.output_nc,
                    config.ndf,
                )
                for _ in range(config.num_d)
            ]
        )
        self.downsample = nn.AvgPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
            count_include_pad=False,
        )

    def forward(
        self,
        condition: torch.Tensor,
        response: torch.Tensor,
    ) -> list[list[torch.Tensor]]:
        results: list[list[torch.Tensor]] = []
        condition_scaled = condition
        response_scaled = response
        for index, discriminator in enumerate(self.discriminators):
            results.append(
                discriminator(condition_scaled, response_scaled)
            )
            if index != self.num_d - 1:
                condition_scaled = self.downsample(condition_scaled)
                response_scaled = self.downsample(response_scaled)
        return results


class GANLoss(nn.Module):
    def __init__(
        self,
        real_label: float = 1.0,
        fake_label: float = 0.0,
    ) -> None:
        super().__init__()
        self.real_label = float(real_label)
        self.fake_label = float(fake_label)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        predictions: Iterable[list[torch.Tensor]],
        target_is_real: bool,
    ) -> torch.Tensor:
        total = 0.0
        count = 0
        value = self.real_label if target_is_real else self.fake_label

        for scale_predictions in predictions:
            logits = scale_predictions[-1]
            target = torch.full_like(logits, value)
            total = total + self.loss(logits, target)
            count += 1

        return total / max(1, count)


def feature_matching_loss(
    fake_features: list[list[torch.Tensor]],
    real_features: list[list[torch.Tensor]],
) -> torch.Tensor:
    total = 0.0
    count = 0
    for fake_scale, real_scale in zip(fake_features, real_features):
        for fake, real in zip(fake_scale[:-1], real_scale[:-1]):
            total = total + F.l1_loss(fake, real.detach())
            count += 1
    return total / max(1, count)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(-1.0, 1.0)
    array = (
        ((tensor + 1.0) * 127.5)
        .round()
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def make_models(
    config: Config,
    device: torch.device,
) -> tuple[Pix2PixHDSeismicGenerator, MultiscaleDiscriminator]:
    generator = Pix2PixHDSeismicGenerator(config).to(device)
    discriminator = MultiscaleDiscriminator(config).to(device)
    assert_corrected_architecture(generator, discriminator)
    return generator, discriminator


def set_requires_grad(
    model: nn.Module,
    requires_grad: bool,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(requires_grad)


def save_checkpoint(
    path: Path,
    generator: Pix2PixHDSeismicGenerator,
    discriminator: MultiscaleDiscriminator,
    optimizer_g: optim.Optimizer,
    optimizer_d: optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    include_optimizers: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "pix2pixHD_paper_corrected_v1",
        "epoch": int(epoch),
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "metrics": metrics,
        "model_config": asdict(generator.config),
    }
    if include_optimizers:
        checkpoint["optimizer_g"] = optimizer_g.state_dict()
        checkpoint["optimizer_d"] = optimizer_d.state_dict()
    torch.save(checkpoint, path)


def corrected_architecture_report(
    generator: Pix2PixHDSeismicGenerator,
    discriminator: MultiscaleDiscriminator,
) -> list[str]:
    return [
        "Implementation status: closer-to-paper reconstruction; not author source code",
        f"Input padding multiple: {generator.config.pad_multiple}",
        "153x1000 -> 160x1024 padding -> exact network reconstruction -> crop",
        f"G1 downsampling blocks: {len(generator.g1.down_blocks)}",
        f"G1 residual blocks: {len(generator.g1.residual_blocks)}",
        f"G1 upsampling blocks with skip fusion: {len(generator.g1.up_blocks)}",
        f"G2 parallel local branches: {len(generator.g2.local_branches)}",
        f"Multiscale discriminators: {len(discriminator.discriminators)}",
        "Output activation: Tanh",
    ]


def assert_corrected_architecture(
    generator: Pix2PixHDSeismicGenerator,
    discriminator: MultiscaleDiscriminator,
) -> None:
    if len(generator.g1.down_blocks) != 4:
        raise AssertionError("G1 must have 4 downsampling blocks.")
    if len(generator.g1.residual_blocks) != 9:
        raise AssertionError("G1 must have 9 residual blocks.")
    if len(generator.g1.up_blocks) != 4:
        raise AssertionError("G1 must have 4 U-Net upsampling blocks.")
    if len(generator.g2.local_branches) < 2:
        raise AssertionError("G2 must have multiple local branches.")
    if len(discriminator.discriminators) != 2:
        raise AssertionError("Two multiscale discriminators are required.")
