from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class OperatorConfig:
    width: int = 32
    blocks: int = 3
    temporal_downsample: int = 4
    fusion: str = "coord_attention"
    architecture: str = "sacno"
    graph_neighbors: int = 8
    dropout: float = 0.05


class TemporalEncoder(nn.Module):
    def __init__(self, in_channels: int, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, width, 9, stride=2, padding=4),
            nn.GELU(),
            nn.Conv1d(width, width, 7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(width, width, 5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiScaleTemporalEncoder(nn.Module):
    """A low-cost multi-receptive-field encoder for the sensor-free prior."""

    def __init__(self, in_channels: int, width: int) -> None:
        super().__init__()
        self.input = nn.Conv1d(in_channels, width, 9, stride=2, padding=4)
        self.branches = nn.ModuleList(
            [nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation, groups=width) for dilation in (1, 2, 4)]
        )
        self.mix = nn.Conv1d(width, width, 1)
        self.downsample = nn.Conv1d(width, width, 7, stride=2, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem = F.gelu(self.input(x))
        multi_scale = sum(F.gelu(branch(stem)) for branch in self.branches) / len(self.branches)
        mixed = stem + F.gelu(self.mix(multi_scale))
        return F.gelu(self.downsample(mixed))


class AdaptiveMultiScaleTemporalEncoder(nn.Module):
    """Sample-adaptive mixture of short-, medium- and long-range branches."""

    def __init__(self, in_channels: int, width: int) -> None:
        super().__init__()
        self.input = nn.Conv1d(in_channels, width, 9, stride=2, padding=4)
        self.branches = nn.ModuleList(
            [nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation, groups=width) for dilation in (1, 2, 4)]
        )
        gate_width = max(width // 4, 4)
        self.scale_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(width, gate_width, 1),
            nn.GELU(),
            nn.Conv1d(gate_width, 3, 1),
        )
        # Start from the fixed multi-scale average; learn sample-dependent
        # scale selection only after the shared prior is already well-defined.
        nn.init.zeros_(self.scale_gate[-1].weight)
        nn.init.zeros_(self.scale_gate[-1].bias)
        self.mix = nn.Conv1d(width, width, 1)
        self.downsample = nn.Conv1d(width, width, 7, stride=2, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem = F.gelu(self.input(x))
        branches = torch.stack([F.gelu(branch(stem)) for branch in self.branches], dim=1)
        gate = torch.softmax(self.scale_gate(stem).squeeze(-1), dim=-1)
        multi_scale = (branches * gate[:, :, None, None]).sum(dim=1)
        mixed = stem + F.gelu(self.mix(multi_scale))
        return F.gelu(self.downsample(mixed))


class SpectralBandTemporalEncoder(nn.Module):
    """Low/high-frequency decomposition for dominant seismic oscillations.

    The decomposition is deterministic and uses only the deployment input.
    It is not a substitute for an FE dynamic residual; it exposes a compact
    frequency mechanism that can be tested independently from the loss.
    """

    def __init__(self, in_channels: int, width: int) -> None:
        super().__init__()
        self.input = nn.Conv1d(in_channels, width, 9, stride=2, padding=4)
        self.local = nn.Conv1d(width, width, 5, padding=2, groups=width)
        self.mix = nn.Conv1d(width * 3, width, 1)
        self.downsample = nn.Conv1d(width, width, 7, stride=2, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem = F.gelu(self.input(x))
        length = stem.shape[-1]
        spectrum = torch.fft.rfft(stem, dim=-1)
        frequency = torch.linspace(0.0, 1.0, spectrum.shape[-1], device=stem.device, dtype=stem.dtype)
        low_mask = torch.exp(-6.0 * frequency).view(1, 1, -1)
        high_mask = 1.0 - low_mask
        low = torch.fft.irfft(spectrum * low_mask, n=length, dim=-1)
        high = torch.fft.irfft(spectrum * high_mask, n=length, dim=-1)
        local = F.gelu(self.local(stem))
        mixed = stem + F.gelu(self.mix(torch.cat([local, low, high], dim=1)))
        return F.gelu(self.downsample(mixed))


def coordinate_adjacency(coords: torch.Tensor, neighbors: int) -> torch.Tensor:
    distance = torch.cdist(coords, coords)
    k = min(int(neighbors) + 1, int(coords.shape[0]))
    values, indices = torch.topk(distance, k=k, dim=1, largest=False)
    sigma = values[:, 1:].mean().clamp_min(1.0e-6)
    adjacency = torch.zeros_like(distance)
    weights = torch.exp(-(values * values) / (2.0 * sigma * sigma))
    adjacency.scatter_(1, indices, weights)
    adjacency = 0.5 * (adjacency + adjacency.T)
    adjacency = adjacency + torch.eye(coords.shape[0], device=coords.device, dtype=coords.dtype)
    return adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1.0e-8)


class GraphTemporalBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        groups = 8 if width % 8 == 0 else 4
        self.norm = nn.GroupNorm(groups, width)
        self.temporal = nn.Conv2d(width, width, (1, 7), padding=(0, 3), groups=width)
        self.channel = nn.Conv2d(width, width, 1)
        self.graph_mix = nn.Conv2d(width, width, 1)
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        temporal = self.channel(F.gelu(self.temporal(z)))
        graph = torch.einsum("nm,bcmt->bcnt", adjacency, z)
        graph = self.graph_mix(F.gelu(graph))
        mixed = temporal + torch.sigmoid(self.gate) * graph
        return x + self.dropout(mixed)


class SensorFusion(nn.Module):
    def __init__(self, width: int, fusion: str) -> None:
        super().__init__()
        if fusion not in {"mean", "coord_attention", "coord_quality"}:
            raise ValueError(fusion)
        self.fusion = fusion
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.distance_scale = nn.Parameter(torch.tensor(1.0))
        self.quality = (
            nn.Sequential(
                nn.Linear(width, max(width // 4, 4)),
                nn.GELU(),
                nn.Linear(max(width // 4, 4), 1),
            )
            if fusion == "coord_quality"
            else None
        )

    def forward(
        self,
        sensor_features: torch.Tensor,
        node_embedding: torch.Tensor,
        sensor_embedding: torch.Tensor,
        sensor_mask: torch.Tensor,
        node_coords: torch.Tensor,
        sensor_coords: torch.Tensor,
    ) -> torch.Tensor:
        # sensor_features [B,S,C,Td], mask [B,S]
        mask = sensor_mask.to(sensor_features.dtype)
        if self.fusion == "mean":
            denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (sensor_features * mask[:, :, None, None]).sum(dim=1) / denominator[:, :, None]
            return pooled[:, :, None, :].expand(-1, -1, node_embedding.shape[0], -1)

        q = self.query(node_embedding)
        k = self.key(sensor_embedding)
        logits = q @ k.T / math.sqrt(q.shape[-1])
        distance = torch.cdist(node_coords, sensor_coords)
        logits = logits - F.softplus(self.distance_scale) * distance
        logits = logits[None, :, :].expand(sensor_features.shape[0], -1, -1)
        maximum = logits.max(dim=-1, keepdim=True).values
        effective_mask = mask
        if self.fusion == "coord_quality":
            assert self.quality is not None
            quality = torch.sigmoid(self.quality(sensor_features.mean(dim=-1)).squeeze(-1))
            effective_mask = effective_mask * quality
        weights = torch.exp(logits - maximum) * effective_mask[:, None, :]
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        context = torch.einsum("bns,bsct->bcnt", weights, sensor_features)
        return context


class SensorAvailabilityConditionedOperator(nn.Module):
    """One nested operator: ground-only prior plus an exactly mask-gated sensor correction."""

    def __init__(self, coords: torch.Tensor, sensor_rows: list[int], config: OperatorConfig) -> None:
        super().__init__()
        self.config = config
        if config.architecture not in {"sacno", "ms_sacno", "ams_sacno", "spectral_sacno", "qg_sacno"}:
            raise ValueError(config.architecture)
        self.register_buffer("coords", coords.float().clone())
        self.register_buffer("sensor_rows", torch.as_tensor(sensor_rows, dtype=torch.long))
        self.register_buffer("adjacency", coordinate_adjacency(coords.float(), config.graph_neighbors))
        if config.architecture == "ms_sacno":
            encoder_type = MultiScaleTemporalEncoder
        elif config.architecture == "ams_sacno":
            encoder_type = AdaptiveMultiScaleTemporalEncoder
        elif config.architecture == "spectral_sacno":
            encoder_type = SpectralBandTemporalEncoder
        else:
            encoder_type = TemporalEncoder
        self.ground_encoder = encoder_type(3, config.width)
        self.sensor_encoder = encoder_type(3, config.width)
        self.coord_encoder = nn.Sequential(
            nn.Linear(3, config.width),
            nn.GELU(),
            nn.Linear(config.width, config.width),
        )
        self.base_stem = nn.Conv2d(config.width, config.width, 1)
        self.base_blocks = nn.ModuleList(
            [GraphTemporalBlock(config.width, config.dropout) for _ in range(config.blocks)]
        )
        fusion_type = "coord_quality" if config.architecture == "qg_sacno" else config.fusion
        self.fusion = SensorFusion(config.width, fusion_type)
        self.correction_stem = nn.Conv2d(config.width * 2, config.width, 1)
        self.correction_blocks = nn.ModuleList(
            [GraphTemporalBlock(config.width, config.dropout) for _ in range(config.blocks)]
        )
        self.base_head = nn.Sequential(
            nn.Conv2d(config.width, config.width, 1), nn.GELU(), nn.Conv2d(config.width, 3, 1)
        )
        self.correction_head = nn.Sequential(
            nn.Conv2d(config.width, config.width, 1), nn.GELU(), nn.Conv2d(config.width, 3, 1)
        )
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

    def strategy_signature(self) -> dict[str, object]:
        return {"class": type(self).__name__, "config": asdict(self.config), "sensor_rows": self.sensor_rows.tolist()}

    def forward(
        self,
        ground: torch.Tensor,
        sensor: torch.Tensor,
        sensor_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # ground [B,3,T], sensor [B,S,3,T], mask [B,S]
        batch, _, full_time = ground.shape
        sensor_count = sensor.shape[1]
        node_embedding = self.coord_encoder(self.coords)
        sensor_embedding = node_embedding.index_select(0, self.sensor_rows)

        ground_feature = self.ground_encoder(ground)
        reduced_time = ground_feature.shape[-1]
        base = ground_feature[:, :, None, :] + node_embedding.T[None, :, :, None]
        base = self.base_stem(base)
        for block in self.base_blocks:
            base = block(base, self.adjacency)

        sensor_flat = sensor.reshape(batch * sensor_count, 3, full_time)
        sensor_feature = self.sensor_encoder(sensor_flat).reshape(
            batch, sensor_count, self.config.width, reduced_time
        )
        context = self.fusion(
            sensor_feature,
            node_embedding,
            sensor_embedding,
            sensor_mask,
            self.coords,
            self.coords.index_select(0, self.sensor_rows),
        )
        correction = self.correction_stem(torch.cat([base, context], dim=1))
        for block in self.correction_blocks:
            correction = block(correction, self.adjacency)

        base_output = self.base_head(base)
        correction_output = self.correction_head(correction)
        base_output = F.interpolate(base_output, size=(self.coords.shape[0], full_time), mode="bilinear", align_corners=False)
        correction_output = F.interpolate(
            correction_output, size=(self.coords.shape[0], full_time), mode="bilinear", align_corners=False
        )
        available = (sensor_mask.sum(dim=1, keepdim=True) > 0).to(correction_output.dtype)
        correction_output = correction_output * available[:, :, None, None]
        return {
            "prior": base_output,
            "correction": correction_output,
            "posterior": base_output + correction_output,
        }
