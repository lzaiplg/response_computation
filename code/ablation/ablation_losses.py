from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossWeights:
    waveform: float = 1.0
    slope: float = 0.0
    spectrum: float = 0.0
    phase: float = 0.0
    spatial: float = 0.0
    sensor: float = 0.0
    peak: float = 0.0
    direction: float = 0.0
    ranking: float = 0.0
    correction: float = 1.0e-4


def profile_weights(profile: str) -> LossWeights:
    if profile == "basic":
        return LossWeights()
    if profile == "physical":
        return LossWeights(waveform=1.0, slope=0.15, spectrum=0.08, peak=0.20, ranking=0.10)
    if profile == "peak_balanced":
        # Keep the peak term deliberately small: the previous physical profile
        # over-weighted rare peaks and degraded global full-field error.
        return LossWeights(waveform=1.0, slope=0.02, spectrum=0.015, peak=0.06, ranking=0.0)
    if profile == "phase_balanced":
        # Preserve the global waveform objective while explicitly penalizing
        # phase distortion and the worst direction. This targets the observed
        # sensor-free X-direction amplitude/phase collapse.
        return LossWeights(
            waveform=1.0,
            slope=0.05,
            spectrum=0.04,
            phase=0.12,
            spatial=0.03,
            sensor=0.20,
            peak=0.02,
            direction=0.30,
        )
    if profile == "physics_proxy":
        # No M/C/K matrices are currently available. These are auditable
        # physics proxies: temporal derivative, frequency phase, spatial
        # neighbor consistency, sensor self-consistency, and worst-direction
        # control. A future FE-backed profile must be kept separate.
        return LossWeights(
            waveform=1.0,
            slope=0.08,
            spectrum=0.04,
            phase=0.06,
            spatial=0.10,
            sensor=0.30,
            peak=0.03,
            direction=0.20,
        )
    if profile == "direction_balanced":
        return LossWeights(
            waveform=1.0,
            slope=0.03,
            spectrum=0.02,
            phase=0.05,
            spatial=0.04,
            sensor=0.20,
            peak=0.02,
            direction=0.50,
        )
    raise ValueError(profile)


def weighted_mean(value: torch.Tensor, row_weights: torch.Tensor) -> torch.Tensor:
    weights = row_weights.view(1, 1, -1, 1).to(value)
    return (value * weights).sum() / (weights.sum() * value.shape[0] * value.shape[1] * value.shape[3])


def component_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    row_weights: torch.Tensor,
    profile: str,
    sensor_target: torch.Tensor | None = None,
    sensor_rows: torch.Tensor | None = None,
    spatial_edges: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    weights = profile_weights(profile)
    waveform = weighted_mean(F.smooth_l1_loss(prediction, target, reduction="none", beta=0.05), row_weights)
    zero = waveform.new_zeros(())
    slope = zero
    spectrum = zero
    phase = zero
    spatial = zero
    sensor = zero
    peak = zero
    direction = zero
    if weights.slope:
        pred_slope = prediction[..., 1:] - prediction[..., :-1]
        true_slope = target[..., 1:] - target[..., :-1]
        slope = weighted_mean(F.smooth_l1_loss(pred_slope, true_slope, reduction="none", beta=0.02), row_weights)
    if weights.spectrum:
        # A fixed node subsample makes the spectral term cheap and reproducible.
        indices = torch.linspace(0, prediction.shape[2] - 1, 40, device=prediction.device).round().long().unique()
        pred_fft = torch.fft.rfft(prediction.index_select(2, indices), dim=-1)
        true_fft = torch.fft.rfft(target.index_select(2, indices), dim=-1)
        spectrum = F.l1_loss(torch.log1p(pred_fft.abs()), torch.log1p(true_fft.abs()))
        if weights.phase:
            # Compare unit complex spectra so the term targets phase rather
            # than merely duplicating the amplitude spectrum loss.
            pred_unit = pred_fft[..., 1:] / pred_fft[..., 1:].abs().clamp_min(1.0e-4)
            true_unit = true_fft[..., 1:] / true_fft[..., 1:].abs().clamp_min(1.0e-4)
            phase = F.l1_loss(torch.view_as_real(pred_unit), torch.view_as_real(true_unit))
    elif weights.phase:
        indices = torch.linspace(0, prediction.shape[2] - 1, 40, device=prediction.device).round().long().unique()
        pred_fft = torch.fft.rfft(prediction.index_select(2, indices), dim=-1)[..., 1:]
        true_fft = torch.fft.rfft(target.index_select(2, indices), dim=-1)[..., 1:]
        pred_unit = pred_fft / pred_fft.abs().clamp_min(1.0e-4)
        true_unit = true_fft / true_fft.abs().clamp_min(1.0e-4)
        phase = F.l1_loss(torch.view_as_real(pred_unit), torch.view_as_real(true_unit))
    if weights.spatial and spatial_edges is not None and spatial_edges.numel() > 0:
        source, destination = spatial_edges[0], spatial_edges[1]
        pred_difference = prediction.index_select(2, source) - prediction.index_select(2, destination)
        true_difference = target.index_select(2, source) - target.index_select(2, destination)
        spatial = F.smooth_l1_loss(pred_difference, true_difference, beta=0.02)
    if weights.sensor and sensor_target is not None and sensor_rows is not None and sensor_rows.numel() > 0:
        predicted_sensor = prediction.index_select(2, sensor_rows).permute(0, 2, 1, 3)
        sensor = F.smooth_l1_loss(predicted_sensor, sensor_target, beta=0.02)
    if weights.peak:
        pred_peak = prediction.abs().amax(dim=-1)
        true_peak = target.abs().amax(dim=-1)
        peak_error = F.smooth_l1_loss(pred_peak, true_peak, reduction="none", beta=0.05)
        # Emphasize high-response nodes without allowing a single rare peak to
        # dominate the waveform objective. All terms remain in normalized space.
        emphasis = (true_peak.detach() / true_peak.detach().mean().clamp_min(1.0e-3)).clamp(0.5, 3.0)
        peak = (peak_error * emphasis).mean()
    if weights.direction:
        error = prediction - target
        direction_mse = (error.square() * row_weights.view(1, 1, -1, 1).to(error)).sum(dim=(2, 3))
        direction_mse = direction_mse / (row_weights.sum().clamp_min(1.0e-6) * prediction.shape[-1])
        direction_rmse = torch.sqrt(direction_mse.clamp_min(1.0e-8))
        # Stop the largest direction from being hidden by the average while
        # retaining smooth gradients for the other directions.
        attention = torch.softmax(direction_rmse.detach() * 10.0, dim=1)
        direction = (direction_rmse * attention).sum(dim=1).mean()
    total = (
        weights.waveform * waveform
        + weights.slope * slope
        + weights.spectrum * spectrum
        + weights.phase * phase
        + weights.spatial * spatial
        + weights.sensor * sensor
        + weights.peak * peak
        + weights.direction * direction
    )
    return {
        "total": total,
        "waveform": waveform,
        "slope": slope,
        "spectrum": spectrum,
        "phase": phase,
        "spatial": spatial,
        "sensor": sensor,
        "peak": peak,
        "direction": direction,
    }


def per_sample_rmse(prediction: torch.Tensor, target: torch.Tensor, unobserved: torch.Tensor) -> torch.Tensor:
    error = prediction.index_select(2, unobserved) - target.index_select(2, unobserved)
    return torch.sqrt(torch.mean(error * error, dim=(1, 2, 3)).clamp_min(1.0e-12))
