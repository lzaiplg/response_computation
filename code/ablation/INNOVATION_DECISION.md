# Innovation decision record

The package retains a sensor-availability-conditioned neural operator as the common B0/B1 family. The model outputs the full `[B, 3, 153, 1000]` response field; B0 uses zero sensor values and mask, while B1 uses the same ground input plus five sparse sensors and mask.

The current candidate mechanisms are:

1. coordinate-aware sensor correction rather than global sensor averaging;
2. spectral/multi-scale temporal processing for earthquake response bands;
3. phase, slope, spectrum, spatial-neighbor, sensor-consistency, peak and direction-balanced losses;
4. exact zero-sensor limit, so the B0 branch is not contaminated by unavailable observations;
5. immutable campaigns with matched B0/B1 pairs and per-record gains.

The first full screen is defined in `experiment_registry_v3.json`. It contains 14 cases: seven strategies represented by paired B0/B1 conditions. The formal 78-record test is available in this package and is evaluated only after model and strategy selection. It must never be used for training, early stopping or ranking.

The physics boundary is explicit: `physics_proxy` is not a full finite-element equilibrium residual. A genuine M/C/K residual can be added only after validated mass, damping, stiffness or modal data are supplied.

For a publishable claim, retain a strategy only after three seeds, matched B0/B1 comparisons, full-field and per-direction metrics, peak error, pier-top error, per-record win rate and formal-test reporting are all complete.
