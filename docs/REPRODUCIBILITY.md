# Reproducibility scope

## Released directly
- Five selected 11,906-parameter OMRR core checkpoints.
- Frozen K4 restoration-output / M2 anchor configurations.
- UIEBD 626/70/190 split manifests.
- UFO120, RUIE, OceanDark, and EUVP split manifests.
- Exact formal multidataset trainer for UFO120/EUVP/RUIE/OceanDark.
- Reachability-realization and runtime summary tables.

## External asset
The formal bank includes one frozen Restormer output. Its approximately 100 MB
checkpoint is intentionally kept outside the Git repository. The exact SHA256
is fixed in `assets/restormer/RESTORMER_WEIGHT_SHA256.txt`.

## UIEBD training provenance
The released UIEBD OMRR core is an exact seven-tensor extraction from the
development-selected E200 formal run. The extraction was verified by exact
tensor round-trip and against the full E200 router. The clean public inference
path consumes only this OMRR core. The earlier instrumented training container
contained retired architecture modules that are not part of the revised
manuscript and are therefore not exposed as the main public API.

## Public path sanitization
See `docs/PUBLIC_RELEASE_SANITIZATION.md`.
