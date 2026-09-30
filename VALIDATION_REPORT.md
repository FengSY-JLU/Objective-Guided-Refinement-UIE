# Public release validation report

Release package: `public_release_final`

The following checks were run after public-path sanitization.

## Checkpoint contract
All five released OMRR checkpoints passed:
- checkpoint format: `r27_pnp_omrr_core_v1`
- tensor count: 7
- parameter count: 11,906
- SHA256 matched the frozen release manifest

## Core forward
A CPU synthetic forward through the released UIEBD OMRR core passed.
The output was finite, shape-preserving, and routing weights summed to one.

## Python compilation
The public inference/training wrappers, exact frozen formal trainer, OMRR core,
objective implementations, prior builder, and Restormer architecture compiled
successfully.

## Formal provenance
The retained frozen multidataset trainer and its two checksum-gated inputs
matched the original frozen SHA256 values:

- formal trainer:
  `1c8efb04ab45b24f32b4977da58c1e028c65ff6986e4b85d411f209029447956`
- formal training protocol:
  `77bfe759bf0a2f3ed2cdb55ea259fd611262f093492d21a5ddaa3773a2b3821d`
- frozen dataset profiles:
  `3140cdfc8822f949e6a3e3f13f0db166d061d4b715d706d1733d4330b72a2fa0`

## UIEBD split sanitization
Public UIEBD manifests retain the frozen membership:
- train: 626
- validation: 70
- test: 190
- excluded train/test duplicates: 4

Only machine-local path strings were rewritten to portable relative paths.

## Real end-to-end inference
A real end-to-end CPU smoke test was run with the exact frozen Restormer prior
(SHA256 `b2f6fa945708afff4ba58a1f0a3c4aab95965ef6086d646b3a5d4d6f73f4486c`)
and the released UIEBD OMRR core.

Pipeline:
`raw image -> formal K4 restoration-output bank -> M2 synthesis anchor -> OMRR -> refined image`

Result:
`PASS_INFERENCE count=1`

Mean routing weights on the synthetic smoke image:
- CLAHE: 0.3857336342
- Unsharp: 0.0478681028
- GrayWorld: 0.3330027163
- Restormer: 0.2333955318

The smoke values are implementation checks only and are not paper results.
