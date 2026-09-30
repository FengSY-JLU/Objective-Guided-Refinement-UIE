# OGRR-UIE

**Objective-Guided Restoration-output Refinement for Underwater Image Enhancement**

Official code repository for the paper:

**Objective-Guided Refinement from Restoration Outputs for Self-Supervised Underwater Image Enhancement**

Shiyao Feng, Jun-Hong Cui, ChunRu Wan, and Gaochao Xu.

Repository: https://github.com/FengSY-JLU/Objective-Guided-Refinement-UIE

## Core idea

The method first establishes a parameter-free synthesis anchor and then retains
each complete restoration output as an individually queryable correction
direction. The Objective-Modulated Reliability Router (OMRR) combines
correction-evidence-conditioned spatial reliability with output-relative
objective regret and applies an output-supported bounded correction.

The released learned core contains **11,906 trainable parameters**, independent
of restoration-output cardinality.

## Important terminology note

The public wrappers use the manuscript terminology. A few checksum-frozen
research files retain historical internal names such as `ContributorBank` and
`StateObjectiveReliabilityRouterV1`. These names are retained for provenance,
not as current scientific claims. See `docs/TERMINOLOGY.md`.

## Repository contents

- `checkpoints/`: all five selected OMRR core checkpoints (~50 KB each).
- `configs/`: frozen bank/anchor/checkpoint configuration for each dataset.
- `src/model/pnp/`: frozen post-synthesis refinement implementation.
- `tools/infer.py`: end-to-end inference from an input image folder.
- `tools/infer_from_bank.py`: apply OMRR to precomputed restoration outputs.
- `tools/train.py`: path-configurable wrapper around the frozen formal
  multidataset trainer used for UFO120/EUVP/RUIE/OceanDark.
- `reproducibility/`: released mechanism and runtime tables plus UIEBD
  checkpoint provenance.
- `protocols/`: frozen development/test split manifests and formal protocol.

## 1. Installation

The formal development environment used Python 3.13 and PyTorch 2.7.1+cu118.
A practical environment can be created with:

```bash
conda create -n ogrr python=3.13 -y
conda activate ogrr
pip install -r requirements.txt
```

For CUDA, install the PyTorch build appropriate for your system if the plain
pip command does not provide the required CUDA runtime.

See `VALIDATION_REPORT.md` for the release validation record.

## 2. Verify released OMRR checkpoints

```bash
python tools/verify_release.py
```

Expected contract for every checkpoint:
- format: `r27_pnp_omrr_core_v1`
- tensors: 7
- parameters: 11,906

## 3. Restormer prior asset

The formal output banks contain one frozen Restormer restoration output. The
~100 MB prior checkpoint is kept out of GitHub.

Place:

```text
assets/restormer/best_restormer_prior_woruie.pth
```

Expected SHA256:

```text
b2f6fa945708afff4ba58a1f0a3c4aab95965ef6086d646b3a5d4d6f73f4486c
```

**Download:** https://drive.google.com/file/d/1Blm76DUeL-MAgOYJzUYlvgzwBBVThNTE/view?usp=drive_link

See `assets/restormer/README.md` for provenance and upstream Restormer links.

## 4. End-to-end inference

Example:

```bash
python tools/infer.py \
  --dataset UIEBD \
  --input-dir /path/to/images \
  --output-dir results/uiebd \
  --restormer-weight assets/restormer/best_restormer_prior_woruie.pth \
  --device cuda:0 \
  --save-routing
```

Outputs:
- `results/uiebd/refined/`
- `results/uiebd/synthesis/`
- optional `routing_mean.json`

Use `--objective color_cast|visibility|balanced|underwater` to change the
requested objective without retraining.

## 5. Core-only inference from a precomputed restoration-output bank

This is the cleanest way to isolate the paper's post-synthesis mechanism and
does not require the Restormer model itself.

Prepare:

```text
bank_root/
  clahe/
  unsharp/
  grayworld/ or msrcr/ or udcp/
  restormer_woruie/
```

using the exact method names required by the selected dataset config, then run:

```bash
python tools/infer_from_bank.py \
  --dataset UIEBD \
  --input-dir /path/to/raw_images \
  --bank-root /path/to/bank_root \
  --output-dir results/uiebd_from_bank \
  --device cuda:0
```

Bank images are interpreted as sRGB files and converted to linear RGB before
OMRR refinement.

## 6. Formal retraining of the four multidataset runs

The checksum-frozen formal trainer used for UFO120/EUVP/RUIE/OceanDark is
retained unchanged. `tools/train.py` only replaces local filesystem roots.

```bash
python tools/train.py \
  --dataset RUIE \
  --data-root /path/to/RUIE/train \
  --restormer-weight assets/restormer/best_restormer_prior_woruie.pth \
  --device cuda:0
```

The reported RUIE checkpoint was selected at E190 from the E10..E200 formal
sequence; the other released multidataset checkpoints are E200.

The UIEBD released checkpoint is provided directly with exact extraction
provenance; see `docs/REPRODUCIBILITY.md`.

## 7. Frozen banks

See `docs/RESTORATION_BANKS.md`. The paper uses K=4 restoration outputs and an
M=2 parameter-free anchor for every formal dataset profile.

## 8. Mechanism evidence

Per-source tables for reachability and frozen-router realization are under:

```text
reproducibility/mechanism/
```

These tables allow the reported method-level and source-wise associations to
be recomputed without rerunning external UIE systems.

## 9. License and third-party code

The authors' original code in this repository is released under the **MIT License**; see `LICENSE`.

Restormer is third-party software. Its upstream source and license are linked
in `assets/restormer/README.md`.

## Repository

https://github.com/FengSY-JLU/Objective-Guided-Refinement-UIE

## Citation

See `CITATION.cff`. If you use this code or the released models, please cite the associated paper.
