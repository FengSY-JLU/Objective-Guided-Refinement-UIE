#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.train_r27f9g4d_multidataset_pnp_formal_v1 as formal


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Path-configurable wrapper around the checksum-frozen formal "
            "multidataset trainer used for UFO120/EUVP/RUIE/OceanDark."
        )
    )
    ap.add_argument("--dataset", required=True,
                    choices=["UFO120", "EUVP", "RUIE", "OceanDark"])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--restormer-weight", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--auto-resume", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    restormer = Path(args.restormer_weight).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    if not restormer.is_file():
        raise FileNotFoundError(restormer)

    formal.DATASETS[args.dataset]["root"] = data_root
    formal.RESTORMER_PATH = restormer

    argv = [
        "train_r27f9g4d_multidataset_pnp_formal_v1.py",
        "--dataset", args.dataset,
        "--device", args.device,
    ]
    if args.auto_resume:
        argv.append("--auto_resume")

    old = sys.argv
    try:
        sys.argv = argv
        formal.main()
    finally:
        sys.argv = old


if __name__ == "__main__":
    main()
