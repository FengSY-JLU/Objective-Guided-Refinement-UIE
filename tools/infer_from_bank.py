#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.public_api import build_core_only
from src.io_utils import load_rgb, save_rgb

EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Apply the released post-synthesis OMRR core to precomputed "
            "complete restoration outputs. Bank images are interpreted as sRGB."
        )
    )
    ap.add_argument("--dataset", required=True,
                    choices=["UIEBD", "UFO120", "RUIE", "OceanDark", "EUVP"])
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--bank-root", required=True,
                    help="Contains one subdirectory per formal restoration output.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--objective",
                    choices=["balanced", "color_cast", "visibility", "underwater"],
                    default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    profile = json.loads(
        (ROOT / "configs" / f"{args.dataset.lower()}.json").read_text()
    )
    core = build_core_only(
        profile, ROOT/profile["checkpoint"], device, objective=args.objective
    )

    input_dir = Path(args.input_dir)
    bank_root = Path(args.bank_root)
    out_root = Path(args.output_dir)

    files = sorted(p for p in input_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in EXTS)
    if not files:
        raise RuntimeError(f"No images found in {input_dir}")

    for name in profile["slot_order"]:
        d = bank_root / name
        if not d.is_dir():
            raise FileNotFoundError(
                f"Missing restoration-output directory: {d}"
            )

    with torch.inference_mode():
        for p in files:
            x = load_rgb(p, device=device)
            outputs = OrderedDict(
                (name, load_rgb(bank_root/name/p.name, device=device))
                for name in profile["slot_order"]
            )
            selected = [outputs[name] for name in profile["anchor_methods"]]
            I_s = selected[0]
            for value in selected[1:]:
                I_s = I_s + value
            I_s = I_s / float(len(selected))

            out = core(
                I_input=x,
                I_synthesis=I_s,
                contributor_outputs=outputs,
            )
            save_rgb(out_root/"refined"/p.name, out["I_refined"])
            save_rgb(out_root/"synthesis"/p.name, I_s)

    print(f"PASS_BANK_INFERENCE count={len(files)} output={out_root}")


if __name__ == "__main__":
    main()
