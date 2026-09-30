#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.public_api import build_model
from src.io_utils import load_rgb, save_rgb

EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def main():
    ap = argparse.ArgumentParser(
        description="End-to-end inference for Objective-Guided Refinement."
    )
    ap.add_argument("--dataset", required=True,
                    choices=["UIEBD", "UFO120", "RUIE", "OceanDark", "EUVP"])
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--restormer-weight", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--objective",
                    choices=["balanced", "color_cast", "visibility", "underwater"],
                    default=None)
    ap.add_argument("--save-routing", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    profile = json.loads(
        (ROOT / "configs" / f"{args.dataset.lower()}.json").read_text()
    )
    ckpt = ROOT / profile["checkpoint"]

    model = build_model(
        profile, ckpt, device,
        restormer_weight=args.restormer_weight,
        objective=args.objective,
    )

    input_dir = Path(args.input_dir)
    out_root = Path(args.output_dir)
    files = sorted(p for p in input_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in EXTS)
    if not files:
        raise RuntimeError(f"No images found in {input_dir}")

    routing_summary = {}
    with torch.inference_mode():
        for p in files:
            x = load_rgb(p, device=device)
            out = model(x)
            save_rgb(out_root / "refined" / p.name, out["I_refined"])
            save_rgb(out_root / "synthesis" / p.name, out["I_synthesis"])

            if args.save_routing:
                w = out["route_weights"].mean(dim=(0, 2, 3)).detach().cpu().tolist()
                routing_summary[p.name] = dict(
                    zip(out["prior_names"], [float(v) for v in w])
                )

    if args.save_routing:
        (out_root / "routing_mean.json").write_text(
            json.dumps(routing_summary, indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"PASS_INFERENCE count={len(files)} output={out_root}")


if __name__ == "__main__":
    main()
