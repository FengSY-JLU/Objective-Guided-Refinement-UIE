#!/usr/bin/env python3
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest = json.loads(
        (ROOT/"checkpoints/CHECKPOINT_MANIFEST.json").read_text()
    )
    for ds, entry in manifest.items():
        p = ROOT/"checkpoints"/entry["file"]
        assert p.is_file(), p
        assert sha256(p) == entry["sha256"], (ds, sha256(p), entry["sha256"])
        obj = torch.load(p, map_location="cpu")
        assert obj["format"] == "r27_pnp_omrr_core_v1"
        assert len(obj["state_dict"]) == 7
        n = sum(int(v.numel()) for v in obj["state_dict"].values())
        assert n == 11906
        print(f"PASS {ds}: tensors=7 params=11906 sha256={entry['sha256']}")
    print("PASS_RELEASE_CHECKPOINT_CONTRACT")


if __name__ == "__main__":
    main()
