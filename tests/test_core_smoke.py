import json
from collections import OrderedDict
from pathlib import Path

import torch

from src.public_api import build_core_only

ROOT = Path(__file__).resolve().parents[1]


def test_core_smoke():
    profile = json.loads((ROOT/"configs/uiebd.json").read_text())
    core = build_core_only(
        profile,
        ROOT/profile["checkpoint"],
        torch.device("cpu"),
    )

    torch.manual_seed(3407)
    x = torch.rand(1, 3, 32, 32)
    outputs = OrderedDict()
    for i, name in enumerate(profile["slot_order"]):
        outputs[name] = (x + (i - 1.5) * 0.01).clamp(0, 1)

    selected = [outputs[n] for n in profile["anchor_methods"]]
    anchor = sum(selected) / float(len(selected))

    with torch.inference_mode():
        out = core(
            I_input=x,
            I_synthesis=anchor,
            contributor_outputs=outputs,
        )

    assert out["I_refined"].shape == x.shape
    assert torch.isfinite(out["I_refined"]).all()
    w = out["route_weights"]
    assert torch.max(torch.abs(w.sum(dim=1) - 1.0)).item() < 1e-6
