import hashlib
import json
from pathlib import Path

import torch


SOURCE = Path(
    "checkpoints/"
    "R27FORMAL_UIEBD_CRUG_OMRR_B025_W0020_S3407/"
    "model_epoch_200.pth"
)

OUT = Path(
    "checkpoints/"
    "r27f6_pnp_core_v1/"
    "omrr_core_from_e200_v1.pth"
)

REPORT = Path(
    "audits/"
    "r27f6_pnp_core_export_v1/"
    "omrr_core_export_report_v1.json"
)

EXPECTED_SOURCE_SHA = (
    "2e412ffeadd1bcbae3fa54ebf441ee85"
    "ef218f0e58e61c595e118674f233bbe2"
)

PREFIX = "stage2.sorr."

EXPECTED_KEYS = {
    "encoder.0.weight",
    "encoder.0.bias",
    "encoder.2.weight",
    "encoder.2.bias",
    "raw_tau",
    "score_head.weight",
    "score_head.bias",
}


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def main():

    assert SOURCE.is_file(), SOURCE

    source_sha = sha256(SOURCE)

    assert source_sha == EXPECTED_SOURCE_SHA, (
        source_sha
    )

    obj = torch.load(
        SOURCE,
        map_location="cpu",
    )

    assert isinstance(obj, dict)
    assert "state_dict" in obj

    full_sd = obj["state_dict"]

    core_sd = {
        k[len(PREFIX):]: v.detach().cpu()
        for k, v in full_sd.items()
        if k.startswith(PREFIX)
    }

    assert set(core_sd) == EXPECTED_KEYS, (
        sorted(core_sd)
    )

    total_params = sum(
        int(v.numel())
        for v in core_sd.values()
    )

    assert total_params == 11906, (
        total_params
    )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_obj = {
        "format": "r27_omrr_core_v1",

        "state_dict": core_sd,

        "metadata": {
            "source_checkpoint": str(SOURCE),
            "source_checkpoint_sha256": source_sha,
            "source_epoch": 200,
            "router": "OMRR/SORR-v1",
            "parameter_count": total_params,
            "scientific_core_changed": False,
            "external_contributors_included": False,
            "legacy_cose_included": False,
            "legacy_cas_included": False,
            "legacy_como_included": False,
        },
    }

    torch.save(
        save_obj,
        OUT,
    )

    out_sha = sha256(OUT)

    reload_obj = torch.load(
        OUT,
        map_location="cpu",
    )

    assert (
        reload_obj["metadata"]["parameter_count"]
        ==
        11906
    )

    reload_sd = reload_obj["state_dict"]

    assert set(reload_sd) == EXPECTED_KEYS

    for key in EXPECTED_KEYS:
        assert torch.equal(
            reload_sd[key],
            core_sd[key],
        )

    report = {
        "source_checkpoint": str(SOURCE),
        "source_sha256": source_sha,
        "output_checkpoint": str(OUT),
        "output_sha256": out_sha,
        "parameter_count": total_params,
        "tensor_count": len(core_sd),
        "keys": sorted(core_sd),
        "exact_tensor_roundtrip": True,
        "test190_used": False,
    }

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print("R27F6 OMRR CORE EXPORT V1")
    print("=" * 100)

    print("SOURCE_SHA =", source_sha)
    print("CORE_SHA   =", out_sha)
    print("TENSOR_COUNT =", len(core_sd))
    print("PARAMETER_COUNT =", total_params)

    for key in sorted(core_sd):
        print(
            key,
            tuple(core_sd[key].shape),
            core_sd[key].numel(),
        )

    print()
    print("CORE_CHECKPOINT =", OUT)
    print("REPORT =", REPORT)
    print("TEST_USED=False")
    print("PASS_R27F6_OMRR_CORE_EXPORT_V1")
    print("TERMINAL_SURVIVED")


if __name__ == "__main__":
    main()
