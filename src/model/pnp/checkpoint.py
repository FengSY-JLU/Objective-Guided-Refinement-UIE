import hashlib
from pathlib import Path

import torch


FORMAT = "r27_pnp_omrr_core_v1"


def sha256(path):
    path = Path(path)

    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def core_parameter_count(
    model,
):
    return sum(
        p.numel()
        for p in
        model.refinement_core.router.parameters()
    )


def save_core_checkpoint(
    model,
    path,
    *,
    metadata=None,
):
    """
    Save ONLY the contributor-agnostic OMRR core.

    Contributor modules and external enhancer parameters are
    intentionally excluded.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    router = (
        model
        .refinement_core
        .router
    )

    state_dict = {
        k:
            v.detach().cpu()
        for k, v in
        router.state_dict().items()
    }

    n_params = sum(
        int(v.numel())
        for v in state_dict.values()
    )

    if n_params != 11906:
        raise RuntimeError(
            "Unexpected OMRR parameter count: "
            f"{n_params}"
        )

    user_metadata = dict(
        metadata or {}
    )

    obj = {
        "format":
            FORMAT,

        "state_dict":
            state_dict,

        "metadata": {
            "core_parameter_count":
                n_params,

            "external_contributors_included":
                False,

            "contributor_cardinality_specific":
                False,

            **user_metadata,
        },
    }

    torch.save(
        obj,
        path,
    )

    return {
        "path":
            str(path),

        "sha256":
            sha256(path),

        "parameter_count":
            n_params,

        "tensor_count":
            len(state_dict),
    }


def load_core_checkpoint(
    model,
    path,
    *,
    strict=True,
):
    path = Path(path)

    obj = torch.load(
        path,
        map_location="cpu",
    )

    if not isinstance(
        obj,
        dict,
    ):
        raise RuntimeError(
            "Invalid PnP core checkpoint"
        )

    if obj.get(
        "format"
    ) != FORMAT:
        raise RuntimeError(
            "Unexpected checkpoint format: "
            f"{obj.get('format')}"
        )

    if "state_dict" not in obj:
        raise RuntimeError(
            "Missing core state_dict"
        )

    msg = (
        model
        .refinement_core
        .router
        .load_state_dict(
            obj[
                "state_dict"
            ],
            strict=strict,
        )
    )

    return msg
