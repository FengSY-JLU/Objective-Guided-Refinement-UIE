from collections import OrderedDict
from pathlib import Path

import torch

from src.model.priors import default_priors
from src.model.pnp.contributor_bank import ContributorBank
from src.model.pnp.anchor_policy import NamedMeanAnchorPolicy
from src.model.pnp.omrr_refinement_core import PnPOMRRRefinementCore
from src.model.pnp.network import PnPRefinementNet
from src.model.pnp.checkpoint import load_core_checkpoint


def build_restoration_operator(name, device, restormer_weight=None):
    name = str(name).strip().lower()

    if name == "restormer_woruie":
        if restormer_weight is None:
            raise FileNotFoundError(
                "The formal bank requires the external Restormer prior weight. "
                "Pass --restormer-weight; see assets/restormer/README.md."
            )
        restormer_weight = Path(restormer_weight)
        if not restormer_weight.is_file():
            raise FileNotFoundError(restormer_weight)

        modules = default_priors(
            device=device,
            learned_prior_cfgs=[{
                "type": "restormer",
                "path": str(restormer_weight),
                "name": "restormer_woruie",
            }],
            handcrafted_prior_names=(),
        )
    else:
        modules = default_priors(
            device=device,
            learned_prior_cfgs=None,
            handcrafted_prior_names=(name,),
        )

    if len(modules) != 1:
        raise RuntimeError(f"Expected one restoration operator for {name}")
    return modules[0]


def build_model(profile, checkpoint, device, restormer_weight=None,
                objective=None, goal_kappa=None):
    contributors = OrderedDict(
        (name, build_restoration_operator(
            name, device, restormer_weight=restormer_weight
        ))
        for name in profile["slot_order"]
    )

    bank = ContributorBank(contributors, freeze=True)
    anchor = NamedMeanAnchorPolicy(profile["anchor_methods"])
    core = PnPOMRRRefinementCore(
        objective=objective or profile.get("objective_mode", "balanced"),
        goal_kappa=float(
            profile.get("goal_kappa", 1.0)
            if goal_kappa is None else goal_kappa
        ),
        refinement_gamma=float(profile.get("refinement_gamma", 0.25)),
        router_mode="direct_objective",
    )
    model = PnPRefinementNet(
        contributor_bank=bank,
        anchor_policy=anchor,
        refinement_core=core,
    ).to(device)

    load_core_checkpoint(model, checkpoint, strict=True)
    model.eval()
    return model


def build_core_only(profile, checkpoint, device, objective=None, goal_kappa=None):
    core = PnPOMRRRefinementCore(
        objective=objective or profile.get("objective_mode", "balanced"),
        goal_kappa=float(
            profile.get("goal_kappa", 1.0)
            if goal_kappa is None else goal_kappa
        ),
        refinement_gamma=float(profile.get("refinement_gamma", 0.25)),
        router_mode="direct_objective",
    ).to(device)

    obj = torch.load(checkpoint, map_location="cpu")
    if obj.get("format") != "r27_pnp_omrr_core_v1":
        raise RuntimeError(f"Unexpected checkpoint format: {obj.get('format')}")
    core.router.load_state_dict(obj["state_dict"], strict=True)
    core.eval()
    return core
