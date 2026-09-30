from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

from src.model.module.state_objective_reliability_router_v1 import (
    StateObjectiveReliabilityRouterV1,
)


class PnPOMRRRefinementCore(nn.Module):
    """
    Contributor-cardinality-agnostic refinement core.

    Scientific path:

        V_k = H_k - I_s

        {w_k} = OMRR(
            I,
            I_s,
            {V_k},
            {H_k},
            Q
        )

        D = sum_k w_k V_k

        rho = LowerMedian_k mean(|H_k - I_s|)

        s_proj = min(
            1,
            rho / (mean(|D|) + eps)
        )

        I_r = clip(
            I_s + gamma * s_proj * D,
            0,
            1
        )
    """

    VALID_OBJECTIVES = {
        "balanced",
        "color_cast",
        "visibility",
        "underwater",
    }


    def __init__(
        self,
        *,
        objective="balanced",
        goal_kappa=1.0,
        refinement_gamma=0.25,
        router_mode="direct_objective",
        eps=1e-8,
    ):
        super().__init__()

        self.router = (
            StateObjectiveReliabilityRouterV1()
        )

        self.router_mode = str(
            router_mode
        ).strip().lower()

        self.eps = float(
            eps
        )

        self.set_objective(
            objective,
            goal_kappa,
        )

        self.set_refinement_gamma(
            refinement_gamma
        )

        self.last_weights = None
        self.last_router_diag = None


    def set_objective(
        self,
        objective,
        goal_kappa=1.0,
    ):

        objective = str(
            objective
        ).strip().lower()

        if objective not in self.VALID_OBJECTIVES:
            raise ValueError(
                f"Unsupported objective: {objective}"
            )

        self.objective = objective
        self.goal_kappa = float(
            goal_kappa
        )

        return self


    def set_refinement_gamma(
        self,
        gamma,
    ):

        gamma = float(
            gamma
        )

        if not (
            0.0
            <=
            gamma
            <=
            1.0
        ):
            raise ValueError(
                "refinement gamma must be in [0,1]"
            )

        self.refinement_gamma = gamma

        return self


    def load_core_checkpoint(
        self,
        path,
    ):

        path = Path(
            path
        )

        obj = torch.load(
            path,
            map_location="cpu",
        )

        if not (
            isinstance(obj, dict)
            and
            "state_dict" in obj
        ):
            raise RuntimeError(
                "Invalid OMRR core checkpoint"
            )

        msg = self.router.load_state_dict(
            obj["state_dict"],
            strict=True,
        )

        return msg


    @staticmethod
    def _trust_radius(
        I_synthesis,
        values,
    ):

        support_distances = torch.stack(
            [
                (
                    value
                    -
                    I_synthesis
                )
                .abs()
                .mean(
                    dim=[1, 2, 3]
                )

                for value in values
            ],
            dim=1,
        )

        # PyTorch median = lower median for even K.
        return torch.median(
            support_distances,
            dim=1,
        ).values


    def forward(
        self,
        *,
        I_input,
        I_synthesis,
        contributor_outputs,
    ):

        if not isinstance(
            contributor_outputs,
            (dict, OrderedDict),
        ):
            raise TypeError(
                "contributor_outputs must be mapping"
            )

        names = tuple(
            contributor_outputs.keys()
        )

        if len(names) == 0:
            raise ValueError(
                "At least one contributor is required"
            )

        priors = [
            contributor_outputs[name]
            for name in names
        ]


        direct_values = [
            (
                P
                -
                I_synthesis
            ).clamp(
                -1.0,
                1.0,
            )

            for P in priors
        ]


        (
            weighted_values,
            weights,
            router_diag,
        ) = self.router(
            I_input=I_input,
            I_synthesis=I_synthesis,
            value_list=direct_values,
            priors=priors,
            mode=self.router_mode,
            objective_mode=self.objective,
            goal_kappa=self.goal_kappa,
        )


        aggregated_evidence = torch.stack(
            weighted_values,
            dim=0,
        ).sum(
            dim=0
        )


        I_candidate = (
            I_synthesis
            +
            aggregated_evidence
        )


        trust_radius = self._trust_radius(
            I_synthesis,
            priors,
        )


        candidate_radius = (
            aggregated_evidence
            .abs()
            .mean(
                dim=[1, 2, 3]
            )
        )


        projection_scale = torch.minimum(
            torch.ones_like(
                candidate_radius
            ),
            trust_radius
            /
            (
                candidate_radius
                +
                self.eps
            ),
        )


        execution_scale = (
            float(
                self.refinement_gamma
            )
            *
            projection_scale
        )


        scale_map = (
            execution_scale
            .view(
                -1,
                1,
                1,
                1,
            )
        )


        I_refined = torch.clamp(
            I_synthesis
            +
            scale_map
            *
            aggregated_evidence,
            0.0,
            1.0,
        )


        executed_radius = (
            I_refined
            -
            I_synthesis
        ).abs().mean(
            dim=[1, 2, 3]
        )


        diagnostics = {
            "Iy":
                I_candidate,

            "I_candidate":
                I_candidate,

            "aggregated_evidence":
                aggregated_evidence,

            "trust_radius":
                trust_radius,

            "candidate_radius":
                candidate_radius,

            "projection_scale":
                projection_scale,

            "execution_scale":
                execution_scale,

            "executed_radius":
                executed_radius,

            "refinement_beta":
                float(
                    self.refinement_gamma
                ),

            "refinement_gamma":
                float(
                    self.refinement_gamma
                ),

            "execution_mode":
                "trust_region",

            "coordinator_backend":
                "direct_value",

            "route_feature_mode":
                "direct",

            "router_backend":
                "sorr_v1",

            "router_mode":
                self.router_mode,

            "contributor_names":
                names,
        }

        if router_diag:
            diagnostics.update(
                router_diag
            )


        self.last_weights = weights
        self.last_router_diag = (
            router_diag
        )


        return {
            "I_refined":
                I_refined,

            "weighted_values":
                weighted_values,

            "route_weights":
                weights,

            "diagnostics":
                diagnostics,
        }
