import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.losses_goal_v3 import (
    cast_score,
    local_contrast_score,
    edge_visibility_score,
    supported_scale,
    cauchy_penalty,
)

from src.model.losses_goal_v4 import (
    chromatic_spread_score,
)


class StateObjectiveReliabilityRouterV1(nn.Module):
    """
    State-Objective Reliability Router (SORR-v1).

    Inputs
    ------
    I_input:
        degraded input I

    I_synthesis:
        current synthesis state I_s

    value_list:
        explicit contributor corrections
            V_k = H_k - I_s

    priors:
        complete restoration outputs H_k

    Principle
    ---------
    Learning determines contributor reliability.

    The actual executed correction remains the explicit
    restoration displacement V_k.

    Objective relevance is a bounded contributor-relative
    modulation and never replaces spatial reliability.
    """

    VALID_MODES = {
        "direct_only",
        "direct_objective",
        "state_only",
        "state_objective",
    }

    VALID_OBJECTIVES = {
        "balanced",
        "color_cast",
        "visibility",
        "underwater",
    }

    def __init__(
        self,
        hidden=32,
        tau_max=2.0,
        tau_init=0.5,
    ):
        super().__init__()

        self.hidden = int(hidden)
        self.tau_max = float(tau_max)

        if not (
            0.0
            <
            float(tau_init)
            <
            self.tau_max
        ):
            raise ValueError(
                "tau_init must satisfy "
                f"0 < tau_init < {self.tau_max}"
            )

        # Shared across all contributors.
        #
        # [I, I_s, V_k] -> local reliability feature
        self.encoder = nn.Sequential(
            nn.Conv2d(
                9,
                self.hidden,
                3,
                padding=1,
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Conv2d(
                self.hidden,
                self.hidden,
                3,
                padding=1,
            ),
            nn.ReLU(
                inplace=True
            ),
        )

        self.score_head = nn.Conv2d(
            self.hidden,
            1,
            kernel_size=1,
        )

        p = (
            float(tau_init)
            /
            self.tau_max
        )

        raw = math.log(
            p / (1.0 - p)
        )

        self.raw_tau = nn.Parameter(
            torch.tensor(
                raw,
                dtype=torch.float32,
            )
        )

        self.last_regret = None
        self.last_tau = None
        self.last_base_scores = None


    def objective_tau(self):
        return (
            self.tau_max
            *
            torch.sigmoid(
                self.raw_tau
            )
        )


    @staticmethod
    def _trust_radius(
        I_synthesis,
        priors,
    ):
        support_distances = torch.stack(
            [
                (
                    P.detach()
                    -
                    I_synthesis.detach()
                )
                .abs()
                .mean(
                    dim=[1, 2, 3]
                )
                for P in priors
            ],
            dim=1,
        )

        return torch.median(
            support_distances,
            dim=1,
        ).values.detach()


    @staticmethod
    def _execute_single(
        I_synthesis,
        value,
        trust_radius,
        eps=1e-8,
    ):
        radius = (
            value.detach()
            .abs()
            .mean(
                dim=[1, 2, 3]
            )
        )

        scale = torch.minimum(
            torch.ones_like(
                radius
            ),
            trust_radius
            /
            (
                radius
                + eps
            ),
        )

        return torch.clamp(
            I_synthesis.detach()
            +
            scale.view(
                -1, 1, 1, 1
            )
            *
            value.detach(),
            0.0,
            1.0,
        )


    @staticmethod
    def _component_context(
        I_synthesis,
        priors,
        fn,
        component,
        kappa,
    ):
        syn = I_synthesis.detach()

        q_syn = fn(
            syn
        )

        scale = supported_scale(
            fn,
            syn,
            priors,
        )

        if component == "cast":

            target = torch.minimum(
                torch.zeros_like(
                    q_syn
                ),
                q_syn
                +
                float(kappa)
                *
                scale,
            )

        else:

            target = (
                q_syn
                +
                float(kappa)
                *
                scale
            )

        return (
            target.detach(),
            scale.detach(),
        )


    def contributor_regret(
        self,
        I_synthesis,
        value_list,
        priors,
        objective_mode,
        kappa,
    ):
        """
        Returns detached B x K regret in [0,1].

        No GT, labels or post-hoc oracle information is used.
        Only synthesis state, contributor outputs and the
        explicit Goal-V4 definition are involved.
        """

        mode = str(
            objective_mode
        ).strip().lower()

        if mode not in self.VALID_OBJECTIVES:
            raise ValueError(
                f"Unsupported objective mode: {mode}"
            )

        with torch.no_grad():

            rho = self._trust_radius(
                I_synthesis,
                priors,
            )

            components = []

            if mode in {
                "balanced",
                "color_cast",
                "underwater",
            }:
                components.append(
                    (
                        "cast",
                        cast_score,
                    )
                )

            if mode == "underwater":
                components.append(
                    (
                        "chromatic_spread",
                        chromatic_spread_score,
                    )
                )

            if mode in {
                "balanced",
                "visibility",
                "underwater",
            }:
                components.extend(
                    [
                        (
                            "contrast",
                            local_contrast_score,
                        ),
                        (
                            "edge",
                            edge_visibility_score,
                        ),
                    ]
                )

            context = {}

            for (
                component,
                fn,
            ) in components:

                target, scale = (
                    self._component_context(
                        I_synthesis,
                        priors,
                        fn,
                        component,
                        kappa,
                    )
                )

                context[
                    component
                ] = (
                    fn,
                    target,
                    scale,
                )

            candidate_losses = []

            for value in value_list:

                Rk = self._execute_single(
                    I_synthesis,
                    value,
                    rho,
                )

                per_component = []

                for (
                    component,
                    _,
                ) in components:

                    (
                        fn,
                        target,
                        scale,
                    ) = context[
                        component
                    ]

                    q = fn(
                        Rk
                    )

                    z = (
                        q
                        - target
                    ) / (
                        scale
                        + 1e-6
                    )

                    per_component.append(
                        cauchy_penalty(
                            z
                        )
                    )

                candidate_losses.append(
                    torch.stack(
                        per_component,
                        dim=1,
                    ).mean(
                        dim=1
                    )
                )

            # B x K
            loss_matrix = torch.stack(
                candidate_losses,
                dim=1,
            )

            vmin = loss_matrix.min(
                dim=1,
                keepdim=True,
            ).values

            vmax = loss_matrix.max(
                dim=1,
                keepdim=True,
            ).values

            span = (
                vmax
                - vmin
            )

            regret = (
                loss_matrix
                - vmin
            ) / (
                span
                + 1e-6
            )

            regret = torch.where(
                span < 1e-8,
                torch.zeros_like(
                    regret
                ),
                regret,
            )

            regret = regret.clamp(
                0.0,
                1.0,
            )

        return (
            regret.detach(),
            loss_matrix.detach(),
        )


    def forward(
        self,
        I_input,
        I_synthesis,
        value_list,
        priors,
        mode="state_objective",
        objective_mode="balanced",
        goal_kappa=1.0,
    ):
        mode = str(
            mode
        ).strip().lower()

        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unsupported SORR mode: {mode}; "
                f"expected {sorted(self.VALID_MODES)}"
            )

        if len(value_list) == 0:
            raise ValueError(
                "value_list must be non-empty"
            )

        if len(value_list) != len(priors):
            raise ValueError(
                "value_list and priors must "
                "have equal length"
            )

        scores = []

        for value in value_list:

            if mode in {
                "direct_only",
                "direct_objective",
            }:

                zero_I = torch.zeros_like(
                    I_input
                )

                zero_S = torch.zeros_like(
                    I_synthesis
                )

                feature = torch.cat(
                    [
                        zero_I,
                        zero_S,
                        value,
                    ],
                    dim=1,
                )

            else:

                feature = torch.cat(
                    [
                        I_input,
                        I_synthesis,
                        value,
                    ],
                    dim=1,
                )

            h = self.encoder(
                feature
            )

            scores.append(
                self.score_head(
                    h
                )
            )

        base_scores = torch.cat(
            scores,
            dim=1,
        )

        if mode in {
            "direct_objective",
            "state_objective",
        }:

            regret, loss_matrix = (
                self.contributor_regret(
                    I_synthesis,
                    value_list,
                    priors,
                    objective_mode,
                    goal_kappa,
                )
            )

            tau = self.objective_tau()

            score_map = (
                base_scores
                -
                tau
                *
                regret[
                    :,
                    :,
                    None,
                    None,
                ]
            )

        else:

            B = base_scores.shape[0]
            K = base_scores.shape[1]

            regret = torch.zeros(
                B,
                K,
                dtype=base_scores.dtype,
                device=base_scores.device,
            )

            loss_matrix = torch.zeros_like(
                regret
            )

            tau = self.objective_tau()

            score_map = base_scores

        weights = F.softmax(
            score_map,
            dim=1,
        )

        weighted_values = [
            weights[:, k:k+1]
            *
            value_list[k]
            for k in range(
                len(value_list)
            )
        ]

        self.last_regret = regret
        self.last_tau = tau
        self.last_base_scores = base_scores

        diagnostics = {
            "router_backend":
                "sorr_v1",

            "router_mode":
                mode,

            "router_tau":
                tau.detach(),

            "router_regret_mean":
                regret.detach().mean(),

            "router_candidate_loss_mean":
                loss_matrix.detach().mean(),

            "router_entropy":
                (
                    -(
                        weights.clamp_min(
                            1e-8
                        )
                        *
                        weights.clamp_min(
                            1e-8
                        ).log()
                    )
                    .sum(
                        dim=1
                    )
                    .mean()
                ).detach(),
        }

        return (
            weighted_values,
            weights,
            diagnostics,
        )
