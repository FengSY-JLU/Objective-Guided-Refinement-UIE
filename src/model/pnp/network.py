import torch.nn as nn


class PnPRefinementNet(nn.Module):
    """
    Contributor layer and refinement core are explicitly separated.

    The contributor bank may change cardinality without changing
    the OMRR parameterization.
    """

    def __init__(
        self,
        *,
        contributor_bank,
        anchor_policy,
        refinement_core,
    ):
        super().__init__()

        self.contributor_bank = (
            contributor_bank
        )

        self.anchor_policy = (
            anchor_policy
        )

        self.refinement_core = (
            refinement_core
        )


    def forward(
        self,
        x,
    ):

        outputs = (
            self.contributor_bank(
                x
            )
        )

        I_synthesis = (
            self.anchor_policy(
                outputs
            )
        )

        core_out = (
            self.refinement_core(
                I_input=x,
                I_synthesis=I_synthesis,
                contributor_outputs=outputs,
            )
        )

        names = tuple(
            outputs.keys()
        )

        prior_list = [
            outputs[name]
            for name in names
        ]

        residuals = [
            (
                P
                -
                x
            ).clamp(
                -1.0,
                1.0,
            )

            for P in prior_list
        ]


        return {
            "I_refined":
                core_out[
                    "I_refined"
                ],

            "I_synthesis":
                I_synthesis,

            "priors":
                prior_list,

            "prior_names":
                names,

            "contributor_outputs":
                outputs,

            "residuals_stage1":
                residuals,

            "offsets_stage2":
                core_out[
                    "weighted_values"
                ],

            "route_weights":
                core_out[
                    "route_weights"
                ],

            "como_out":
                core_out[
                    "diagnostics"
                ],
        }
