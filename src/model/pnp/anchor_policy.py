import torch
import torch.nn as nn


class NamedMeanAnchorPolicy(nn.Module):
    """
    Parameter-free anchor selected by contributor identity,
    not by list position.
    """

    def __init__(
        self,
        contributor_names,
    ):
        super().__init__()

        names = tuple(
            str(x).strip()
            for x in contributor_names
        )

        if not names:
            raise ValueError(
                "Anchor requires >=1 contributor"
            )

        if len(set(names)) != len(names):
            raise ValueError(
                "Anchor contributor names must be unique"
            )

        self.contributor_names = names


    def forward(
        self,
        outputs,
    ):

        missing = [
            name
            for name in self.contributor_names
            if name not in outputs
        ]

        if missing:
            raise KeyError(
                "Anchor contributors missing: "
                f"{missing}"
            )

        selected = [
            outputs[name]
            for name in self.contributor_names
        ]

        # Use explicit add/divide rather than stack.mean so
        # the current two-contributor CR anchor reproduces:
        #
        #   (CLAHE + Restormer) / 2
        #
        anchor = selected[0]

        for value in selected[1:]:
            anchor = (
                anchor
                +
                value
            )

        anchor = (
            anchor
            /
            float(
                len(selected)
            )
        )

        return anchor
