from collections import OrderedDict

import torch
import torch.nn as nn


class ContributorBank(nn.Module):
    """
    Named plug-and-play contributor bank.

    Contract
    --------
    input:
        BCHW linear RGB in [0,1]

    output:
        OrderedDict[name -> BCHW linear RGB in [0,1]]

    Contributors are frozen by default and kept in eval mode.
    """

    def __init__(
        self,
        contributors,
        *,
        freeze=True,
    ):
        super().__init__()

        if not contributors:
            raise ValueError(
                "ContributorBank requires at least one contributor"
            )

        self.contributors = nn.ModuleDict()
        self._frozen = bool(freeze)

        for name, module in contributors.items():
            self.add(
                name,
                module,
                freeze=freeze,
            )


    @staticmethod
    def _validate_name(name):

        name = str(name).strip()

        if not name:
            raise ValueError(
                "Contributor name must be non-empty"
            )

        if "." in name:
            raise ValueError(
                "Contributor name must not contain '.'"
            )

        return name


    @staticmethod
    def _freeze_module(module):

        module.eval()

        for p in module.parameters():
            p.requires_grad_(False)


    def add(
        self,
        name,
        module,
        *,
        freeze=True,
        replace=False,
    ):

        name = self._validate_name(
            name
        )

        if (
            name in self.contributors
            and
            not replace
        ):
            raise KeyError(
                f"Contributor already exists: {name}"
            )

        if not isinstance(
            module,
            nn.Module,
        ):
            raise TypeError(
                "Contributor must be nn.Module"
            )

        if freeze:
            self._freeze_module(
                module
            )

        self.contributors[name] = module

        return self


    def remove(
        self,
        name,
    ):

        name = self._validate_name(
            name
        )

        if name not in self.contributors:
            raise KeyError(
                f"Unknown contributor: {name}"
            )

        # Transactional mutation contract:
        # refuse the operation BEFORE changing bank state.
        if len(self.contributors) <= 1:
            raise RuntimeError(
                "ContributorBank cannot become empty"
            )

        del self.contributors[name]

        return self


    @property
    def names(self):

        return tuple(
            self.contributors.keys()
        )


    def train(
        self,
        mode=True,
    ):

        super().train(
            mode
        )

        if self._frozen:
            for module in self.contributors.values():
                module.eval()

        return self


    def forward(
        self,
        x,
    ):

        if (
            x.ndim != 4
            or
            x.shape[1] != 3
        ):
            raise ValueError(
                "ContributorBank expects BCHW RGB input"
            )

        outputs = OrderedDict()

        for name, module in self.contributors.items():

            y = module(
                x
            )

            if not torch.is_tensor(
                y
            ):
                raise TypeError(
                    f"{name} did not return a tensor"
                )

            # Exact legacy Stage-I compatibility:
            # contributor implementations may internally
            # produce CPU tensors. The historical synthesis
            # path moved every contributor output onto the
            # input device before synthesis/refinement.
            #
            # Deliberately preserve dtype here; legacy code
            # used Pi.to(device), not Pi.to(device, dtype).
            y = y.to(
                x.device
            )

            if y.shape != x.shape:
                raise RuntimeError(
                    f"{name} geometry mismatch: "
                    f"{tuple(y.shape)} vs {tuple(x.shape)}"
                )

            if not torch.isfinite(
                y
            ).all():
                raise RuntimeError(
                    f"{name} produced non-finite output"
                )

            outputs[name] = y

        return outputs
