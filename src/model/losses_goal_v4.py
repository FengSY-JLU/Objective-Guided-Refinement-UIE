"""
Goal-V4: explicit Multi-Q objective family.

Scientific contract
-------------------
1. V3 behavior is preserved exactly for:
       color_cast
       visibility
       balanced

2. Underwater-Q adds exactly one new positive attribute:
       chromatic_spread

3. Contributor outputs determine attainable transition scale.
   They do not determine the desired objective direction.

4. Safety remains objective-independent.

5. This module introduces no trainable parameters.
"""

import torch

from src.model.losses_goal_v3 import (
    rgb_to_lab,
    cast_score,
    local_contrast_score,
    edge_visibility_score,
    supported_scale,
    target_loss_and_utility,
    chroma_safety_loss,
    clipping_safety_loss,
    compute_goal_objective_v3,
)


# ============================================================
# New V4 descriptor
# ============================================================

def chromatic_spread_score(img):
    """
    Spatial chromatic dispersion in Lab space.

    Higher = greater chromatic diversity.

    Unlike V3 chroma_score, this descriptor is centered by the
    per-image Lab chromatic mean. Therefore a spatially uniform
    color cast does not receive a large positive score merely
    because its chroma magnitude is high.

        m_cspread =
            sqrt(
                Var_pop(a) + Var_pop(b) + eps
            )

    Population variance is implemented explicitly for fully
    deterministic semantics across tensor shapes.
    """
    lab = rgb_to_lab(img)

    a = lab[:, 1]
    b = lab[:, 2]

    ma = a.mean(
        dim=[1, 2],
        keepdim=True,
    )

    mb = b.mean(
        dim=[1, 2],
        keepdim=True,
    )

    va = (
        (a - ma) ** 2
    ).mean(
        dim=[1, 2]
    )

    vb = (
        (b - mb) ** 2
    ).mean(
        dim=[1, 2]
    )

    return torch.sqrt(
        va + vb + 1e-8
    )


# ============================================================
# Goal-V4
# ============================================================

def compute_goal_objective_v4(
    I_ref,
    I_syn,
    priors,
    mode="balanced",
    kappa=1.0,
):
    """
    Explicit Multi-Q objective family.

    Modes
    -----
    color_cast:
        cast correction

    visibility:
        local contrast + edge visibility

    balanced:
        cast correction + local contrast + edge visibility

    underwater:
        cast correction
        + chromatic spread
        + local contrast
        + edge visibility

    Existing V3 modes are delegated directly to V3 so their
    numerical behavior is preserved exactly.
    """
    mode = str(
        mode
    ).strip().lower()

    old_modes = {
        "balanced",
        "color_cast",
        "visibility",
    }

    if mode in old_modes:
        return compute_goal_objective_v3(
            I_ref=I_ref,
            I_syn=I_syn,
            priors=priors,
            mode=mode,
            kappa=kappa,
        )

    if mode != "underwater":
        raise ValueError(
            f"Unsupported Goal-V4 mode: {mode}; "
            "expected one of "
            "['balanced', 'color_cast', "
            "'underwater', 'visibility']"
        )

    syn = I_syn.detach()

    goal_losses = []
    goal_utils = []

    components = [
        (
            "cast",
            cast_score,
        ),
        (
            "chromatic_spread",
            chromatic_spread_score,
        ),
        (
            "contrast",
            local_contrast_score,
        ),
        (
            "edge",
            edge_visibility_score,
        ),
    ]

    for component, fn in components:

        q_syn = fn(
            syn
        )

        q_ref = fn(
            I_ref
        )

        scale = supported_scale(
            fn,
            syn,
            priors,
        )

        if component == "cast":

            # Cast score has theoretical optimum 0.
            target = torch.minimum(
                torch.zeros_like(
                    q_syn
                ),
                q_syn
                + float(kappa)
                * scale,
            )

        else:

            target = (
                q_syn
                + float(kappa)
                * scale
            )

        loss, utility = (
            target_loss_and_utility(
                q_ref=q_ref,
                q_syn=q_syn,
                target=target,
                scale=scale,
            )
        )

        goal_losses.append(
            loss
        )

        goal_utils.append(
            utility
        )

    # Equal weighting is meaningful because every active
    # component is normalized by its contributor-supported
    # per-sample scale.
    goal_loss = torch.stack(
        goal_losses,
        dim=1,
    ).mean()

    goal_utility = torch.stack(
        goal_utils,
        dim=1,
    ).mean()

    # --------------------------------------------------------
    # Common safety bundle
    # --------------------------------------------------------

    # Important:
    # V3 chroma safety remains unchanged. Positive chromatic
    # spread does NOT imply unlimited chroma magnitude.
    chroma_safe = chroma_safety_loss(
        I_ref,
        syn,
        priors,
    )

    clip_safe = clipping_safety_loss(
        I_ref,
        syn,
        priors,
    )

    safety_loss = 0.5 * (
        chroma_safe
        + clip_safe
    )

    objective_loss = 0.5 * (
        goal_loss
        + safety_loss
    )

    return {
        "objective_loss":
            objective_loss,

        "goal_loss":
            goal_loss,

        "goal_utility":
            goal_utility,

        "chroma_safe":
            chroma_safe,

        "clip_safe":
            clip_safe,

        "safety_loss":
            safety_loss,
    }
