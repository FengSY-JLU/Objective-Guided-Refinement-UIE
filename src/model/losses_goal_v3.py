import torch
import torch.nn.functional as F


# ============================================================
# Color conversion
# Keep local here to avoid circular imports.
# Input convention: linear RGB in [0,1].
# ============================================================

def linear_rgb_to_xyz(img):
    img = torch.clamp(img, 0.0, 1.0)

    r = img[:, 0:1]
    g = img[:, 1:2]
    b = img[:, 2:3]

    x = 0.4124564*r + 0.3575761*g + 0.1804375*b
    y = 0.2126729*r + 0.7151522*g + 0.0721750*b
    z = 0.0193339*r + 0.1191920*g + 0.9503041*b

    return torch.cat([x, y, z], dim=1)


def xyz_to_lab(xyz):
    ref = torch.tensor(
        [0.95047, 1.0, 1.08883],
        device=xyz.device,
        dtype=xyz.dtype,
    ).view(1, 3, 1, 1)

    xyz = xyz / (ref + 1e-12)

    eps = 0.008856
    kappa = 903.3

    mask = (xyz > eps).to(xyz.dtype)

    f = (
        xyz.clamp_min(0.0).pow(1.0/3.0) * mask
        + ((kappa*xyz + 16.0)/116.0) * (1.0-mask)
    )

    L = 116.0*f[:, 1:2] - 16.0
    a = 500.0*(f[:, 0:1] - f[:, 1:2])
    b = 200.0*(f[:, 1:2] - f[:, 2:3])

    return torch.cat([L, a, b], dim=1)


def rgb_to_lab(img):
    return xyz_to_lab(
        linear_rgb_to_xyz(img)
    )


# ============================================================
# Goal descriptors
# ============================================================

def cast_score(img):
    """
    Higher = weaker global Lab color cast.
    Zero is the theoretical neutral optimum.
    """
    lab = rgb_to_lab(img)

    a = lab[:, 1]
    b = lab[:, 2]

    ma = a.mean(dim=[1, 2])
    mb = b.mean(dim=[1, 2])

    return -torch.sqrt(
        ma**2 + mb**2 + 1e-8
    )


def luminance(img):
    return (
        0.299 * img[:, 0:1]
        + 0.587 * img[:, 1:2]
        + 0.114 * img[:, 2:3]
    )


def local_contrast_score(img):
    """
    Local luminance separability.
    Higher = stronger local visibility.
    """
    y = luminance(img)

    mu = F.avg_pool2d(
        y,
        kernel_size=15,
        stride=1,
        padding=7,
    )

    return (
        y - mu
    ).abs().mean(
        dim=[1, 2, 3]
    )


def edge_visibility_score(img):
    """
    First-order luminance edge visibility.
    Existing structural losses remain responsible for
    preventing arbitrary structural distortion.
    """
    y = luminance(img)

    gx = (
        y[:, :, :, 1:]
        - y[:, :, :, :-1]
    ).abs().mean(
        dim=[1, 2, 3]
    )

    gy = (
        y[:, :, 1:, :]
        - y[:, :, :-1, :]
    ).abs().mean(
        dim=[1, 2, 3]
    )

    return gx + gy


def chroma_score(img):
    lab = rgb_to_lab(img)

    a = lab[:, 1]
    b = lab[:, 2]

    return torch.sqrt(
        a**2 + b**2 + 1e-8
    ).mean(
        dim=[1, 2]
    )


def soft_boundary_mass(
    img,
    lo=0.02,
    hi=0.98,
    temperature=0.01,
):
    """
    Smooth diagnostic of newly created near-boundary RGB mass.
    Unlike the legacy design, reducing this term gives no
    positive utility reward; only increases are penalized.
    """
    low = torch.sigmoid(
        (lo - img) / temperature
    )

    high = torch.sigmoid(
        (img - hi) / temperature
    )

    return (
        low + high
    ).mean(
        dim=[1, 2, 3]
    )


# ============================================================
# Contributor-supported scale
# ============================================================

def supported_scale(
    fn,
    I_syn,
    priors,
    eps=1e-4,
):
    """
    Robust per-sample attribute scale supplied by contributors.

    Crucially detached:
    the model cannot increase its allowed objective scale by
    moving Stage-I or contributor outputs through this loss.
    """
    syn = I_syn.detach()

    q_syn = fn(syn)

    q_prior = torch.stack(
        [
            fn(P.detach())
            for P in priors
        ],
        dim=1,
    )

    scale = torch.median(
        torch.abs(
            q_prior
            - q_syn.unsqueeze(1)
        ),
        dim=1,
    ).values

    return torch.clamp(
        scale,
        min=eps,
    ).detach()


# ============================================================
# Robust bounded penalty
# ============================================================

def cauchy_penalty(z):
    """
    Cauchy robust penalty:

        rho(z) = 0.5 * log(1 + z^2)

    It grows only logarithmically for severe violations while
    retaining a useful correction gradient:

        d rho / dz = z / (1 + z^2)

    This avoids both:
      - unbounded linear domination, and
      - the excessive far-from-target gradient saturation of
        the previous rational bounded penalty.
    """
    return 0.5 * torch.log1p(
        z * z
    )


# ============================================================
# Target-reaching loss / diagnostic utility
# ============================================================

def target_loss_and_utility(
    q_ref,
    q_syn,
    target,
    scale,
):
    """
    Training minimizes normalized distance to a desired target.

    Diagnostic utility is positive iff refinement moves closer
    to the desired target than synthesis.
    """
    q_syn = q_syn.detach()
    target = target.detach()
    scale = scale.detach()

    z = (
        q_ref - target
    ) / scale

    # Bounded robust target-reaching penalty.
    # Prevents a badly misaligned candidate from dominating
    # the complete canonical objective by raw magnitude.
    loss = cauchy_penalty(
        z
    )

    d0 = torch.abs(
        q_syn - target
    )

    d1 = torch.abs(
        q_ref - target
    )

    utility = torch.tanh(
        (d0 - d1) / scale
    )

    return loss, utility


# ============================================================
# Safety constraints
# ============================================================

def chroma_safety_loss(
    I_ref,
    I_syn,
    priors,
    eps=1e-4,
):
    """
    One-sided chroma safety.

    Chroma is NOT a positive enhancement objective.

    We only prevent unsupported chroma expansion, because the
    legacy failure mode was excessive chroma amplification.

    Reducing chroma is not automatically penalized here; this
    avoids conflicting with color-cast correction.
    """
    q_ref = chroma_score(
        I_ref
    )

    syn = I_syn.detach()

    q_syn = chroma_score(
        syn
    )

    q_prior = torch.stack(
        [
            chroma_score(
                P.detach()
            )
            for P in priors
        ],
        dim=1,
    )

    supported = torch.cat(
        [
            q_syn.unsqueeze(1),
            q_prior,
        ],
        dim=1,
    )

    upper = supported.max(
        dim=1
    ).values.detach()

    scale = torch.median(
        torch.abs(
            q_prior
            - q_syn.unsqueeze(1)
        ),
        dim=1,
    ).values

    scale = torch.clamp(
        scale,
        min=eps,
    ).detach()

    normalized_excess = F.relu(
        (
            q_ref
            - upper
        )
        / scale
    )

    return cauchy_penalty(
        normalized_excess
    ).mean()


def clipping_safety_loss(
    I_ref,
    I_syn,
    priors,
    eps=1e-3,
):
    """
    One-sided safety:
    only penalize NEW near-boundary mass relative to synthesis.
    """
    q_ref = soft_boundary_mass(
        I_ref
    )

    syn = I_syn.detach()

    q_syn = soft_boundary_mass(
        syn
    )

    q_prior = torch.stack(
        [
            soft_boundary_mass(
                P.detach()
            )
            for P in priors
        ],
        dim=1,
    )

    scale = torch.median(
        torch.abs(
            q_prior
            - q_syn.unsqueeze(1)
        ),
        dim=1,
    ).values

    scale = torch.clamp(
        scale,
        min=eps,
    ).detach()

    excess = F.relu(
        (q_ref - q_syn) / scale
    )

    return cauchy_penalty(
        excess
    ).mean()


# ============================================================
# Goal-specific objective
# ============================================================

def compute_goal_objective_v3(
    I_ref,
    I_syn,
    priors,
    mode="balanced",
    kappa=1.0,
):
    """
    Goal terms and safety terms are intentionally separated.

    Goals:
      balanced:
          cast correction + local contrast + edge visibility

      color_cast:
          cast correction only

      visibility:
          local contrast + edge visibility

    Contributors supply per-sample change scales.
    They do NOT define the desired final answer.
    """
    mode = str(mode).strip().lower()

    valid = {
        "balanced",
        "color_cast",
        "visibility",
    }

    if mode not in valid:
        raise ValueError(
            f"Unsupported goal mode: {mode}; "
            f"expected one of {sorted(valid)}"
        )

    syn = I_syn.detach()

    goal_losses = []
    goal_utils = []

    # --------------------------------------------------------
    # Color-cast correction
    # --------------------------------------------------------
    if mode in {
        "balanced",
        "color_cast",
    }:
        q_syn = cast_score(syn)
        q_ref = cast_score(I_ref)

        scale = supported_scale(
            cast_score,
            syn,
            priors,
        )

        # Desired transition:
        # improve cast score by one contributor-supported unit,
        # but never request a value above neutral (0).
        target = torch.minimum(
            torch.zeros_like(q_syn),
            q_syn
            + float(kappa) * scale,
        )

        loss, utility = (
            target_loss_and_utility(
                q_ref,
                q_syn,
                target,
                scale,
            )
        )

        goal_losses.append(loss)
        goal_utils.append(utility)

    # --------------------------------------------------------
    # Visibility
    # --------------------------------------------------------
    if mode in {
        "balanced",
        "visibility",
    }:
        for fn in [
            local_contrast_score,
            edge_visibility_score,
        ]:
            q_syn = fn(syn)
            q_ref = fn(I_ref)

            scale = supported_scale(
                fn,
                syn,
                priors,
            )

            target = (
                q_syn
                + float(kappa) * scale
            )

            loss, utility = (
                target_loss_and_utility(
                    q_ref,
                    q_syn,
                    target,
                    scale,
                )
            )

            goal_losses.append(loss)
            goal_utils.append(utility)

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

    # Equal treatment of:
    #   desired-goal attainment
    #   common safety bundle
    #
    # The outer canonical coefficient remains 0.10.
    # Both bundles are bounded to [0,1).
    # Equal averaging prevents either goal or safety from
    # dominating the canonical objective by raw scale.
    objective_loss = 0.5 * (
        goal_loss
        + safety_loss
    )

    return {
        "objective_loss": objective_loss,
        "goal_loss": goal_loss,
        "goal_utility": goal_utility,
        "chroma_safe": chroma_safe,
        "clip_safe": clip_safe,
        "safety_loss": safety_loss,
    }


# ============================================================
# Contributor-supported proposal regularization
# ============================================================

def proposal_support_loss(
    I_candidate,
    I_syn,
    priors,
    eps=1e-8,
):
    """
    Remove the radial null direction introduced by active
    hard trust-region projection.

    The candidate is unconstrained inside the contributor-
    supported radius. Only unsupported radial expansion is
    penalized.

        rho(x) = median_k mean |H_k - I_s|

        r_J = mean |J - I_s| / rho

        L_prop = Cauchy( relu(r_J - 1) )

    Important:
      - rho is detached.
      - the synthesis baseline in the explicit radius
        computation is detached.
      - this is NOT an identity/zero-residual penalty.
      - no penalty is applied when r_J <= 1.
    """

    syn = I_syn.detach()

    support_distances = torch.stack(
        [
            (
                P.detach()
                - syn
            ).abs().mean(
                dim=[1, 2, 3]
            )
            for P in priors
        ],
        dim=1,
    )

    trust_radius = torch.median(
        support_distances,
        dim=1,
    ).values.detach()

    candidate_radius = (
        I_candidate
        - syn
    ).abs().mean(
        dim=[1, 2, 3]
    )

    support_ratio = (
        candidate_radius
        / (
            trust_radius
            + eps
        )
    )

    excess = F.relu(
        support_ratio
        - 1.0
    )

    loss = cauchy_penalty(
        excess
    ).mean()

    return {
        "loss": loss,
        "candidate_radius": candidate_radius.mean(),
        "trust_radius": trust_radius.mean(),
        "support_ratio": support_ratio.mean(),
        "active_fraction": (
            support_ratio > 1.0
        ).float().mean(),
    }
