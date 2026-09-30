import csv
import json
import math
import hashlib
from pathlib import Path

import numpy as np
import torch
import piq

from PIL import Image
from torchvision.transforms import ToTensor

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)

from src.dataset.dataset import srgb_to_linear
from src.model.priors import default_priors
from src.model.network import RefinedColorCorrectionNet
from src.model.losses_goal_v4 import (
    compute_goal_objective_v4,
)
from src.utils.color import linear_to_srgb
from src.measurement.metrics_utils import (
    calculate_uciqe,
    calculate_uiqm,
)


DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

EPOCHS = list(
    range(10, 201, 10)
)

BETA = 0.25
KAPPA = 1.0

BOOTSTRAP_SEED = 3407
BOOTSTRAP_N = 10000


CKPT_DIR = Path(
    "checkpoints/"
    "R27FORMAL_UIEBD_CRUG_OMRR_B025_W0020_S3407"
)

VAL = Path(
    "protocols/uiebd_split_v1/"
    "uiebd_val_split_v1.csv"
)

RESTORMER = Path(
    "src/model/restormer/"
    "best_restormer_prior_woruie.pth"
)

STAGE2 = Path(
    "src/model/stage2_color_refined.py"
)

GOALV4 = Path(
    "src/model/losses_goal_v4.py"
)

ROUTER = Path(
    "src/model/module/"
    "state_objective_reliability_router_v1.py"
)

OUT_DIR = Path(
    "audits/"
    "r27f4_formal_checkpoint_selection_v1"
)


EXPECTED_SHA = {
    STAGE2:
        "254dae3f9437e735720caf025555efe3363e44707c8bc193f994b84038b6401b",

    GOALV4:
        "5f25f87750aa5a93b3c2826d577719310f32ac7e2d48e856dcfe5550e7a0a0fa",

    ROUTER:
        "1978a3fb8d91058112cf00f58417faa09d71124daab0d49b534af6e44224b443",

    RESTORMER:
        "b2f6fa945708afff4ba58a1f0a3c4aab95965ef6086d646b3a5d4d6f73f4486c",
}


def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def sf(x):

    if torch.is_tensor(x):

        return float(
            x.detach()
            .reshape(-1)[0]
            .cpu()
        )

    return float(x)


def load_linear(path):

    image = Image.open(
        path
    ).convert("RGB")

    x = (
        ToTensor()(image)
        .unsqueeze(0)
    )

    return srgb_to_linear(
        x
    ).to(
        DEVICE,
        dtype=torch.float32,
    )


def find_gt(row):

    for key in [
        "label_path",
        "gt_path",
        "target_path",
        "reference_path",
    ]:

        if key not in row:
            continue

        text = str(
            row[key] or ""
        ).strip()

        if text:

            p = Path(text)

            if p.is_file():
                return p


    for key, value in row.items():

        if key == "input_path":
            continue

        if not any(
            token in key.lower()
            for token in [
                "label",
                "gt",
                "target",
                "reference",
            ]
        ):
            continue

        text = str(
            value or ""
        ).strip()

        if not text:
            continue

        p = Path(text)

        if p.is_file():
            return p


    raise RuntimeError(
        "Cannot resolve GT for "
        f"source={row.get('source_id')}"
    )


def to_u8(x):

    return (
        x.detach()
        .clamp(0, 1)
        .mul(255)
        .byte()
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def canonical_metrics(
    pred_linear,
    gt_linear,
    lpips_model,
):

    pred = linear_to_srgb(
        pred_linear
    ).clamp(
        0,
        1,
    )

    gt = linear_to_srgb(
        gt_linear
    ).clamp(
        0,
        1,
    )

    pu8 = to_u8(pred)
    gu8 = to_u8(gt)

    psnr = float(
        peak_signal_noise_ratio(
            gu8,
            pu8,
            data_range=255,
        )
    )

    ssim = float(
        structural_similarity(
            gu8,
            pu8,
            channel_axis=2,
            data_range=255,
        )
    )

    pred_chw = (
        pred.squeeze(0)
        .detach()
        .cpu()
    )

    uiqm = float(
        calculate_uiqm(
            pred_chw
        )
    )

    uciqe = float(
        calculate_uciqe(
            pred_chw
        )
    )

    lpips = float(
        lpips_model(
            pred,
            gt,
        )
        .detach()
        .reshape(-1)[0]
        .cpu()
    )

    return {
        "PSNR":
            psnr,

        "SSIM":
            ssim,

        "LPIPS":
            lpips,

        "UIQM":
            uiqm,

        "UCIQE":
            uciqe,
    }


def bootstrap_ci(values):

    x = np.asarray(
        values,
        dtype=np.float64,
    )

    assert len(x) == 70

    rng = (
        np.random.default_rng(
            BOOTSTRAP_SEED
        )
    )

    means = np.empty(
        BOOTSTRAP_N,
        dtype=np.float64,
    )

    pos = 0

    while pos < BOOTSTRAP_N:

        n = min(
            1000,
            BOOTSTRAP_N - pos,
        )

        idx = rng.integers(
            0,
            70,
            size=(n, 70),
        )

        means[
            pos:pos + n
        ] = x[idx].mean(
            axis=1
        )

        pos += n


    return (
        float(
            np.quantile(
                means,
                0.025,
            )
        ),
        float(
            np.quantile(
                means,
                0.975,
            )
        ),
    )


def normalized_excess(loss):

    loss = np.asarray(
        loss,
        dtype=np.float64,
    )

    return np.sqrt(
        np.maximum(
            np.expm1(
                2.0 * loss
            ),
            0.0,
        )
    )


def q(x, p):

    return float(
        np.quantile(
            np.asarray(
                x,
                dtype=np.float64,
            ),
            p,
        )
    )


def build_model():

    priors = default_priors(
        device=DEVICE,

        learned_prior_cfgs=[
            {
                "type":
                    "restormer",

                "path":
                    str(RESTORMER),

                "name":
                    "woRUIE",
            }
        ],

        handcrafted_prior_names=(
            "clahe",
            "unsharp",
            "grayworld",
        ),
    )

    assert len(
        priors
    ) == 4


    model = RefinedColorCorrectionNet(
        prior_modules=priors,

        stage1_synthesis_mode=(
            "equal_subset"
        ),

        stage1_anchor_indices=(
            0,
            3,
        ),

        coordinator_backend=(
            "direct_value"
        ),

        router_backend=(
            "sorr_v1"
        ),

        router_mode=(
            "direct_objective"
        ),
    ).to(
        DEVICE
    )


    model.stage2.set_mechanism_mode(
        "full"
    )

    model.stage2.set_route_feature_mode(
        "direct"
    )

    model.stage2.set_router_backend(
        "sorr_v1"
    )

    model.stage2.set_router_mode(
        "direct_objective"
    )

    model.stage2.set_router_objective(
        "balanced",
        KAPPA,
    )

    model.stage2.set_gcf_offset_routing_mode(
        "uniform"
    )

    model.stage2.set_execution_mode(
        "trust_region"
    )

    model.stage2.set_refinement_beta(
        BETA
    )

    return model


def main():

    torch.manual_seed(
        BOOTSTRAP_SEED
    )

    np.random.seed(
        BOOTSTRAP_SEED
    )

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    print("=" * 120)
    print(
        "R27F4-A FORMAL CHECKPOINT SELECTION V1"
    )
    print("=" * 120)

    print(
        "DEVICE =",
        DEVICE,
    )

    if DEVICE.type == "cuda":

        print(
            "GPU =",
            torch.cuda.get_device_name(0),
        )


    # ========================================================
    # Provenance
    # ========================================================

    for path, expected in (
        EXPECTED_SHA.items()
    ):

        assert path.is_file(), path

        actual = sha256(
            path
        )

        print(
            actual,
            path,
        )

        assert actual == expected


    ckpts = {
        epoch:
            CKPT_DIR
            /
            f"model_epoch_{epoch}.pth"

        for epoch in EPOCHS
    }

    assert all(
        path.is_file()
        for path in ckpts.values()
    )

    print(
        "CHECKPOINT_COUNT =",
        len(ckpts),
    )

    assert len(
        ckpts
    ) == 20

    print(
        "PASS_GATE_0_PROVENANCE"
    )


    # ========================================================
    # val70
    # ========================================================

    with VAL.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        manifest = list(
            csv.DictReader(f)
        )

    assert len(
        manifest
    ) == 70

    for row in manifest:

        assert (
            row["split"]
            ==
            "validation"
        )

        assert "/test/" not in (
            row[
                "input_path"
            ]
            .replace("\\", "/")
        )

    print(
        "PASS_GATE_1_VAL70_NO_TEST"
    )


    # ========================================================
    # Model + LPIPS
    # ========================================================

    model = build_model()

    lpips_model = piq.LPIPS(
        replace_pooling=True
    ).to(
        DEVICE
    )

    lpips_model.eval()


    all_rows = []
    checkpoint_shas = {}


    # ========================================================
    # Complete 20-checkpoint trajectory
    # ========================================================

    with torch.no_grad():

        for epoch in EPOCHS:

            ckpt = ckpts[
                epoch
            ]

            checkpoint_shas[
                str(epoch)
            ] = sha256(
                ckpt
            )


            obj = torch.load(
                ckpt,
                map_location=DEVICE,
            )

            assert (
                isinstance(
                    obj,
                    dict,
                )
                and
                "state_dict" in obj
            )

            msg = (
                model.load_state_dict(
                    obj["state_dict"],
                    strict=True,
                )
            )

            print()
            print(
                f"EPOCH {epoch} STRICT_LOAD =",
                msg,
            )

            model.eval()


            for index, row in enumerate(
                manifest,
                1,
            ):

                x = load_linear(
                    Path(
                        row[
                            "input_path"
                        ]
                    )
                )

                gt = load_linear(
                    find_gt(row)
                )


                out = model(
                    x=x
                )

                S = out[
                    "I_synthesis"
                ]

                R = out[
                    "I_refined"
                ]

                priors = out[
                    "priors"
                ]

                como = out[
                    "como_out"
                ]


                # --------------------------------------------
                # exact anchor
                # --------------------------------------------

                expected_S = (
                    priors[0]
                    +
                    priors[3]
                ) / 2.0

                assert (
                    (
                        S
                        -
                        expected_S
                    )
                    .abs()
                    .max()
                    .item()
                    <=
                    1e-7
                )


                anchor = (
                    canonical_metrics(
                        S,
                        gt,
                        lpips_model,
                    )
                )

                refined = (
                    canonical_metrics(
                        R,
                        gt,
                        lpips_model,
                    )
                )


                goal = (
                    compute_goal_objective_v4(
                        I_ref=R,
                        I_syn=S,
                        priors=priors,
                        mode="balanced",
                        kappa=KAPPA,
                    )
                )


                route = (
                    model.stage2
                    .last_cas_weights
                )

                assert (
                    route is not None
                )

                entropy = (
                    -(
                        route.clamp_min(
                            1e-12
                        )
                        *
                        route.clamp_min(
                            1e-12
                        ).log()
                    )
                    .sum(
                        dim=1
                    )
                    .mean()
                    /
                    math.log(4.0)
                )


                usage = (
                    route.mean(
                        dim=(0, 2, 3)
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )


                rho = sf(
                    como[
                        "trust_radius"
                    ]
                )

                radius = sf(
                    como[
                        "candidate_radius"
                    ]
                )

                ratio = (
                    radius
                    /
                    (
                        rho
                        +
                        1e-8
                    )
                )


                all_rows.append({
                    "epoch":
                        epoch,

                    "source_id":
                        row[
                            "source_id"
                        ],

                    "PSNR":
                        refined[
                            "PSNR"
                        ],

                    "SSIM":
                        refined[
                            "SSIM"
                        ],

                    "LPIPS":
                        refined[
                            "LPIPS"
                        ],

                    "UIQM":
                        refined[
                            "UIQM"
                        ],

                    "UCIQE":
                        refined[
                            "UCIQE"
                        ],

                    "anchor_PSNR":
                        anchor[
                            "PSNR"
                        ],

                    "anchor_SSIM":
                        anchor[
                            "SSIM"
                        ],

                    "anchor_LPIPS":
                        anchor[
                            "LPIPS"
                        ],

                    "anchor_UIQM":
                        anchor[
                            "UIQM"
                        ],

                    "anchor_UCIQE":
                        anchor[
                            "UCIQE"
                        ],

                    "goal_utility":
                        sf(
                            goal[
                                "goal_utility"
                            ]
                        ),

                    "goal_loss":
                        sf(
                            goal[
                                "goal_loss"
                            ]
                        ),

                    "chroma_safe":
                        sf(
                            goal[
                                "chroma_safe"
                            ]
                        ),

                    "clip_safe":
                        sf(
                            goal[
                                "clip_safe"
                            ]
                        ),

                    "safety_loss":
                        sf(
                            goal[
                                "safety_loss"
                            ]
                        ),

                    "candidate_to_trust":
                        ratio,

                    "trust_active":
                        float(
                            ratio
                            >
                            1.0
                            +
                            1e-6
                        ),

                    "routing_entropy":
                        sf(
                            entropy
                        ),

                    "route_CLAHE":
                        float(
                            usage[0]
                        ),

                    "route_Unsharp":
                        float(
                            usage[1]
                        ),

                    "route_GrayWorld":
                        float(
                            usage[2]
                        ),

                    "route_Restormer":
                        float(
                            usage[3]
                        ),
                })


                if (
                    index % 10 == 0
                    or
                    index == 70
                ):

                    print(
                        f"EPOCH {epoch:03d} "
                        f"VAL {index:02d}/70"
                    )


    print()
    print(
        "PASS_GATE_2_ALL_CHECKPOINTS_EVALUATED"
    )


    # ========================================================
    # Save per-source
    # ========================================================

    detail_csv = (
        OUT_DIR
        /
        "formal_checkpoint_per_source_v1.csv"
    )

    with detail_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                all_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            all_rows
        )


    # ========================================================
    # Aggregate
    # ========================================================

    summaries = []

    for epoch in EPOCHS:

        rs = [
            r
            for r in all_rows
            if (
                r["epoch"]
                ==
                epoch
            )
        ]

        assert len(
            rs
        ) == 70


        s = {
            "epoch":
                epoch,

            "checkpoint_sha256":
                checkpoint_shas[
                    str(epoch)
                ],
        }


        for metric in [
            "PSNR",
            "SSIM",
            "LPIPS",
            "UIQM",
            "UCIQE",
        ]:

            pred = np.asarray(
                [
                    r[metric]
                    for r in rs
                ],
                dtype=np.float64,
            )

            anchor = np.asarray(
                [
                    r[
                        "anchor_"
                        +
                        metric
                    ]
                    for r in rs
                ],
                dtype=np.float64,
            )

            delta = (
                pred
                -
                anchor
            )

            lo, hi = bootstrap_ci(
                delta
            )

            s[
                metric
            ] = float(
                pred.mean()
            )

            s[
                "delta_"
                +
                metric
            ] = float(
                delta.mean()
            )

            s[
                "delta_"
                +
                metric
                +
                "_CI_lo"
            ] = lo

            s[
                "delta_"
                +
                metric
                +
                "_CI_hi"
            ] = hi


        utility = np.asarray(
            [
                r[
                    "goal_utility"
                ]
                for r in rs
            ],
            dtype=np.float64,
        )

        ulo, uhi = bootstrap_ci(
            utility
        )

        s[
            "goal_utility"
        ] = float(
            utility.mean()
        )

        s[
            "goal_utility_CI_lo"
        ] = ulo

        s[
            "goal_utility_CI_hi"
        ] = uhi

        s[
            "beneficial_fraction"
        ] = float(
            np.mean(
                utility > 0
            )
        )


        chroma = np.asarray(
            [
                r[
                    "chroma_safe"
                ]
                for r in rs
            ],
            dtype=np.float64,
        )

        clip = np.asarray(
            [
                r[
                    "clip_safe"
                ]
                for r in rs
            ],
            dtype=np.float64,
        )

        zc = normalized_excess(
            chroma
        )

        zk = normalized_excess(
            clip
        )


        s[
            "z_chroma_q95"
        ] = q(
            zc,
            0.95,
        )

        s[
            "z_chroma_max"
        ] = float(
            zc.max()
        )

        s[
            "z_clip_q95"
        ] = q(
            zk,
            0.95,
        )

        s[
            "z_clip_max"
        ] = float(
            zk.max()
        )


        s[
            "routing_entropy"
        ] = float(
            np.mean(
                [
                    r[
                        "routing_entropy"
                    ]
                    for r in rs
                ]
            )
        )

        s[
            "trust_active_fraction"
        ] = float(
            np.mean(
                [
                    r[
                        "trust_active"
                    ]
                    for r in rs
                ]
            )
        )


        s[
            "mean_route_CLAHE"
        ] = float(
            np.mean(
                [
                    r[
                        "route_CLAHE"
                    ]
                    for r in rs
                ]
            )
        )

        s[
            "mean_route_Unsharp"
        ] = float(
            np.mean(
                [
                    r[
                        "route_Unsharp"
                    ]
                    for r in rs
                ]
            )
        )

        s[
            "mean_route_GrayWorld"
        ] = float(
            np.mean(
                [
                    r[
                        "route_GrayWorld"
                    ]
                    for r in rs
                ]
            )
        )

        s[
            "mean_route_Restormer"
        ] = float(
            np.mean(
                [
                    r[
                        "route_Restormer"
                    ]
                    for r in rs
                ]
            )
        )


        fidelity_pass = (
            s[
                "delta_PSNR"
            ]
            >=
            -0.50
            and
            s[
                "delta_SSIM"
            ]
            >=
            -0.010
            and
            s[
                "delta_LPIPS"
            ]
            <=
            +0.010
        )


        utility_pass = (
            s[
                "goal_utility_CI_lo"
            ]
            >
            0.0
        )


        perceptual_sig_harm = (
            s[
                "delta_UIQM_CI_hi"
            ]
            <
            0.0
            and
            s[
                "delta_UCIQE_CI_hi"
            ]
            <
            0.0
        )


        support_pass = (
            s[
                "z_chroma_q95"
            ]
            <=
            0.25
            and
            s[
                "z_chroma_max"
            ]
            <=
            1.00
            and
            s[
                "z_clip_q95"
            ]
            <=
            0.25
            and
            s[
                "z_clip_max"
            ]
            <=
            1.00
        )


        s[
            "fidelity_pass"
        ] = bool(
            fidelity_pass
        )

        s[
            "utility_pass"
        ] = bool(
            utility_pass
        )

        s[
            "perceptual_sig_harm"
        ] = bool(
            perceptual_sig_harm
        )

        s[
            "support_pass"
        ] = bool(
            support_pass
        )

        s[
            "SURVIVES"
        ] = bool(
            fidelity_pass
            and
            utility_pass
            and
            not
            perceptual_sig_harm
            and
            support_pass
        )


        summaries.append(
            s
        )


    # ========================================================
    # Frozen checkpoint selection
    # ========================================================

    survivors = [
        s
        for s in summaries
        if s[
            "SURVIVES"
        ]
    ]


    if survivors:

        selected = sorted(
            survivors,

            key=lambda x: (
                -x[
                    "goal_utility"
                ],

                -x[
                    "PSNR"
                ],

                -x[
                    "SSIM"
                ],

                x[
                    "LPIPS"
                ],

                x[
                    "epoch"
                ],
            ),
        )[0]

        decision = "PASS"

    else:

        selected = None
        decision = "NO_GO"


    # ========================================================
    # Save summary
    # ========================================================

    summary_csv = (
        OUT_DIR
        /
        "formal_checkpoint_summary_v1.csv"
    )

    with summary_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summaries[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            summaries
        )


    report = {
        "protocol":
            "R27F4-A Formal Checkpoint Selection V1",

        "validation_count":
            70,

        "candidate_epochs":
            EPOCHS,

        "candidate_count":
            20,

        "test_used":
            False,

        "beta":
            BETA,

        "objective":
            "balanced",

        "bootstrap_seed":
            BOOTSTRAP_SEED,

        "bootstrap_replicates":
            BOOTSTRAP_N,

        "summaries":
            summaries,

        "survivor_epochs":
            [
                s[
                    "epoch"
                ]
                for s in survivors
            ],

        "decision":
            decision,

        "selected_epoch":
            (
                None
                if selected is None
                else selected[
                    "epoch"
                ]
            ),

        "selected_checkpoint_sha256":
            (
                None
                if selected is None
                else selected[
                    "checkpoint_sha256"
                ]
            ),
    }


    report_json = (
        OUT_DIR
        /
        "formal_checkpoint_selection_report_v1.json"
    )

    report_json.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )


    # ========================================================
    # Console
    # ========================================================

    print()
    print("=" * 150)
    print(
        "FORMAL CHECKPOINT TRAJECTORY"
    )
    print("=" * 150)


    for s in summaries:

        print(
            f"E{s['epoch']:03d} | "
            f"PSNR={s['PSNR']:.4f} "
            f"d={s['delta_PSNR']:+.4f} | "
            f"SSIM={s['SSIM']:.6f} "
            f"d={s['delta_SSIM']:+.6f} | "
            f"LPIPS={s['LPIPS']:.6f} "
            f"d={s['delta_LPIPS']:+.6f} | "
            f"UIQM_d={s['delta_UIQM']:+.4f} | "
            f"UCIQE_d={s['delta_UCIQE']:+.4f} | "
            f"UTIL={s['goal_utility']:+.5f} "
            f"[{s['goal_utility_CI_lo']:+.5f},"
            f"{s['goal_utility_CI_hi']:+.5f}] | "
            f"zK95={s['z_clip_q95']:.4f} "
            f"zKmax={s['z_clip_max']:.4f} | "
            f"SURVIVES={s['SURVIVES']}"
        )


    print()
    print(
        "SURVIVOR_EPOCHS =",
        [
            s[
                "epoch"
            ]
            for s in survivors
        ],
    )

    print(
        "DECISION =",
        decision,
    )

    print(
        "SELECTED_EPOCH =",
        (
            None
            if selected is None
            else selected[
                "epoch"
            ]
        ),
    )

    if selected is not None:

        print(
            "SELECTED_CHECKPOINT_SHA256 =",
            selected[
                "checkpoint_sha256"
            ],
        )


    print(
        "DETAIL_CSV =",
        detail_csv,
    )

    print(
        "SUMMARY_CSV =",
        summary_csv,
    )

    print(
        "REPORT_JSON =",
        report_json,
    )

    print(
        "TEST_USED=False"
    )

    print(
        "PASS_R27F4A_FORMAL_CHECKPOINT_SELECTION_V1"
    )

    print(
        "TERMINAL_SURVIVED"
    )


if __name__ == "__main__":
    main()
