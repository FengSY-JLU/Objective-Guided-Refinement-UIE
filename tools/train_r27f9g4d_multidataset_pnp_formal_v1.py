from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


from src.dataset.data import get_training_set_manifest
from src.model.priors import default_priors

from src.model.pnp.network import PnPRefinementNet
from src.model.pnp.contributor_bank import ContributorBank
from src.model.pnp.anchor_policy import NamedMeanAnchorPolicy
from src.model.pnp.omrr_refinement_core import PnPOMRRRefinementCore
from src.model.pnp.checkpoint import save_core_checkpoint

from src.model.losses_canonical import compute_total_loss


SEED = 3407
EPOCHS = 200
BATCH_SIZE = 1
PATCH_SIZE = 128
THREADS = 4

LR = 1e-4
MILESTONES = [100]
LR_GAMMA = 0.5

SAVE_EVERY = 10

OBJECTIVE_MODE = "balanced"
GOAL_KAPPA = 1.0
GOAL_OBJECTIVE_WEIGHT = 0.020
PROPOSAL_SUPPORT_WEIGHT = 0.0
REFINEMENT_GAMMA = 0.25


PROFILE_PATH = (
    ROOT
    / "audits/r27f9g3_post_synthesis_evidence_v2"
    / "FINAL_DATASET_PROFILES_V2.json"
)

PROTOCOL_PATH = (
    ROOT
    / "protocols/r27f9g4d_multidataset_formal_training_v1"
    / "FORMAL_TRAINING_PROTOCOL_V1.json"
)

FREEZE_PATH = (
    ROOT
    / "protocols/r27f9g4d_multidataset_formal_training_v1"
    / "FREEZE_V1.json"
)

RESTORMER_PATH = (
    ROOT
    / "src/model/restormer"
    / "best_restormer_prior_woruie.pth"
)


DATASETS = {
    "UFO120": {
        "manifest":
            ROOT
            / "protocols/r27f8d_ufo120_split_geometry_v1"
            / "ufo120_train_split_v1.csv",

        "root":
            ROOT / "Dataset/UIE/UFO120/train_val/lrd",

        "expected_train":
            1345,
    },

    "EUVP": {
        "manifest":
            ROOT
            / "protocols/r27f9c_multidataset_split_v1"
            / "euvp_train_v1.csv",

        "root":
            Path(
                "/media/pi/sdb/fsy/workspaces/"
                "260907_CACR_TMM_resubmission_v1/"
                "Dataset/UIE/EUVP/Paired/"
                "underwater_imagenet/trainA"
            ),

        "expected_train":
            3330,
    },

    "RUIE": {
        "manifest":
            ROOT
            / "protocols/r27f9c_multidataset_split_v1"
            / "ruie_train_v1.csv",

        "root":
            Path(
                "/media/pi/sdb/fsy/"
                "UIE-methods-compare/"
                "Dataset/UIE/RUIE/train"
            ),

        "expected_train":
            2700,
    },

    "OceanDark": {
        "manifest":
            ROOT
            / "protocols/r27f9c_multidataset_split_v1"
            / "oceandark_train_v1.csv",

        "root":
            Path(
                "/media/pi/sdb/fsy/"
                "UIE-methods-compare/"
                "Dataset/UIE/OceanDark"
            ),

        "expected_train":
            109,
    },
}


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = (
        torch.initial_seed()
        % (2 ** 32)
    )

    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_single_contributor(name, device):

    if name == "restormer_woruie":

        modules = default_priors(
            device=device,

            learned_prior_cfgs=[{
                "type": "restormer",
                "path": str(
                    RESTORMER_PATH
                ),
                "name":
                    "restormer_woruie",
            }],

            handcrafted_prior_names=(),
        )

    else:

        modules = default_priors(
            device=device,
            learned_prior_cfgs=None,
            handcrafted_prior_names=(
                name,
            ),
        )

    assert len(modules) == 1

    return modules[0]


def build_model(profile, device):

    contributors = OrderedDict()

    for name in profile["slot_order"]:

        contributors[name] = (
            build_single_contributor(
                name,
                device,
            )
        )

    bank = ContributorBank(
        contributors,
        freeze=True,
    )

    anchor = NamedMeanAnchorPolicy(
        profile["anchor_methods"]
    )

    # Contributor construction can consume RNG.
    # Reset immediately before fresh formal router init.
    set_seed(SEED)

    core = PnPOMRRRefinementCore(
        objective=OBJECTIVE_MODE,
        goal_kappa=GOAL_KAPPA,
        refinement_gamma=REFINEMENT_GAMMA,
        router_mode="direct_objective",
    )

    model = PnPRefinementNet(
        contributor_bank=bank,
        anchor_policy=anchor,
        refinement_core=core,
    ).to(device)

    return model


def router_digest(model):

    h = hashlib.sha256()

    for name, value in sorted(
        model
        .refinement_core
        .router
        .state_dict()
        .items()
    ):
        h.update(
            name.encode("utf-8")
        )

        h.update(
            value
            .detach()
            .cpu()
            .contiguous()
            .numpy()
            .tobytes()
        )

    return h.hexdigest()


def check_finite_loss_dict(loss_dict):

    for key, value in loss_dict.items():

        if torch.is_tensor(value):
            value = float(
                value.detach().item()
            )
        else:
            value = float(value)

        if not math.isfinite(value):
            raise RuntimeError(
                f"non-finite loss field "
                f"{key}={value}"
            )


def optimizer_to_device(
    optimizer,
    device,
):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(
                    device
                )


def save_resume_checkpoint(
    *,
    path,
    dataset,
    epoch,
    model,
    optimizer,
    scheduler,
    train_generator,
    protocol_sha,
    profile_sha,
):

    obj = {
        "format":
            "r27f9g4d_formal_resume_v1",

        "dataset":
            dataset,

        "epoch":
            int(epoch),

        "seed":
            SEED,

        "protocol_sha256":
            protocol_sha,

        "profile_sha256":
            profile_sha,

        "router_state_dict": {
            k:
                v.detach().cpu()

            for k, v in
            model
            .refinement_core
            .router
            .state_dict()
            .items()
        },

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "train_generator_state":
            train_generator.get_state(),

        "python_random_state":
            random.getstate(),

        "numpy_random_state":
            np.random.get_state(),

        "torch_rng_state":
            torch.get_rng_state(),

        "cuda_rng_state_all":
            (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
    }

    torch.save(
        obj,
        path,
    )


def find_latest_resume(ckpt_dir):

    candidates = []

    for path in ckpt_dir.glob(
        "resume_epoch_*.pth"
    ):
        try:
            epoch = int(
                path.stem.split("_")[-1]
            )
        except ValueError:
            continue

        candidates.append(
            (epoch, path)
        )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda x: x[0],
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--dataset",
        required=True,
        choices=tuple(DATASETS.keys()),
    )

    ap.add_argument(
        "--device",
        default="cuda:0",
    )

    ap.add_argument(
        "--auto_resume",
        action="store_true",
    )

    args = ap.parse_args()

    dataset = args.dataset
    device = torch.device(
        args.device
    )

    if device.type == "cuda":
        assert torch.cuda.is_available()
        torch.cuda.set_device(
            device.index or 0
        )

    assert FREEZE_PATH.is_file(), (
        "formal package is not frozen: "
        f"{FREEZE_PATH}"
    )

    freeze = json.loads(
        FREEZE_PATH.read_text(
            encoding="utf-8"
        )
    )

    self_sha = sha256(
        Path(__file__)
    )

    assert (
        self_sha
        ==
        freeze["files"][
            "formal_trainer"
        ]
    )

    assert (
        sha256(PROTOCOL_PATH)
        ==
        freeze["files"][
            "formal_training_protocol"
        ]
    )

    assert (
        sha256(PROFILE_PATH)
        ==
        freeze["files"][
            "final_dataset_profiles"
        ]
    )

    protocol_sha = (
        freeze["files"][
            "formal_training_protocol"
        ]
    )

    profile_sha = (
        freeze["files"][
            "final_dataset_profiles"
        ]
    )

    profiles_obj = json.loads(
        PROFILE_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert (
        profiles_obj[
            "profile_state"
        ]
        ==
        "FULLY_FROZEN"
    )

    profile = (
        profiles_obj[
            "profiles"
        ][dataset]
    )

    assert profile["K"] == 4

    assert (
        profile["slot_order"]
        ==
        profile["evidence_bank"]
    )

    spec = DATASETS[
        dataset
    ]

    assert spec["manifest"].is_file()
    assert spec["root"].is_dir()

    run_name = (
        f"R27F9G4D_{dataset}_"
        "PNP_OMRR_B025_W0020_S3407"
    )

    ckpt_dir = (
        ROOT
        / "checkpoints"
        / run_name
    )

    run_dir = (
        ROOT
        / "runs"
        / "r27f9g4d_formal_multidataset_e200_v1"
        / run_name
    )

    ckpt_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed_path = (
        run_dir
        / "COMPLETED.json"
    )

    if completed_path.is_file():

        completed = json.loads(
            completed_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            completed.get("status")
            ==
            "PASS_FORMAL_TRAINING_COMPLETE"
            and
            int(
                completed.get(
                    "final_epoch",
                    -1,
                )
            )
            ==
            EPOCHS
            and
            completed.get(
                "protocol_sha256"
            )
            ==
            protocol_sha
        ):
            print(
                f"FORMAL_ALREADY_COMPLETE="
                f"{dataset}"
            )

            print(
                "PASS_R27F9G4D_FORMAL_TRAINING_COMPLETE"
            )

            return

    print("=" * 96)
    print(
        "R27F9-G4D FORMAL "
        "MULTIDATASET PNP TRAINING V1"
    )
    print("=" * 96)

    print(
        f"DATASET={dataset}"
    )

    print(
        f"RUN_NAME={run_name}"
    )

    print(
        f"PROTOCOL_SHA256={protocol_sha}"
    )

    print(
        f"PROFILE_SHA256={profile_sha}"
    )

    print(
        "EPOCHS=200"
    )

    print(
        "BATCH_SIZE=1"
    )

    print(
        "PATCH_SIZE=128"
    )

    print(
        "SEED=3407"
    )

    print(
        "OPTIMIZER=Adam"
    )

    print(
        "LR=0.0001"
    )

    print(
        "SCHEDULER=MultiStepLR([100],gamma=0.5)"
    )

    print(
        "SAVE_EVERY=10"
    )

    print(
        "TEST_USED=False"
    )

    print(
        "GT_USED=False"
    )

    print(
        "HR_USED=False"
    )

    print(
        "ANCHOR="
        +
        ",".join(
            profile[
                "anchor_methods"
            ]
        )
    )

    print(
        "BANK="
        +
        ",".join(
            profile[
                "slot_order"
            ]
        )
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    set_seed(SEED)

    train_set = get_training_set_manifest(
        str(spec["manifest"]),
        str(spec["root"]),
        str(spec["root"]),
        PATCH_SIZE,
        True,
    )

    assert (
        len(train_set)
        ==
        spec["expected_train"]
    )

    train_generator = (
        torch.Generator()
    )

    train_generator.manual_seed(
        SEED
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=THREADS,
        pin_memory=False,
        worker_init_fn=seed_worker,
        generator=train_generator,
        persistent_workers=False,
    )

    # --------------------------------------------------------
    # Model / optimizer
    # --------------------------------------------------------

    model = build_model(
        profile,
        device,
    )

    model.train()

    trainable = [
        p
        for p in
        model.parameters()
        if p.requires_grad
    ]

    trainable_count = sum(
        p.numel()
        for p in trainable
    )

    assert trainable_count == 11906

    assert all(
        name.startswith(
            "refinement_core.router."
        )
        for name, p in
        model.named_parameters()
        if p.requires_grad
    )

    for module in (
        model
        .contributor_bank
        .contributors
        .values()
    ):
        assert (
            module.training
            is False
        )

        for p in module.parameters():
            assert (
                p.requires_grad
                is False
            )

    optimizer = torch.optim.Adam(
        trainable,
        lr=LR,
    )

    scheduler = lrs.MultiStepLR(
        optimizer,
        milestones=MILESTONES,
        gamma=LR_GAMMA,
    )

    initial_router_sha = (
        router_digest(
            model
        )
    )

    print(
        "INITIAL_ROUTER_DIGEST="
        f"{initial_router_sha}"
    )

    # --------------------------------------------------------
    # Resume only from this exact formal protocol.
    # --------------------------------------------------------

    start_epoch = 1

    if args.auto_resume:

        latest = find_latest_resume(
            ckpt_dir
        )

        if latest is not None:

            resume_epoch, resume_path = (
                latest
            )

            resume = torch.load(
                resume_path,
                map_location="cpu",
                weights_only=False,
            )

            assert (
                resume["format"]
                ==
                "r27f9g4d_formal_resume_v1"
            )

            assert (
                resume["dataset"]
                ==
                dataset
            )

            assert (
                resume[
                    "protocol_sha256"
                ]
                ==
                protocol_sha
            )

            assert (
                resume[
                    "profile_sha256"
                ]
                ==
                profile_sha
            )

            model.refinement_core.router.load_state_dict(
                resume[
                    "router_state_dict"
                ],
                strict=True,
            )

            optimizer.load_state_dict(
                resume[
                    "optimizer_state_dict"
                ]
            )

            optimizer_to_device(
                optimizer,
                device,
            )

            scheduler.load_state_dict(
                resume[
                    "scheduler_state_dict"
                ]
            )

            train_generator.set_state(
                resume[
                    "train_generator_state"
                ]
            )

            random.setstate(
                resume[
                    "python_random_state"
                ]
            )

            np.random.set_state(
                resume[
                    "numpy_random_state"
                ]
            )

            torch.set_rng_state(
                resume[
                    "torch_rng_state"
                ]
            )

            if (
                torch.cuda.is_available()
                and
                resume[
                    "cuda_rng_state_all"
                ]
                is not None
            ):
                torch.cuda.set_rng_state_all(
                    resume[
                        "cuda_rng_state_all"
                    ]
                )

            start_epoch = (
                int(
                    resume["epoch"]
                )
                + 1
            )

            print(
                "AUTO_RESUME_FROM="
                f"{resume_path}"
            )

            print(
                "AUTO_RESUME_START_EPOCH="
                f"{start_epoch}"
            )

    config_path = (
        run_dir
        / "formal_config.json"
    )

    config = {
        "dataset":
            dataset,

        "run_name":
            run_name,

        "protocol_sha256":
            protocol_sha,

        "profile_sha256":
            profile_sha,

        "trainer_sha256":
            self_sha,

        "seed":
            SEED,

        "epochs":
            EPOCHS,

        "batch_size":
            BATCH_SIZE,

        "patch_size":
            PATCH_SIZE,

        "threads":
            THREADS,

        "optimizer":
            "Adam",

        "lr":
            LR,

        "scheduler":
            {
                "name":
                    "MultiStepLR",

                "milestones":
                    MILESTONES,

                "gamma":
                    LR_GAMMA,
            },

        "objective_mode":
            OBJECTIVE_MODE,

        "goal_kappa":
            GOAL_KAPPA,

        "goal_objective_weight":
            GOAL_OBJECTIVE_WEIGHT,

        "refinement_gamma":
            REFINEMENT_GAMMA,

        "trainable_parameter_count":
            trainable_count,

        "anchor_methods":
            profile[
                "anchor_methods"
            ],

        "slot_order":
            profile[
                "slot_order"
            ],

        "train_manifest":
            str(
                spec[
                    "manifest"
                ]
            ),

        "train_count":
            len(train_set),

        "test_used":
            False,
    }

    config_path.write_text(
        json.dumps(
            config,
            indent=2,
        )
        +
        "\n",
        encoding="utf-8",
    )

    epoch_csv = (
        run_dir
        / "epoch_metrics.csv"
    )

    if (
        start_epoch == 1
        or
        not epoch_csv.exists()
    ):
        with epoch_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "epoch",
                "avg_loss",
                "avg_objective_active",
                "avg_goal_utility",
                "lr_after_scheduler",
                "router_digest",
            ])

    # --------------------------------------------------------
    # Formal training.
    # --------------------------------------------------------

    for epoch in range(
        start_epoch,
        EPOCHS + 1,
    ):

        epoch_start = time.time()

        model.train()

        for module in (
            model
            .contributor_bank
            .contributors
            .values()
        ):
            assert (
                module.training
                is False
            )

        epoch_loss = 0.0
        loss_accum = defaultdict(
            float
        )

        for iteration, batch in enumerate(
            train_loader,
            start=1,
        ):

            x, target, filename = batch

            # Frozen self-supervised contract:
            # target is the same source image.
            input_target_error = float(
                (
                    x
                    -
                    target
                )
                .abs()
                .max()
                .item()
            )

            if input_target_error != 0.0:
                raise RuntimeError(
                    f"{dataset}: "
                    "target != input "
                    f"for {filename}"
                )

            x = x.to(
                device,
                non_blocking=False,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(x)

            I_refined = output[
                "I_refined"
            ]

            I_synthesis = output[
                "I_synthesis"
            ]

            priors = output[
                "priors"
            ]

            weighted_offsets = output[
                "offsets_stage2"
            ]

            route_weights = output[
                "route_weights"
            ]

            como_out = output[
                "como_out"
            ]

            route_sum_error = float(
                (
                    route_weights.sum(
                        dim=1
                    )
                    -
                    1.0
                )
                .abs()
                .max()
                .item()
            )

            if route_sum_error > 1e-6:
                raise RuntimeError(
                    f"route simplex failure: "
                    f"{route_sum_error}"
                )

            loss, loss_dict = (
                compute_total_loss(
                    I_input=x,
                    I_refined=I_refined,
                    I_synthesis=I_synthesis,
                    priors=priors,
                    weighted_offsets=(
                        weighted_offsets
                    ),
                    cas_weights=(
                        route_weights
                    ),
                    attn_offset_only=None,
                    J_phys=None,
                    model=model,
                    epoch=epoch,
                    iteration=iteration,
                    objective_mode=(
                        OBJECTIVE_MODE
                    ),
                    goal_kappa=(
                        GOAL_KAPPA
                    ),
                    goal_objective_weight=(
                        GOAL_OBJECTIVE_WEIGHT
                    ),
                    I_candidate=(
                        como_out[
                            "I_candidate"
                        ]
                    ),
                    proposal_support_weight=(
                        PROPOSAL_SUPPORT_WEIGHT
                    ),
                )
            )

            if not torch.isfinite(
                loss
            ).all():
                raise RuntimeError(
                    "non-finite total loss"
                )

            check_finite_loss_dict(
                loss_dict
            )

            loss.backward()

            grad_tensor_count = 0

            for p in (
                model
                .refinement_core
                .router
                .parameters()
            ):

                if p.grad is None:
                    continue

                if not torch.isfinite(
                    p.grad
                ).all():
                    raise RuntimeError(
                        "non-finite router gradient"
                    )

                grad_tensor_count += 1

            if grad_tensor_count == 0:
                raise RuntimeError(
                    "router received no gradients"
                )

            optimizer.step()

            for p in (
                model
                .refinement_core
                .router
                .parameters()
            ):
                if not torch.isfinite(
                    p
                ).all():
                    raise RuntimeError(
                        "non-finite router parameter"
                    )

            value = float(
                loss.item()
            )

            epoch_loss += value

            for key, item in (
                loss_dict.items()
            ):
                if torch.is_tensor(item):
                    item = float(
                        item.detach().item()
                    )
                else:
                    item = float(item)

                loss_accum[key] += item

            if (
                iteration == 1
                or
                iteration % 500 == 0
                or
                iteration
                ==
                len(train_loader)
            ):

                print(
                    f"[{dataset}] "
                    f"EPOCH={epoch:03d}/{EPOCHS} "
                    f"ITER={iteration:04d}/"
                    f"{len(train_loader):04d} "
                    f"LOSS={value:.8f} "
                    f"OBJ="
                    f"{float(loss_dict['objective_active']):.8f}"
                )

        avg_loss = (
            epoch_loss
            /
            len(train_loader)
        )

        avg_objective = (
            loss_accum[
                "objective_active"
            ]
            /
            len(train_loader)
        )

        avg_goal_utility = (
            loss_accum[
                "goal_utility_v3"
            ]
            /
            len(train_loader)
        )

        scheduler.step()

        current_lr = float(
            optimizer
            .param_groups[0]["lr"]
        )

        current_digest = (
            router_digest(
                model
            )
        )

        elapsed = (
            time.time()
            -
            epoch_start
        )

        print(
            f"[{dataset}] "
            f"EPOCH_COMPLETE={epoch:03d} "
            f"AVG_LOSS={avg_loss:.9f} "
            f"AVG_OBJECTIVE="
            f"{avg_objective:.9f} "
            f"AVG_GOAL_UTILITY="
            f"{avg_goal_utility:.9f} "
            f"LR_NEXT={current_lr:.8g} "
            f"SECONDS={elapsed:.1f}"
        )

        with epoch_csv.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                epoch,
                avg_loss,
                avg_objective,
                avg_goal_utility,
                current_lr,
                current_digest,
            ])

        should_save = (
            epoch % SAVE_EVERY == 0
            or
            epoch == EPOCHS
        )

        if should_save:

            core_path = (
                ckpt_dir
                / f"model_epoch_{epoch}.pth"
            )

            info = save_core_checkpoint(
                model,
                core_path,
                metadata={
                    "dataset":
                        dataset,

                    "epoch":
                        epoch,

                    "seed":
                        SEED,

                    "protocol_sha256":
                        protocol_sha,

                    "profile_sha256":
                        profile_sha,

                    "objective_mode":
                        OBJECTIVE_MODE,

                    "goal_objective_weight":
                        GOAL_OBJECTIVE_WEIGHT,

                    "refinement_gamma":
                        REFINEMENT_GAMMA,

                    "anchor_methods":
                        profile[
                            "anchor_methods"
                        ],

                    "slot_order":
                        profile[
                            "slot_order"
                        ],

                    "test_used":
                        False,

                    "formal_training":
                        True,
                },
            )

            resume_path = (
                ckpt_dir
                / f"resume_epoch_{epoch}.pth"
            )

            save_resume_checkpoint(
                path=resume_path,
                dataset=dataset,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_generator=(
                    train_generator
                ),
                protocol_sha=protocol_sha,
                profile_sha=profile_sha,
            )

            latest_path = (
                run_dir
                / "LATEST_CHECKPOINT.txt"
            )

            latest_path.write_text(
                (
                    f"epoch={epoch}\n"
                    f"core={core_path}\n"
                    f"core_sha256="
                    f"{info['sha256']}\n"
                    f"resume={resume_path}\n"
                ),
                encoding="utf-8",
            )

            print(
                f"[{dataset}] "
                f"SAVED_FORMAL_CHECKPOINT="
                f"{core_path}"
            )

            print(
                f"[{dataset}] "
                f"CHECKPOINT_SHA256="
                f"{info['sha256']}"
            )

    final_core = (
        ckpt_dir
        / "model_epoch_200.pth"
    )

    assert final_core.is_file()

    completed = {
        "status":
            "PASS_FORMAL_TRAINING_COMPLETE",

        "dataset":
            dataset,

        "run_name":
            run_name,

        "final_epoch":
            EPOCHS,

        "seed":
            SEED,

        "protocol_sha256":
            protocol_sha,

        "profile_sha256":
            profile_sha,

        "trainer_sha256":
            self_sha,

        "initial_router_digest":
            initial_router_sha,

        "final_router_digest":
            router_digest(
                model
            ),

        "final_core_checkpoint":
            str(
                final_core
            ),

        "final_core_sha256":
            sha256(
                final_core
            ),

        "test_used":
            False,

        "checkpoint_selection_performed":
            False,
    }

    completed_path.write_text(
        json.dumps(
            completed,
            indent=2,
        )
        +
        "\n",
        encoding="utf-8",
    )

    print()
    print(
        f"FINAL_CORE={final_core}"
    )

    print(
        "FINAL_CORE_SHA256="
        f"{completed['final_core_sha256']}"
    )

    print(
        "CHECKPOINT_SELECTION_PERFORMED=False"
    )

    print(
        "TEST_USED=False"
    )

    print(
        "PASS_R27F9G4D_FORMAL_TRAINING_COMPLETE"
    )


if __name__ == "__main__":
    main()
