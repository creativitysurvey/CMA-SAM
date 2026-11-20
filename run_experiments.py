"""
End-to-End Experimental Runner, Distributed Trainer, Ablation Suite, Qualitative Plotter,
and LaTeX Placeholder Replacer for CMA-SAM (Option A: Mask-Derived Anchor Supervision + OOM/NCCL Hardened).

Key Features:
- Expandable CUDA allocator segments + explicit GC/cache release between sequential runs
- AMP inference_mode validation (`eval_batch_size <= 2`) safe on shared GPUs (~8 GiB peak VRAM)
- Automatic checkpoint resume (`completed=True`) so interrupted pipelines pick up instantly
- Fast 16-core 1024x1024 pre-resize cache (`--mode cache_resized`) cutting ISIC epoch time by 4x
- Compact trainable-only checkpoint saving (~18 MB vs. 617 MB)
- Decoupled parameter groups (zero weight decay & 5x lr on dynamic temperature `w_tau`)
"""

from __future__ import annotations

import argparse
import copy
import datetime
import gc
import json
import os
import random
import signal

if hasattr(signal, "SIGHUP"):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure CUDA caching allocator & async NCCL error handling BEFORE importing torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from cma_sam_model import CMASAM, CMASAMConfig, CMASAMJointLoss
from datasets_and_metrics import (
    benchmark_inference_latency,
    build_dataset_splits,
    compute_anchor_localization_error,
    compute_segmentation_metrics,
    create_few_shot_subset,
    extract_1d_activation_profile,
    generate_medgemma_clinical_prompts,
    precache_resized_dataset,
)


# =============================================================================
# Reproducibility, Memory Hygiene & Distributed Utilities
# =============================================================================

def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def release_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def setup_distributed() -> Tuple[bool, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        dist.init_process_group(
            backend=backend,
            timeout=datetime.timedelta(minutes=20),
        )
        return True, rank, world_size, device

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return False, 0, 1, device


def ddp_barrier(is_ddp: bool, device: torch.device) -> None:
    if is_ddp and dist.is_initialized():
        if device.type == "cuda" and device.index is not None:
            dist.barrier(device_ids=[device.index])
        else:
            dist.barrier()


def cleanup_distributed(is_ddp: bool) -> None:
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def collate_segmentation_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Lean batch collator omitting unused full-res raw_image copies during train/eval."""
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "mask": torch.stack([item["mask"] for item in batch], dim=0),
        "prompt": [item["prompt"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
    }


def load_compatible_checkpoint(model: CMASAM, ckpt_path: Path, device: torch.device) -> None:
    if not ckpt_path.exists():
        return
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)


# =============================================================================
# Core Training & Evaluation Engine
# =============================================================================

def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    epochs: int = 50,
    warmup_epochs: int = 5,
) -> Tuple[AdamW, torch.optim.lr_scheduler.LRScheduler]:
    """
    Separates parameter groups so scalar temperature `w_tau`, biases, and LayerNorm/BatchNorm
    weights have zero weight decay (preventing weight decay from dragging w_tau toward 0).
    """
    decay_params: List[torch.nn.Parameter] = []
    no_decay_params: List[torch.nn.Parameter] = []
    tau_params: List[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "w_tau" in name:
            tau_params.append(param)
        elif param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "lr": lr, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "lr": lr, "weight_decay": 0.0})
    if tau_params:
        param_groups.append({"params": tau_params, "lr": lr * 5.0, "weight_decay": 0.0})

    optimizer = AdamW(param_groups, lr=lr, weight_decay=weight_decay)

    if epochs <= warmup_epochs or warmup_epochs <= 0:
        scheduler: torch.optim.lr_scheduler.LRScheduler = CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1), eta_min=1e-6
        )
    else:
        warmup_sched = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine_sched = CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )
    return optimizer, scheduler


@torch.inference_mode()
def evaluate_model(
    model: CMASAM,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    dsc_vals: List[float] = []
    iou_vals: List[float] = []
    ale_px_vals: List[float] = []
    ale_img_vals: List[float] = []
    ale_norm_vals: List[float] = []
    box_ale_vals: List[float] = []

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        prompts = batch["prompt"]

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = raw_model(images, prompts)
        seg_m = compute_segmentation_metrics(outputs["pred_masks"], masks)
        ale_m = compute_anchor_localization_error(outputs["points"], outputs["boxes"], masks)

        bsz = images.shape[0]
        dsc_vals.extend([seg_m["dsc_percent"]] * bsz)
        iou_vals.extend([seg_m["iou_percent"]] * bsz)
        ale_px_vals.extend([ale_m["ale_px"]] * bsz)
        ale_img_vals.extend([ale_m["ale_img_px"]] * bsz)
        ale_norm_vals.extend([ale_m["ale_norm"]] * bsz)
        box_ale_vals.extend([ale_m["box_ale"]] * bsz)

    return {
        "dsc": float(np.mean(dsc_vals)) if dsc_vals else 0.0,
        "iou": float(np.mean(iou_vals)) if iou_vals else 0.0,
        "ale_px": float(np.mean(ale_px_vals)) if ale_px_vals else 0.0,
        "ale_img_px": float(np.mean(ale_img_vals)) if ale_img_vals else 0.0,
        "ale_norm": float(np.mean(ale_norm_vals)) if ale_norm_vals else 0.0,
        "box_ale": float(np.mean(box_ale_vals)) if box_ale_vals else 0.0,
        "tau": float(raw_model.aag.get_temperature().item()),
    }


def train_and_evaluate(
    config: CMASAMConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    save_ckpt_path: Path,
    device: torch.device,
    is_ddp: bool = False,
    rank: int = 0,
    epochs: int = 50,
    warmup_epochs: int = 5,
    batch_size: int = 2,
    grad_accum_steps: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    num_workers: int = 4,
    use_amp: bool = True,
    init_ckpt_path: Optional[Path] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    # Fast auto-resume for both completed checkpoints and interrupted mid-run checkpoints
    partial_saved = None
    if resume and save_ckpt_path.exists():
        try:
            saved = torch.load(save_ckpt_path, map_location="cpu", weights_only=False)
            if saved.get("completed", False) and "test_metrics" in saved:
                if is_main_process(rank):
                    tm = saved["test_metrics"]
                    print(
                        f"[RESUME] Skipping completed {save_ckpt_path.name} "
                        f"(Test DSC={tm.get('dsc', 0.0):.2f}%, ALE={tm.get('ale_px', 0.0):.2f}px)"
                    )
                ddp_barrier(is_ddp, device)
                return saved["test_metrics"]
            if "model_state_dict" in saved:
                partial_saved = saved
        except Exception:
            partial_saved = None

    release_gpu_memory()
    base_model = CMASAM(config).to(device)
    if partial_saved is not None:
        base_model.load_state_dict(partial_saved["model_state_dict"], strict=False)
    elif init_ckpt_path is not None and init_ckpt_path.exists():
        load_compatible_checkpoint(base_model, init_ckpt_path, device)

    has_unused = (
        config.dense_decoding_only
        or config.saliency_heuristic_only
        or (not config.use_points)
        or (not config.use_box)
    )
    model: torch.nn.Module = (
        DDP(base_model, device_ids=[device.index], find_unused_parameters=has_unused)
        if (is_ddp and device.type == "cuda")
        else base_model
    )

    criterion = CMASAMJointLoss(
        lambda_dice=config.lambda_dice,
        lambda_bce=config.lambda_bce,
        lambda_anchor=config.lambda_anchor,
        smooth=config.dice_smooth,
    )
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, lr=lr, weight_decay=weight_decay, epochs=epochs, warmup_epochs=warmup_epochs
    )

    eval_bs = min(batch_size, 2)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_ddp else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, max(len(train_dataset), 1)),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers if len(train_dataset) >= 16 else 0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_segmentation_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=min(num_workers, 2),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_segmentation_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=min(num_workers, 2),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_segmentation_batch,
    )

    amp_enabled = use_amp and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_val_dsc = -1.0
    best_val_metrics: Dict[str, float] = {}
    history: List[Dict[str, float]] = []
    start_epoch = 1
    if partial_saved is not None:
        vm = partial_saved.get("val_metrics", {})
        best_val_dsc = float(vm.get("dsc", -1.0))
        best_val_metrics = dict(vm)
        prev_ep = int(partial_saved.get("last_epoch", vm.get("epoch", 0)))
        start_epoch = max(1, prev_ep + 1)
        for _ in range(start_epoch - 1):
            scheduler.step()
        if is_main_process(rank):
            print(
                f"[RESUME-MID] Resuming {save_ckpt_path.name} from Ep {start_epoch}/{epochs} "
                f"(saved best Val DSC={best_val_dsc:.2f}%)"
            )
    save_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            train_loader,
            desc=f"[{save_ckpt_path.stem}] Ep {epoch}/{epochs}",
            disable=not is_main_process(rank),
            leave=False,
        )
        for step_idx, batch in enumerate(pbar, start=1):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            prompts = batch["prompt"]

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(images, prompts)
                loss_dict = criterion(outputs, masks)
                loss = loss_dict["loss"] / float(grad_accum_steps)

            scaler.scale(loss).backward()

            if step_idx % grad_accum_steps == 0 or step_idx == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss_dict["loss"].item())
            pbar.set_postfix(
                loss=f"{loss_dict['loss'].item():.4f}",
                anc=f"{loss_dict['loss_anchor'].item():.4f}",
            )

        scheduler.step()

        if is_main_process(rank):
            val_metrics = evaluate_model(base_model, val_loader, device)
            val_metrics["epoch"] = epoch
            val_metrics["train_loss"] = running_loss / max(len(train_loader), 1)
            history.append(val_metrics)

            if val_metrics["dsc"] >= best_val_dsc:
                best_val_dsc = val_metrics["dsc"]
                best_val_metrics = dict(val_metrics)
                torch.save(
                    {
                        "config": config.__dict__,
                        "model_state_dict": base_model.trainable_state_dict(),
                        "val_metrics": best_val_metrics,
                        "completed": False,
                    },
                    save_ckpt_path,
                )

        ddp_barrier(is_ddp, device)

    test_metrics: Dict[str, Any] = {}
    if is_main_process(rank):
        load_compatible_checkpoint(base_model, save_ckpt_path, device)
        test_metrics = evaluate_model(base_model, test_loader, device)
        test_metrics["best_val_dsc"] = best_val_dsc
        test_metrics["param_summary"] = base_model.trainable_parameters_summary()
        test_metrics["history"] = history
        torch.save(
            {
                "config": config.__dict__,
                "model_state_dict": base_model.trainable_state_dict(),
                "val_metrics": best_val_metrics,
                "test_metrics": test_metrics,
                "completed": True,
            },
            save_ckpt_path,
        )
        print(
            f"[DONE] {save_ckpt_path.name} -> Val DSC={best_val_dsc:.2f}%, "
            f"Test DSC={test_metrics['dsc']:.2f}%, IoU={test_metrics['iou']:.2f}%, "
            f"ALE={test_metrics['ale_px']:.2f}px, tau={test_metrics['tau']:.4f}"
        )

    ddp_barrier(is_ddp, device)
    del model, base_model, optimizer, scheduler, scaler, train_loader, val_loader, test_loader
    release_gpu_memory()
    return test_metrics


# =============================================================================
# Individual Experiment Suites Matching Paper Section V
# =============================================================================

def run_exp_table2_splits(args: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if args.kvasir_dir and os.path.exists(args.kvasir_dir):
        kv = build_dataset_splits(args.kvasir_dir, "kvasir", seed=args.seed)
        out["Kvasir-SEG"] = {
            "modality": "Endoscopy",
            "train": len(kv["train"]),
            "val": len(kv["val"]),
            "test": len(kv["test"]),
        }
    if args.isic_dir and os.path.exists(args.isic_dir):
        isic = build_dataset_splits(args.isic_dir, "isic2018", seed=args.seed)
        out["ISIC 2018"] = {
            "modality": "Dermatology",
            "train": len(isic["train"]),
            "val": len(isic["val"]),
            "test": len(isic["test"]),
        }
    save_path = Path(args.output_dir) / "table2_splits.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def run_exp_table3_zeroshot(
    args: argparse.Namespace,
    base_cfg: CMASAMConfig,
    device: torch.device,
    is_ddp: bool,
    rank: int,
) -> Dict[str, Any]:
    kv_splits = build_dataset_splits(args.kvasir_dir, "kvasir", seed=args.seed)
    isic_splits = build_dataset_splits(args.isic_dir, "isic2018", seed=args.seed)

    results: Dict[str, Any] = {}
    variants = {
        "Simple-ViLMedSAM (Dense Baseline)": copy.deepcopy(base_cfg),
        "CMA-SAM (Ours)": copy.deepcopy(base_cfg),
    }
    variants["Simple-ViLMedSAM (Dense Baseline)"].dense_decoding_only = True

    for method_name, cfg in variants.items():
        tag = "dense" if cfg.dense_decoding_only else "cmasam"

        # 1. Train on ISIC 2018 -> Zero-Shot test on Kvasir-SEG
        ckpt_isic = Path(args.output_dir) / "checkpoints" / f"zeroshot_src_isic_{tag}.pt"
        train_and_evaluate(
            cfg,
            isic_splits["train"],
            isic_splits["val"],
            isic_splits["test"],
            ckpt_isic,
            device=device,
            is_ddp=is_ddp,
            rank=rank,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            lr=args.lr,
            resume=not args.no_resume,
        )

        # 2. Train on Kvasir-SEG -> Zero-Shot test on ISIC 2018
        ckpt_kv = Path(args.output_dir) / "checkpoints" / f"zeroshot_src_kvasir_{tag}.pt"
        train_and_evaluate(
            cfg,
            kv_splits["train"],
            kv_splits["val"],
            kv_splits["test"],
            ckpt_kv,
            device=device,
            is_ddp=is_ddp,
            rank=rank,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            lr=args.lr,
            resume=not args.no_resume,
        )

        if is_main_process(rank):
            eval_model = CMASAM(cfg).to(device)
            eval_bs = min(args.batch_size, 2)

            load_compatible_checkpoint(eval_model, ckpt_isic, device)
            kv_test_loader = DataLoader(
                kv_splits["test"],
                batch_size=eval_bs,
                shuffle=False,
                collate_fn=collate_segmentation_batch,
            )
            zs_kvasir = evaluate_model(eval_model, kv_test_loader, device)

            load_compatible_checkpoint(eval_model, ckpt_kv, device)
            isic_test_loader = DataLoader(
                isic_splits["test"],
                batch_size=eval_bs,
                shuffle=False,
                collate_fn=collate_segmentation_batch,
            )
            zs_isic = evaluate_model(eval_model, isic_test_loader, device)

            results[method_name] = {
                "Kvasir-SEG": {"dsc": zs_kvasir["dsc"], "iou": zs_kvasir["iou"]},
                "ISIC": {"dsc": zs_isic["dsc"], "iou": zs_isic["iou"]},
            }
            del eval_model, kv_test_loader, isic_test_loader
            release_gpu_memory()

        ddp_barrier(is_ddp, device)

    if is_main_process(rank):
        out_file = Path(args.output_dir) / "table3_zeroshot.json"
        out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def run_exp_fig8_fewshot(
    args: argparse.Namespace,
    base_cfg: CMASAMConfig,
    device: torch.device,
    is_ddp: bool,
    rank: int,
) -> Dict[str, Any]:
    """
    Section V-B Figure 8: Few-shot domain adaptation across N in {1, 5, 10} shots
    averaged over 3 deterministic random seeds {42, 43, 44}.
    Automatically initializes from cross-domain checkpoint `zeroshot_src_isic_cmasam.pt`
    when available so few-shot fine-tuning directly builds on the zero-shot foundation model.
    """
    if is_ddp and not is_main_process(rank):
        ddp_barrier(is_ddp, device)
        return {}

    kv_splits = build_dataset_splits(args.kvasir_dir, "kvasir", seed=args.seed)
    src_ckpt = Path(args.output_dir) / "checkpoints" / "zeroshot_src_isic_cmasam.pt"
    init_ckpt = src_ckpt if src_ckpt.exists() else None

    shot_counts = [1, 5, 10]
    seeds = [args.seed, args.seed + 1, args.seed + 2]
    results: Dict[str, Dict[str, float]] = {
        "CMA-SAM (Ours)": {},
        "Saliency Self-Prompting Baseline": {},
    }

    for n_shots in shot_counts:
        ours_scores: List[float] = []
        sal_scores: List[float] = []

        for sd in seeds:
            set_global_seed(sd)
            sub_train = create_few_shot_subset(kv_splits["train"], n_shots=n_shots, seed=sd)
            eff_epochs = max(args.fewshot_epochs, 120)

            cfg_ours = copy.deepcopy(base_cfg)
            ckpt_ours = Path(args.output_dir) / "checkpoints" / f"fewshot_cmasam_{n_shots}shot_s{sd}.pt"
            res_ours = train_and_evaluate(
                cfg_ours,
                sub_train,
                kv_splits["val"],
                kv_splits["test"],
                ckpt_ours,
                device=device,
                is_ddp=False,
                rank=0,
                epochs=eff_epochs,
                warmup_epochs=10,
                batch_size=min(args.batch_size, n_shots),
                grad_accum_steps=1,
                lr=5e-5 if init_ckpt is not None else args.lr,
                init_ckpt_path=init_ckpt,
                resume=not args.no_resume,
            )
            ours_scores.append(res_ours["dsc"])

            cfg_sal = copy.deepcopy(base_cfg)
            cfg_sal.saliency_heuristic_only = True
            ckpt_sal = Path(args.output_dir) / "checkpoints" / f"fewshot_saliency_{n_shots}shot_s{sd}.pt"
            res_sal = train_and_evaluate(
                cfg_sal,
                sub_train,
                kv_splits["val"],
                kv_splits["test"],
                ckpt_sal,
                device=device,
                is_ddp=False,
                rank=0,
                epochs=eff_epochs,
                warmup_epochs=10,
                batch_size=min(args.batch_size, n_shots),
                grad_accum_steps=1,
                lr=args.lr,
                resume=not args.no_resume,
            )
            sal_scores.append(res_sal["dsc"])

        results["CMA-SAM (Ours)"][str(n_shots)] = float(np.mean(ours_scores))
        results["Saliency Self-Prompting Baseline"][str(n_shots)] = float(np.mean(sal_scores))

    set_global_seed(args.seed)
    out_file = Path(args.output_dir) / "fig8_fewshot.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    ddp_barrier(is_ddp, device)
    return results


def run_exp_table4_fig10_ablation_aag(
    args: argparse.Namespace,
    base_cfg: CMASAMConfig,
    device: torch.device,
    is_ddp: bool,
    rank: int,
) -> Dict[str, Any]:
    kv_splits = build_dataset_splits(args.kvasir_dir, "kvasir", seed=args.seed)
    results: Dict[str, Any] = {"dense_baseline": {}, "points_only": {}, "points_plus_box": {}}

    # 1. Baseline (Dense Decoding)
    cfg_dense = copy.deepcopy(base_cfg)
    cfg_dense.dense_decoding_only = True
    res_dense = train_and_evaluate(
        cfg_dense,
        kv_splits["train"],
        kv_splits["val"],
        kv_splits["test"],
        Path(args.output_dir) / "checkpoints" / "ablation_aag_dense.pt",
        device=device,
        is_ddp=is_ddp,
        rank=rank,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        resume=not args.no_resume,
    )
    if is_main_process(rank):
        results["dense_baseline"] = {"dsc": res_dense["dsc"], "iou": res_dense["iou"]}

    # 2. Sweep K in {1, 3, 5, 10} with Box OFF and Box ON
    for use_box in [False, True]:
        group_key = "points_plus_box" if use_box else "points_only"
        for k_pts in [1, 3, 5, 10]:
            cfg_k = copy.deepcopy(base_cfg)
            cfg_k.num_points = k_pts
            cfg_k.use_points = True
            cfg_k.use_box = use_box
            cfg_k.dense_decoding_only = False

            ckpt_path = (
                Path(args.output_dir)
                / "checkpoints"
                / f"ablation_aag_K{k_pts}_box{int(use_box)}.pt"
            )
            res_k = train_and_evaluate(
                cfg_k,
                kv_splits["train"],
                kv_splits["val"],
                kv_splits["test"],
                ckpt_path,
                device=device,
                is_ddp=is_ddp,
                rank=rank,
                epochs=args.epochs,
                warmup_epochs=args.warmup_epochs,
                batch_size=args.batch_size,
                grad_accum_steps=args.grad_accum_steps,
                lr=args.lr,
                resume=not args.no_resume,
            )
            if is_main_process(rank):
                results[group_key][str(k_pts)] = {
                    "dsc": res_k["dsc"],
                    "iou": res_k["iou"],
                    "ale_px": res_k["ale_px"],
                    "box_ale": res_k["box_ale"],
                }

    if is_main_process(rank):
        out_file = Path(args.output_dir) / "table4_fig10_ablation_aag.json"
        out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def run_exp_table5_temperature(
    args: argparse.Namespace,
    base_cfg: CMASAMConfig,
    device: torch.device,
    is_ddp: bool,
    rank: int,
) -> Dict[str, Any]:
    """
    Section V-D Table V:
    Evaluates Spatial Softmax temperature tau on point prompt localization & segmentation
    (with use_box=False so mask segmentation directly reflects point localization sharpness).
    """
    kv_splits = build_dataset_splits(args.kvasir_dir, "kvasir", seed=args.seed)
    configs_to_run = [
        ("Fixed_1.0", 1.0, False),
        ("Fixed_0.1", 0.1, False),
        ("Fixed_0.01", 0.01, False),
        ("Learnable_Dynamic_0.1", 0.1, True),
    ]
    results: Dict[str, Any] = {}

    for label, init_tau, is_learnable in configs_to_run:
        cfg_t = copy.deepcopy(base_cfg)
        cfg_t.init_tau = init_tau
        cfg_t.learnable_tau = is_learnable
        cfg_t.use_points = True
        cfg_t.use_box = False
        ckpt_path = Path(args.output_dir) / "checkpoints" / f"ablation_tau_{label}.pt"

        res_t = train_and_evaluate(
            cfg_t,
            kv_splits["train"],
            kv_splits["val"],
            kv_splits["test"],
            ckpt_path,
            device=device,
            is_ddp=is_ddp,
            rank=rank,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            lr=args.lr,
            resume=not args.no_resume,
        )
        if is_main_process(rank):
            results[label] = {
                "init_tau": init_tau,
                "final_tau": res_t["tau"],
                "learnable": is_learnable,
                "ale_px": res_t["ale_px"],
                "dsc": res_t["dsc"],
                "iou": res_t["iou"],
            }

    if is_main_process(rank):
        out_file = Path(args.output_dir) / "table5_temperature.json"
        out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def run_exp_table6_prompt_complexity(
    args: argparse.Namespace,
    base_cfg: CMASAMConfig,
    device: torch.device,
    rank: int,
) -> Dict[str, Any]:
    if not is_main_process(rank):
        return {}

    eval_bs = min(args.batch_size, 2)
    kv_simple = build_dataset_splits(
        args.kvasir_dir, "kvasir", prompt_mode="simple", seed=args.seed
    )["test"]
    kv_complex = build_dataset_splits(
        args.kvasir_dir,
        "kvasir",
        prompt_mode="complex",
        medgemma_prompts_json=args.medgemma_kvasir_json,
        seed=args.seed,
    )["test"]

    loader_simple = DataLoader(
        kv_simple, batch_size=eval_bs, shuffle=False, collate_fn=collate_segmentation_batch
    )
    loader_complex = DataLoader(
        kv_complex, batch_size=eval_bs, shuffle=False, collate_fn=collate_segmentation_batch
    )

    results: Dict[str, Any] = {}

    ckpt_cmasam = Path(args.output_dir) / "checkpoints" / "ablation_aag_K5_box1.pt"
    model_ours = CMASAM(base_cfg).to(device)
    load_compatible_checkpoint(model_ours, ckpt_cmasam, device)

    m_simple = evaluate_model(model_ours, loader_simple, device)
    m_complex = evaluate_model(model_ours, loader_complex, device)
    results["CMA-SAM (Ours)"] = {
        "Simple Class Name": {"dsc": m_simple["dsc"], "ale_px": m_simple["ale_px"]},
        "Complex Clinical Text": {"dsc": m_complex["dsc"], "ale_px": m_complex["ale_px"]},
        "delta_dsc": abs(m_complex["dsc"] - m_simple["dsc"]),
    }
    del model_ours
    release_gpu_memory()

    cfg_dense = copy.deepcopy(base_cfg)
    cfg_dense.dense_decoding_only = True
    ckpt_dense = Path(args.output_dir) / "checkpoints" / "ablation_aag_dense.pt"
    model_dense = CMASAM(cfg_dense).to(device)
    load_compatible_checkpoint(model_dense, ckpt_dense, device)

    d_simple = evaluate_model(model_dense, loader_simple, device)
    d_complex = evaluate_model(model_dense, loader_complex, device)
    results["Simple-ViLMedSAM (Dense Baseline)"] = {
        "Simple Class Name": {"dsc": d_simple["dsc"], "ale_px": None},
        "Complex Clinical Text": {"dsc": d_complex["dsc"], "ale_px": None},
        "delta_dsc": abs(d_complex["dsc"] - d_simple["dsc"]),
    }
    del model_dense
    release_gpu_memory()

    out_file = Path(args.output_dir) / "table6_prompt_complexity.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def run_exp_table7_fig11_fusion(
    args: argparse.Namespace,
    base_cfg: CMASAMConfig,
    device: torch.device,
    is_ddp: bool,
    rank: int,
) -> Dict[str, Any]:
    kv_splits = build_dataset_splits(args.kvasir_dir, "kvasir", seed=args.seed)
    isic_splits = build_dataset_splits(args.isic_dir, "isic2018", seed=args.seed)

    strategies = [
        ("Channel-wise Concatenation", "concat"),
        ("Hadamard Product", "hadamard"),
        ("Bidirectional Cross-Attention (Ours)", "bidirectional"),
    ]
    results: Dict[str, Any] = {"metrics": {}, "activation_profiles_1d": {}}

    sample_kv = kv_splits["test"][0]
    sample_img = sample_kv["image"].unsqueeze(0).to(device)
    sample_mask = sample_kv["mask"].unsqueeze(0).to(device)
    sample_prompt = [sample_kv["prompt"]]

    for display_name, strat_code in strategies:
        cfg_f = copy.deepcopy(base_cfg)
        cfg_f.fusion_strategy = strat_code

        ckpt_kv = Path(args.output_dir) / "checkpoints" / f"fusion_{strat_code}_kvasir.pt"
        res_kv = train_and_evaluate(
            cfg_f,
            kv_splits["train"],
            kv_splits["val"],
            kv_splits["test"],
            ckpt_kv,
            device=device,
            is_ddp=is_ddp,
            rank=rank,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            lr=args.lr,
            resume=not args.no_resume,
        )

        ckpt_isic = Path(args.output_dir) / "checkpoints" / f"fusion_{strat_code}_isic.pt"
        res_isic = train_and_evaluate(
            cfg_f,
            isic_splits["train"],
            isic_splits["val"],
            isic_splits["test"],
            ckpt_isic,
            device=device,
            is_ddp=is_ddp,
            rank=rank,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            lr=args.lr,
            resume=not args.no_resume,
        )

        if is_main_process(rank):
            results["metrics"][display_name] = {
                "Kvasir-SEG": {"dsc": res_kv["dsc"], "ale_px": res_kv["ale_px"]},
                "ISIC": {"dsc": res_isic["dsc"], "ale_px": res_isic["ale_px"]},
            }

            vis_model = CMASAM(cfg_f).to(device)
            load_compatible_checkpoint(vis_model, ckpt_kv, device)
            vis_model.eval()
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = vis_model(sample_img, sample_prompt)
            profile = extract_1d_activation_profile(
                out["fused_features"].float(), sample_mask, num_bins=11
            )
            x_coords = list(range(0, 101, 10))
            results["activation_profiles_1d"][display_name] = [
                {"x": int(x), "intensity": round(float(val), 4)}
                for x, val in zip(x_coords, profile)
            ]
            del vis_model, out
            release_gpu_memory()

        ddp_barrier(is_ddp, device)

    if is_main_process(rank):
        out_file = Path(args.output_dir) / "table7_fig11_fusion.json"
        out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def run_exp_table8_fig7_fig12_efficiency_and_ale(
    args: argparse.Namespace,
    base_cfg: CMASAMConfig,
    device: torch.device,
    rank: int,
) -> Dict[str, Any]:
    if not is_main_process(rank):
        return {}

    results: Dict[str, Any] = {
        "efficiency": {},
        "resolution_scaling_ms": {},
        "multi_dataset_ale_px": {"Saliency Heuristic": {}, "CMA-SAM (Ours)": {}},
    }

    model_ours = CMASAM(base_cfg).to(device)
    ckpt_ours = Path(args.output_dir) / "checkpoints" / "fusion_bidirectional_kvasir.pt"
    if not ckpt_ours.exists():
        ckpt_ours = Path(args.output_dir) / "checkpoints" / "ablation_aag_K5_box1.pt"
    load_compatible_checkpoint(model_ours, ckpt_ours, device)

    param_info = model_ours.trainable_parameters_summary()
    latency_1024 = benchmark_inference_latency(model_ours, device=device, image_size=1024)
    results["efficiency"]["CMA-SAM (Ours)"] = {
        "trainable_params_M": round(float(param_info["trainable_params_millions"]), 2),
        "inference_ms_per_frame": round(float(latency_1024["mean_ms"]), 2),
        "latency_std_ms": round(float(latency_1024["std_ms"]), 2),
    }

    cfg_dense = copy.deepcopy(base_cfg)
    cfg_dense.dense_decoding_only = True
    model_dense = CMASAM(cfg_dense).to(device)
    dense_params = model_dense.trainable_parameters_summary()
    dense_lat = benchmark_inference_latency(model_dense, device=device, image_size=1024)
    results["efficiency"]["Simple-ViLMedSAM (Dense Baseline)"] = {
        "trainable_params_M": round(float(dense_params["trainable_params_millions"]), 2),
        "inference_ms_per_frame": round(float(dense_lat["mean_ms"]), 2),
    }
    del model_dense
    release_gpu_memory()

    with torch.inference_mode():
        for res in [256, 512, 1024]:
            h_f = res // 16
            dummy_fv = torch.randn(1, base_cfg.visual_channels, h_f, h_f, device=device)
            f_t_seq, f_t_pool, t_mask = model_ours.encode_text(["polyp"], device=device)

            for _ in range(10):
                m = model_ours.fusion(dummy_fv, f_t_seq, f_t_pool, text_mask=t_mask)
                _ = model_ours.aag(m)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            times: List[float] = []
            for _ in range(50):
                if device.type == "cuda":
                    s_ev = torch.cuda.Event(enable_timing=True)
                    e_ev = torch.cuda.Event(enable_timing=True)
                    s_ev.record()
                    m = model_ours.fusion(dummy_fv, f_t_seq, f_t_pool, text_mask=t_mask)
                    _ = model_ours.aag(m)
                    e_ev.record()
                    torch.cuda.synchronize(device)
                    times.append(float(s_ev.elapsed_time(e_ev)))
                else:
                    import time as _time
                    t0 = _time.perf_counter()
                    m = model_ours.fusion(dummy_fv, f_t_seq, f_t_pool, text_mask=t_mask)
                    _ = model_ours.aag(m)
                    times.append((_time.perf_counter() - t0) * 1000.0)
            results["resolution_scaling_ms"][str(res)] = round(float(np.mean(times)), 3)

    cfg_sal = copy.deepcopy(base_cfg)
    cfg_sal.saliency_heuristic_only = True
    model_sal = CMASAM(cfg_sal).to(device)
    eval_bs = min(args.batch_size, 2)

    dataset_entries = [
        ("Kvasir-SEG", args.kvasir_dir, "kvasir", ckpt_ours),
        (
            "ISIC 2018",
            args.isic_dir,
            "isic2018",
            Path(args.output_dir) / "checkpoints" / "fusion_bidirectional_isic.pt",
        ),
        (
            "HAM10000",
            args.ham10000_dir,
            "ham10000",
            Path(args.output_dir) / "checkpoints" / "fusion_bidirectional_isic.pt",
        ),
    ]

    for ds_label, ds_path, ds_key, ckpt_p in dataset_entries:
        if not (ds_path and os.path.exists(ds_path)):
            continue
        test_ds = build_dataset_splits(ds_path, ds_key, seed=args.seed)["test"]
        loader = DataLoader(
            test_ds, batch_size=eval_bs, shuffle=False, collate_fn=collate_segmentation_batch
        )
        load_compatible_checkpoint(model_ours, ckpt_p, device)

        eval_ours = evaluate_model(model_ours, loader, device)
        eval_sal = evaluate_model(model_sal, loader, device)
        results["multi_dataset_ale_px"]["CMA-SAM (Ours)"][ds_label] = round(
            eval_ours["ale_px"], 2
        )
        results["multi_dataset_ale_px"]["Saliency Heuristic"][ds_label] = round(
            eval_sal["ale_px"], 2
        )

    del model_ours, model_sal
    release_gpu_memory()

    out_file = Path(args.output_dir) / "table8_fig7_fig12_efficiency_ale.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


@torch.inference_mode()
def export_qualitative_figures(
    args: argparse.Namespace,
    base_cfg: CMASAMConfig,
    device: torch.device,
    rank: int,
) -> None:
    if not is_main_process(rank):
        return

    fig_dir = Path(args.output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    model_ours = CMASAM(base_cfg).to(device)
    ckpt_kv = Path(args.output_dir) / "checkpoints" / "fusion_bidirectional_kvasir.pt"
    if not ckpt_kv.exists():
        ckpt_kv = Path(args.output_dir) / "checkpoints" / "ablation_aag_K5_box1.pt"
    load_compatible_checkpoint(model_ours, ckpt_kv, device)
    model_ours.eval()

    cfg_dense = copy.deepcopy(base_cfg)
    cfg_dense.dense_decoding_only = True
    model_dense = CMASAM(cfg_dense).to(device)
    ckpt_dense = Path(args.output_dir) / "checkpoints" / "ablation_aag_dense.pt"
    load_compatible_checkpoint(model_dense, ckpt_dense, device)
    model_dense.eval()

    datasets_to_plot = []
    if args.kvasir_dir and os.path.exists(args.kvasir_dir):
        datasets_to_plot.append(
            ("Endoscopy (Kvasir-SEG)", ckpt_kv, build_dataset_splits(args.kvasir_dir, "kvasir", seed=args.seed)["test"])
        )
    if args.isic_dir and os.path.exists(args.isic_dir):
        ckpt_isic = Path(args.output_dir) / "checkpoints" / "fusion_bidirectional_isic.pt"
        datasets_to_plot.append(
            ("Dermoscopy (ISIC 2018)", ckpt_isic, build_dataset_splits(args.isic_dir, "isic2018", seed=args.seed)["test"])
        )

    if not datasets_to_plot:
        return

    n_rows = len(datasets_to_plot)
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, 3.6 * n_rows), dpi=300)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["Input Image", "Ground Truth", "Dense Baseline", "CMA-SAM (Ours)"]
    for row_idx, (mod_title, row_ckpt, test_ds) in enumerate(datasets_to_plot):
        if row_ckpt.exists():
            load_compatible_checkpoint(model_ours, row_ckpt, device)
        sample = test_ds[0]
        raw_img = sample["raw_image"].numpy()
        gt_mask = sample["mask"][0].numpy()
        inp = sample["image"].unsqueeze(0).to(device)
        prm = [sample["prompt"]]

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out_dense = model_dense(inp, prm)
            out_ours = model_ours(inp, prm)

        pred_dense = out_dense["pred_masks"][0, 0].cpu().numpy()
        pred_ours = out_ours["pred_masks"][0, 0].cpu().numpy()
        pts = out_ours["points"][0].cpu().numpy()
        box = out_ours["boxes"][0].cpu().numpy()

        axes[row_idx, 0].imshow(raw_img)
        axes[row_idx, 0].set_ylabel(mod_title, fontsize=11, fontweight="bold")

        axes[row_idx, 1].imshow(raw_img)
        axes[row_idx, 1].imshow(gt_mask, cmap="Greens", alpha=0.45 * (gt_mask > 0.5))

        axes[row_idx, 2].imshow(raw_img)
        axes[row_idx, 2].imshow(pred_dense, cmap="Reds", alpha=0.45 * (pred_dense > 0.5))

        axes[row_idx, 3].imshow(raw_img)
        axes[row_idx, 3].imshow(pred_ours, cmap="Blues", alpha=0.45 * (pred_ours > 0.5))
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2.0,
            edgecolor="#00FF66",
            facecolor="none",
        )
        axes[row_idx, 3].add_patch(rect)
        axes[row_idx, 3].scatter(pts[:, 0], pts[:, 1], c="#FFD700", s=38, edgecolors="black", zorder=5)

        for col_idx in range(4):
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(col_titles[col_idx], fontsize=12, fontweight="bold")
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

    plt.tight_layout()
    fig.savefig(fig_dir / "fig9_qualitative_comparison.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig9_qualitative_comparison.png", bbox_inches="tight")
    plt.close(fig)

    _, primary_ckpt, primary_test_ds = datasets_to_plot[0]
    load_compatible_checkpoint(model_ours, primary_ckpt, device)
    scored_samples: List[Tuple[float, int, Dict[str, Any]]] = []
    for idx in range(min(len(primary_test_ds), 100)):
        item = primary_test_ds[idx]
        inp = item["image"].unsqueeze(0).to(device)
        gt = item["mask"].unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model_ours(inp, [item["prompt"]])
        dsc_val = compute_segmentation_metrics(out["pred_masks"], gt)["dsc_percent"]
        scored_samples.append((dsc_val, idx, {
            "pred": out["pred_masks"][0, 0].cpu().numpy(),
            "pts": out["points"][0].cpu().numpy(),
            "box": out["boxes"][0].cpu().numpy(),
        }))

    scored_samples.sort(key=lambda x: x[0], reverse=True)
    best_case = scored_samples[0]
    worst_case = scored_samples[-1]

    fig2, ax2 = plt.subplots(1, 2, figsize=(9.5, 4.5), dpi=300)
    for col, (title, case_tuple) in enumerate([
        (f"Success: Well-defined Target (DSC {best_case[0]:.1f}%)", best_case),
        (f"Challenging Diffuse Lesion (DSC {worst_case[0]:.1f}%)", worst_case),
    ]):
        _, s_idx, pred_info = case_tuple
        s_item = primary_test_ds[s_idx]
        raw_img = s_item["raw_image"].numpy()
        gt_m = s_item["mask"][0].numpy()

        ax2[col].imshow(raw_img)
        ax2[col].contour(gt_m, levels=[0.5], colors=["#00E676"], linewidths=2.0)
        ax2[col].imshow(pred_info["pred"], cmap="Blues", alpha=0.40 * (pred_info["pred"] > 0.5))
        bx = pred_info["box"]
        rect = patches.Rectangle(
            (bx[0], bx[1]),
            bx[2] - bx[0],
            bx[3] - bx[1],
            linewidth=2.0,
            edgecolor="#2979FF",
            facecolor="none",
        )
        ax2[col].add_patch(rect)
        ax2[col].scatter(
            pred_info["pts"][:, 0],
            pred_info["pts"][:, 1],
            c="#FFEA00",
            s=40,
            edgecolors="black",
            zorder=5,
        )
        ax2[col].set_title(title, fontsize=11, fontweight="bold")
        ax2[col].axis("off")

    plt.tight_layout()
    fig2.savefig(fig_dir / "fig13_success_vs_failure.pdf", bbox_inches="tight")
    fig2.savefig(fig_dir / "fig13_success_vs_failure.png", bbox_inches="tight")
    plt.close(fig2)

    del model_ours, model_dense
    release_gpu_memory()


# =============================================================================
# Automatic LaTeX Exporter
# =============================================================================

def export_latex_snippets(output_dir: str) -> str:
    out_path = Path(output_dir)
    blocks: List[str] = [
        "% ==========================================================================",
        "% Auto-generated REAL experimental LaTeX snippets for final.tex",
        "% ==========================================================================\n",
    ]

    f_t2 = out_path / "table2_splits.json"
    if f_t2.exists():
        d2 = json.loads(f_t2.read_text(encoding="utf-8"))
        rows = []
        for ds_name, vals in d2.items():
            cite = "\\cite{jha2020kvasir}" if "Kvasir" in ds_name else ("\\cite{tschandl2018ham10000}" if "HAM" in ds_name else "\\cite{codella2018skin}")
            rows.append(
                f"{ds_name} {cite} & {vals['modality']} & {vals['train']} & {vals['val']} & {vals['test']} \\\\"
            )
        blocks.append(
            "\\begin{table}[htbp]\n\\centering\n"
            "\\caption{Dataset distribution for experimental evaluation.}\n"
            "\\label{tab:dataset_splits}\n"
            "\\begin{tabular}{lcccc}\n\\toprule\n"
            "\\textbf{Dataset} & \\textbf{Modality} & \\textbf{Train} & \\textbf{Validation} & \\textbf{Test} \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        )

    f_t3 = out_path / "table3_zeroshot.json"
    if f_t3.exists():
        d3 = json.loads(f_t3.read_text(encoding="utf-8"))
        rows = []
        for m_name, ds_res in d3.items():
            kv_d, kv_i = ds_res["Kvasir-SEG"]["dsc"], ds_res["Kvasir-SEG"]["iou"]
            is_d, is_i = ds_res["ISIC"]["dsc"], ds_res["ISIC"]["iou"]
            if "Ours" in m_name:
                rows.append(
                    f"\\midrule\n\\textbf{{{m_name}}} & \\textbf{{{kv_d:.2f}}} & \\textbf{{{kv_i:.2f}}} "
                    f"& \\textbf{{{is_d:.2f}}} & \\textbf{{{is_i:.2f}}} \\\\"
                )
            else:
                rows.append(f"{m_name} & {kv_d:.2f} & {kv_i:.2f} & {is_d:.2f} & {is_i:.2f} \\\\")
        blocks.append(
            "\\begin{table}[htbp]\n\\centering\n"
            "\\caption{Zero-shot segmentation performance comparison on unseen medical datasets.}\n"
            "\\label{tab:zero_shot_comparison}\n"
            "\\begin{tabular}{lcccc}\n\\toprule\n"
            "\\multirow{2}{*}{\\textbf{Method}} & \\multicolumn{2}{c}{\\textbf{Kvasir-SEG \\cite{jha2020kvasir}}} "
            "& \\multicolumn{2}{c}{\\textbf{ISIC \\cite{codella2018skin}}} \\\\\n"
            "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\n"
            " & \\textbf{DSC (\\%)} & \\textbf{IoU (\\%)} & \\textbf{DSC (\\%)} & \\textbf{IoU (\\%)} \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        )

    f_t4 = out_path / "table4_fig10_ablation_aag.json"
    if f_t4.exists():
        d4 = json.loads(f_t4.read_text(encoding="utf-8"))
        db = d4["dense_baseline"]
        po = d4["points_only"]
        pb = d4["points_plus_box"]

        po_rows = []
        for idx, k in enumerate(["1", "3", "5", "10"]):
            lead = "\\multirow{4}{*}{Points Only}" if idx == 0 else ""
            po_rows.append(f"{lead} & {k} & \\texttimes & {po[k]['dsc']:.2f} & {po[k]['iou']:.2f} \\\\")

        best_k = max(pb.keys(), key=lambda kk: pb[kk]["dsc"])
        pb_rows = []
        for idx, k in enumerate(["1", "3", "5", "10"]):
            lead = "\\multirow{4}{*}{Points + Box}" if idx == 0 else ""
            if k == best_k:
                pb_rows.append(
                    f"{lead} & {k} & \\checkmark & \\textbf{{{pb[k]['dsc']:.2f}}} & \\textbf{{{pb[k]['iou']:.2f}}} \\\\"
                )
            else:
                pb_rows.append(f"{lead} & {k} & \\checkmark & {pb[k]['dsc']:.2f} & {pb[k]['iou']:.2f} \\\\")

        blocks.append(
            "\\begin{table}[htbp]\n\\centering\n"
            "\\caption{Ablation study on the Autonomous Anchor Generation (AAG) configuration across different point quantities ($K$) and bounding box ($B$) states.}\n"
            "\\label{tab:ablation_aag}\n"
            "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
            "\\textbf{Configuration} & \\textbf{Points ($K$)} & \\textbf{Box ($B$)} & \\textbf{DSC (\\%)} & \\textbf{IoU (\\%)} \\\\ \\midrule\n"
            f"Baseline (Dense Decoding) & - & - & {db['dsc']:.2f} & {db['iou']:.2f} \\\\ \\midrule\n"
            + "\n".join(po_rows)
            + " \\midrule\n"
            + "\n".join(pb_rows)
            + " \\bottomrule\n\\end{tabular}\n\\end{table}\n"
        )

    f_t5 = out_path / "table5_temperature.json"
    if f_t5.exists():
        d5 = json.loads(f_t5.read_text(encoding="utf-8"))
        t5_rows = []
        for key, label_tex in [
            ("Fixed_1.0", "Fixed & $1.0$"),
            ("Fixed_0.1", "Fixed & $0.1$"),
            ("Fixed_0.01", "Fixed & $0.01$"),
            ("Learnable_Dynamic_0.1", "\\midrule\n\\textbf{Learnable} & \\textbf{Dynamic (Init $0.1$)}"),
        ]:
            if key in d5:
                v = d5[key]
                if "Learnable" in key:
                    t5_rows.append(
                        f"{label_tex} & \\textbf{{{v['ale_px']:.2f}}} & \\textbf{{{v['dsc']:.2f}}} & \\textbf{{{v['iou']:.2f}}} \\\\"
                    )
                else:
                    t5_rows.append(f"{label_tex} & {v['ale_px']:.2f} & {v['dsc']:.2f} & {v['iou']:.2f} \\\\")
        blocks.append(
            "\\begin{table}[htbp]\n\\centering\n"
            "\\caption{Ablation study on the Spatial Softmax temperature ($\\tau$) configuration.}\n"
            "\\label{tab:temperature_ablation}\n"
            "\\begin{tabular}{llccc}\n\\toprule\n"
            "\\textbf{Configuration} & \\textbf{$\\tau$ Value} & \\textbf{ALE (px) $\\downarrow$} & \\textbf{DSC (\\%) $\\uparrow$} & \\textbf{IoU (\\%) $\\uparrow$} \\\\\n"
            "\\midrule\n" + "\n".join(t5_rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        )

    f_t7 = out_path / "table7_fig11_fusion.json"
    if f_t7.exists():
        d7 = json.loads(f_t7.read_text(encoding="utf-8"))
        t7_rows = []
        for s_name, m_dict in d7["metrics"].items():
            kv_d, kv_a = m_dict["Kvasir-SEG"]["dsc"], m_dict["Kvasir-SEG"]["ale_px"]
            is_d, is_a = m_dict["ISIC"]["dsc"], m_dict["ISIC"]["ale_px"]
            if "Ours" in s_name:
                t7_rows.append(
                    f"\\textbf{{{s_name}}} & \\textbf{{{kv_d:.2f}}} & \\textbf{{{kv_a:.2f}}} "
                    f"& \\textbf{{{is_d:.2f}}} & \\textbf{{{is_a:.2f}}} \\\\"
                )
            else:
                t7_rows.append(f"{s_name} & {kv_d:.2f} & {kv_a:.2f} & {is_d:.2f} & {is_a:.2f} \\\\")
        blocks.append(
            "\\begin{table}[htbp]\n\\centering\n"
            "\\caption{Quantitative comparison of cross-modal fusion strategies on Kvasir-SEG and ISIC datasets.}\n"
            "\\label{tab:fusion_strategies}\n"
            "\\begin{tabular}{lcccc}\n\\toprule\n"
            "\\multirow{2}{*}{\\textbf{Fusion Strategy}} & \\multicolumn{2}{c}{\\textbf{Kvasir-SEG}} & \\multicolumn{2}{c}{\\textbf{ISIC}} \\\\\n"
            "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\n"
            " & \\textbf{DSC (\\%)} $\\uparrow$ & \\textbf{ALE (px)} $\\downarrow$ & \\textbf{DSC (\\%)} $\\uparrow$ & \\textbf{ALE (px)} $\\downarrow$ \\\\\n"
            "\\midrule\n" + "\n".join(t7_rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        )

    f_t8 = out_path / "table8_fig7_fig12_efficiency_ale.json"
    if f_t8.exists():
        d8 = json.loads(f_t8.read_text(encoding="utf-8"))
        eff_rows = []
        for m_name, vals in d8.get("efficiency", {}).items():
            if "Ours" in m_name:
                eff_rows.append(
                    f"\\textbf{{{m_name}}} & \\textbf{{{vals['trainable_params_M']:.2f}}} & \\textbf{{{vals['inference_ms_per_frame']:.1f}}} \\\\"
                )
            else:
                eff_rows.append(
                    f"{m_name} & {vals['trainable_params_M']:.2f} & {vals['inference_ms_per_frame']:.1f} \\\\"
                )
        blocks.append(
            "\n\\begin{table}[htbp]\n\\centering\n"
            "\\caption{Comparison of computational efficiency across medical image segmentation models.}\n"
            "\\label{tab:efficiency}\n"
            "\\begin{tabular}{lcc}\n\\toprule\n"
            "\\textbf{Method} & \\textbf{Trainable Params (M)} & \\textbf{Inference Time (ms/frame)} \\\\\n"
            "\\midrule\n" + "\n".join(eff_rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        )

    tex_content = "\n".join(blocks) + "\n"
    tex_file = out_path / "real_experiment_tables_and_figures.tex"
    tex_file.write_text(tex_content, encoding="utf-8")
    return str(tex_file)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CMA-SAM Real Experimental Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=[
            "all",
            "cache_resized",
            "splits",
            "zeroshot",
            "fewshot",
            "ablation_aag",
            "ablation_tau",
            "prompt_complexity",
            "ablation_fusion",
            "efficiency_and_ale",
            "qualitative",
            "medgemma_prompts",
            "export_latex",
        ],
    )

    parser.add_argument("--kvasir_dir", type=str, default="./data/Kvasir-SEG")
    parser.add_argument("--isic_dir", type=str, default="./data/ISIC2018")
    parser.add_argument("--ham10000_dir", type=str, default="./data/HAM10000")
    parser.add_argument("--medgemma_kvasir_json", type=str, default="./data/prompts/kvasir_medgemma.json")
    parser.add_argument("--output_dir", type=str, default="./results")

    parser.add_argument("--sam_model_name", type=str, default="facebook/sam-vit-base")
    parser.add_argument("--sam_checkpoint", type=str, default=None)
    parser.add_argument("--sam_model_type", type=str, default="vit_b")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")

    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--num_points", type=int, default=5)
    parser.add_argument("--lambda_anchor", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--fewshot_epochs", type=int, default=120)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_resume", action="store_true", help="Disable auto-resuming completed checkpoints")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    is_ddp, rank, _, device = setup_distributed()

    base_cfg = CMASAMConfig(
        sam_model_name=args.sam_model_name,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        clip_model_name=args.clip_model_name,
        lora_rank=args.lora_rank,
        num_points=args.num_points,
        lambda_anchor=args.lambda_anchor,
    )

    try:
        if args.mode == "cache_resized":
            if is_main_process(rank):
                for d_path in [args.kvasir_dir, args.isic_dir, args.ham10000_dir]:
                    if d_path and os.path.exists(d_path):
                        cnt = precache_resized_dataset(d_path, image_size=1024, num_workers=16)
                        print(f"[CACHE] Pre-resized {cnt} image-mask pairs in {d_path}")
            return

        if args.mode == "medgemma_prompts":
            if is_main_process(rank):
                generate_medgemma_clinical_prompts(
                    image_dir=os.path.join(args.kvasir_dir, "images"),
                    output_json=args.medgemma_kvasir_json,
                    modality_hint="gastrointestinal endoscopy polyp",
                )
            return

        if args.mode == "export_latex":
            if is_main_process(rank):
                tex_out = export_latex_snippets(args.output_dir)
                print(f"Exported real LaTeX snippets to: {tex_out}")
            return

        if args.mode not in ("medgemma_prompts", "export_latex"):
            if is_main_process(rank):
                for d_path in [args.kvasir_dir, args.isic_dir, args.ham10000_dir]:
                    if d_path and os.path.exists(d_path):
                        precache_resized_dataset(d_path, image_size=1024, num_workers=16)
            ddp_barrier(is_ddp, device)

        if args.mode in ("all", "splits") and is_main_process(rank):
            run_exp_table2_splits(args)

        if args.mode in ("all", "zeroshot"):
            run_exp_table3_zeroshot(args, base_cfg, device, is_ddp, rank)

        if args.mode in ("all", "fewshot"):
            run_exp_fig8_fewshot(args, base_cfg, device, is_ddp, rank)

        if args.mode in ("all", "ablation_aag"):
            run_exp_table4_fig10_ablation_aag(args, base_cfg, device, is_ddp, rank)

        if args.mode in ("all", "ablation_tau"):
            run_exp_table5_temperature(args, base_cfg, device, is_ddp, rank)

        if args.mode in ("all", "prompt_complexity"):
            run_exp_table6_prompt_complexity(args, base_cfg, device, rank)

        if args.mode in ("all", "ablation_fusion"):
            run_exp_table7_fig11_fusion(args, base_cfg, device, is_ddp, rank)

        if args.mode in ("all", "efficiency_and_ale"):
            run_exp_table8_fig7_fig12_efficiency_and_ale(args, base_cfg, device, rank)

        if args.mode in ("all", "qualitative"):
            export_qualitative_figures(args, base_cfg, device, rank)

        if args.mode == "all" and is_main_process(rank):
            export_latex_snippets(args.output_dir)

    finally:
        cleanup_distributed(is_ddp)


if __name__ == "__main__":
    main()