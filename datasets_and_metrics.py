"""
Datasets, Fast Pre-Resize Caching, MedGemma Prompt Loader, and Evaluation Metrics for CMA-SAM.

Implements Section V-A Experimental Settings:
- Public datasets: Kvasir-SEG (800/100/100), ISIC 2018 Task 1 (2075/259/260), HAM10000
- Fast 16-worker 1024x1024 dataset pre-caching (`precache_resized_dataset`) to eliminate 4K/6K JPEG I/O bottlenecks
- Preprocessing: Resize to 1024x1024, SAM pre-training mean/std normalization
- Text prompts: Standardized simple class names and complex clinical descriptions
- Metrics: DSC (%), IoU (%), Point ALE (px & norm), Box ALE (1 - IoU_box), CUDA Latency, 1D Activation Profiles
"""

from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, Subset


SAM_PIXEL_MEAN = (123.675, 116.28, 103.53)
SAM_PIXEL_STD = (58.395, 57.12, 57.375)

DEFAULT_SIMPLE_PROMPTS: Dict[str, str] = {
    "kvasir": "polyp",
    "isic2018": "melanoma",
    "ham10000": "pigmented skin lesion",
}

FALLBACK_CLINICAL_TEMPLATES: Dict[str, str] = {
    "kvasir": (
        "an endoscopic gastrointestinal mucosal polypoid lesion with irregular "
        "surface vascularity and indistinct peripheral borders"
    ),
    "isic2018": (
        "a dermoscopic cutaneous melanocytic lesion exhibiting asymmetric borders, "
        "heterogeneous pigment network, and irregular peripheral streaks"
    ),
    "ham10000": (
        "a dermatoscopic pigmented skin lesion presenting irregular morphological "
        "margins and heterogeneous chromatic distribution against surrounding epidermis"
    ),
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _locate_raw_dirs(root: Path) -> Tuple[Path, Path]:
    if (root / "images_1024").exists() and (root / "masks_1024").exists():
        return root / "images_1024", root / "masks_1024"
    if (root / "images").exists() and (root / "masks").exists():
        return root / "images", root / "masks"
    if (root / "ISIC2018_Task1-2_Training_Input").exists():
        return (
            root / "ISIC2018_Task1-2_Training_Input",
            root / "ISIC2018_Task1_Training_GroundTruth",
        )
    if (root / "HAM10000_images").exists():
        return root / "HAM10000_images", root / "HAM10000_segmentations_lesion_tschandl"
    raise FileNotFoundError(
        f"Could not locate `images/` & `masks/` subdirectories inside {root}."
    )


def _match_image_mask_pairs(
    image_dir: Path,
    mask_dir: Path,
) -> List[Tuple[Path, Path, str]]:
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"Dataset directories not found: images={image_dir}, masks={mask_dir}"
        )

    mask_lookup: Dict[str, Path] = {}
    for m_path in sorted(mask_dir.iterdir()):
        if m_path.is_file() and m_path.suffix.lower() in IMAGE_EXTENSIONS:
            stem = m_path.stem
            mask_lookup[stem] = m_path
            if stem.endswith("_segmentation"):
                mask_lookup[stem[: -len("_segmentation")]] = m_path
            if stem.endswith("_mask"):
                mask_lookup[stem[: -len("_mask")]] = m_path

    pairs: List[Tuple[Path, Path, str]] = []
    for img_path in sorted(image_dir.iterdir()):
        if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
            stem = img_path.stem
            if stem in mask_lookup:
                pairs.append((img_path, mask_lookup[stem], stem))

    if not pairs:
        raise RuntimeError(
            f"No matching image-mask pairs found between {image_dir} and {mask_dir}."
        )
    return pairs


def _resize_one_pair(task: Tuple[str, str, str, str, int]) -> None:
    img_src, mask_src, img_dst, mask_dst, image_size = task
    if os.path.exists(img_dst) and os.path.exists(mask_dst):
        return
    with Image.open(img_src) as im:
        im_rgb = im.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        im_rgb.save(img_dst, format="JPEG", quality=95)
    with Image.open(mask_src) as mk:
        mk_l = mk.convert("L").resize((image_size, image_size), Image.NEAREST)
        mk_l.save(mask_dst, format="PNG")


def precache_resized_dataset(
    data_root: str,
    image_size: int = 1024,
    num_workers: int = 16,
) -> int:
    """
    Pre-resizes raw high-resolution images (such as 4K/6K ISIC 2018 JPEGs) once into
    `<data_root>/images_1024` and `<data_root>/masks_1024` in parallel across CPU cores.
    `build_dataset_splits` automatically prefers `images_1024` when present.
    """
    root = Path(data_root)
    if not root.exists():
        return 0
    if (root / "images_1024").exists() and (root / "masks_1024").exists():
        existing = list((root / "images_1024").glob("*.jpg"))
        if len(existing) > 0:
            return len(existing)

    # Locate original uncached folders
    if (root / "ISIC2018_Task1-2_Training_Input").exists():
        raw_img_dir = root / "ISIC2018_Task1-2_Training_Input"
        raw_mask_dir = root / "ISIC2018_Task1_Training_GroundTruth"
    elif (root / "images").exists() and (root / "masks").exists():
        raw_img_dir, raw_mask_dir = root / "images", root / "masks"
    else:
        return 0

    pairs = _match_image_mask_pairs(raw_img_dir, raw_mask_dir)
    dst_img_dir = root / "images_1024"
    dst_mask_dir = root / "masks_1024"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_mask_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (
            str(img_p),
            str(msk_p),
            str(dst_img_dir / f"{stem}.jpg"),
            str(dst_mask_dir / f"{stem}.png"),
            image_size,
        )
        for img_p, msk_p, stem in pairs
    ]
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        list(pool.map(_resize_one_pair, tasks, chunksize=16))
    return len(tasks)


def create_deterministic_splits(
    pairs: Sequence[Tuple[Path, Path, str]],
    dataset_name: str,
    seed: int = 42,
) -> Dict[str, List[Tuple[Path, Path, str]]]:
    items = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(items)
    n_total = len(items)
    ds_key = dataset_name.lower()

    if ds_key == "kvasir" and n_total == 1000:
        n_train, n_val = 800, 100
    elif ds_key in ("isic", "isic2018") and n_total == 2594:
        n_train, n_val = 2075, 259
    else:
        n_train = int(round(0.8 * n_total))
        n_val = int(round(0.1 * n_total))

    return {
        "train": items[:n_train],
        "val": items[n_train : n_train + n_val],
        "test": items[n_train + n_val :],
    }


class MedicalSegmentationDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[Path, Path, str]],
        dataset_name: str = "kvasir",
        image_size: int = 1024,
        prompt_mode: str = "simple",
        medgemma_prompts_json: Optional[str] = None,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.samples = list(samples)
        self.dataset_name = dataset_name.lower()
        self.image_size = image_size
        self.prompt_mode = prompt_mode.lower()
        self.augment = augment

        self.medgemma_map: Dict[str, str] = {}
        if medgemma_prompts_json and os.path.exists(medgemma_prompts_json):
            with open(medgemma_prompts_json, "r", encoding="utf-8") as f:
                self.medgemma_map = json.load(f)

        self.mean = np.array(SAM_PIXEL_MEAN, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(SAM_PIXEL_STD, dtype=np.float32).reshape(3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def _get_prompt(self, sample_id: str) -> str:
        if self.prompt_mode == "simple":
            return DEFAULT_SIMPLE_PROMPTS.get(self.dataset_name, "lesion")
        if sample_id in self.medgemma_map:
            return self.medgemma_map[sample_id]
        return FALLBACK_CLINICAL_TEMPLATES.get(
            self.dataset_name,
            "a pathological medical lesion with heterogeneous tissue boundaries",
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, mask_path, sample_id = self.samples[idx]

        img_pil = Image.open(img_path).convert("RGB")
        mask_pil = Image.open(mask_path).convert("L")

        if img_pil.size != (self.image_size, self.image_size):
            img_pil = img_pil.resize((self.image_size, self.image_size), Image.BILINEAR)
        if mask_pil.size != (self.image_size, self.image_size):
            mask_pil = mask_pil.resize((self.image_size, self.image_size), Image.NEAREST)

        img_np = np.asarray(img_pil, dtype=np.float32)
        mask_np = (np.asarray(mask_pil, dtype=np.float32) >= 127.5).astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                img_np = np.ascontiguousarray(img_np[:, ::-1, :])
                mask_np = np.ascontiguousarray(mask_np[:, ::-1])
            if random.random() < 0.5:
                img_np = np.ascontiguousarray(img_np[::-1, :, :])
                mask_np = np.ascontiguousarray(mask_np[::-1, :])

        img_chw = img_np.transpose(2, 0, 1)
        img_norm = (img_chw - self.mean) / self.std

        return {
            "image": torch.from_numpy(img_norm),
            "raw_image": torch.from_numpy(img_np.astype(np.uint8)),
            "mask": torch.from_numpy(mask_np).unsqueeze(0),
            "prompt": self._get_prompt(sample_id),
            "sample_id": sample_id,
        }


def build_dataset_splits(
    data_root: str,
    dataset_name: str,
    image_size: int = 1024,
    prompt_mode: str = "simple",
    medgemma_prompts_json: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, MedicalSegmentationDataset]:
    root = Path(data_root)
    img_dir, mask_dir = _locate_raw_dirs(root)
    pairs = _match_image_mask_pairs(img_dir, mask_dir)
    split_pairs = create_deterministic_splits(pairs, dataset_name=dataset_name, seed=seed)

    return {
        split: MedicalSegmentationDataset(
            samples=items,
            dataset_name=dataset_name,
            image_size=image_size,
            prompt_mode=prompt_mode,
            medgemma_prompts_json=medgemma_prompts_json,
            augment=(split == "train"),
        )
        for split, items in split_pairs.items()
    }


def create_few_shot_subset(
    train_dataset: Dataset,
    n_shots: int,
    seed: int = 42,
) -> Subset:
    total = len(train_dataset)
    if n_shots > total:
        raise ValueError(f"Requested {n_shots} shots, but dataset only has {total} samples.")
    rng = random.Random(seed)
    indices = rng.sample(range(total), k=n_shots)
    return Subset(train_dataset, indices)


# =============================================================================
# Section V-A Evaluation Metrics
# =============================================================================

@torch.no_grad()
def compute_segmentation_metrics(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    eps: float = 1e-6,
) -> Dict[str, float]:
    pred_b = (pred_masks > 0.5).float().view(pred_masks.shape[0], -1)
    gt_b = (gt_masks > 0.5).float().view(gt_masks.shape[0], -1)

    inter = (pred_b * gt_b).sum(dim=-1)
    pred_sum = pred_b.sum(dim=-1)
    gt_sum = gt_b.sum(dim=-1)
    union = pred_sum + gt_sum - inter

    dsc = ((2.0 * inter + eps) / (pred_sum + gt_sum + eps)) * 100.0
    iou = ((inter + eps) / (union + eps)) * 100.0

    return {
        "dsc_percent": float(dsc.mean().item()),
        "iou_percent": float(iou.mean().item()),
    }


@torch.no_grad()
def compute_anchor_localization_error(
    pred_points_px: torch.Tensor,
    pred_boxes_px: torch.Tensor,
    gt_masks: torch.Tensor,
    feature_grid_scale: float = 64.0 / 1024.0,
) -> Dict[str, float]:
    bsz, _, _, width = gt_masks.shape
    gt_bin = (gt_masks[:, 0] > 0.5)

    ale_img_list: List[float] = []
    ale_grid_list: List[float] = []
    ale_norm_list: List[float] = []
    box_ale_list: List[float] = []

    for b in range(bsz):
        fg_yx = torch.nonzero(gt_bin[b], as_tuple=False).float()
        if fg_yx.numel() == 0:
            continue

        gt_cy = fg_yx[:, 0].mean()
        gt_cx = fg_yx[:, 1].mean()
        gt_centroid = torch.stack([gt_cx, gt_cy])

        pred_center = pred_points_px[b].mean(dim=0)
        dist_img_px = torch.norm(pred_center - gt_centroid, p=2).item()
        ale_img_list.append(dist_img_px)
        ale_grid_list.append(dist_img_px * feature_grid_scale)
        ale_norm_list.append(dist_img_px / float(width))

        gt_ymin, gt_ymax = fg_yx[:, 0].min(), fg_yx[:, 0].max() + 1.0
        gt_xmin, gt_xmax = fg_yx[:, 1].min(), fg_yx[:, 1].max() + 1.0

        px_min, py_min, px_max, py_max = pred_boxes_px[b].unbind(dim=-1)
        inter_xmin = torch.maximum(px_min, gt_xmin)
        inter_ymin = torch.maximum(py_min, gt_ymin)
        inter_xmax = torch.minimum(px_max, gt_xmax)
        inter_ymax = torch.minimum(py_max, gt_ymax)

        inter_area = (inter_xmax - inter_xmin).clamp(min=0.0) * (inter_ymax - inter_ymin).clamp(min=0.0)
        pred_area = (px_max - px_min).clamp(min=0.0) * (py_max - py_min).clamp(min=0.0)
        gt_area = (gt_xmax - gt_xmin).clamp(min=0.0) * (gt_ymax - gt_ymin).clamp(min=0.0)
        union_area = pred_area + gt_area - inter_area + 1e-6

        iou_box = (inter_area / union_area).item()
        box_ale_list.append(1.0 - iou_box)

    if not ale_img_list:
        return {"ale_px": 0.0, "ale_img_px": 0.0, "ale_norm": 0.0, "box_ale": 0.0}

    return {
        "ale_px": float(np.mean(ale_grid_list)),
        "ale_img_px": float(np.mean(ale_img_list)),
        "ale_norm": float(np.mean(ale_norm_list)),
        "box_ale": float(np.mean(box_ale_list)),
    }


@torch.no_grad()
def extract_1d_activation_profile(
    fused_features: torch.Tensor,
    gt_mask: torch.Tensor,
    num_bins: int = 101,
) -> np.ndarray:
    act_2d = torch.norm(fused_features[0].float(), p=2, dim=0)
    h_f, w_f = act_2d.shape

    gt_2d = (gt_mask[0, 0] > 0.5).float()
    fg_y = torch.nonzero(gt_2d, as_tuple=False)[:, 0]
    if fg_y.numel() > 0:
        cy_ratio = fg_y.float().mean().item() / float(gt_2d.shape[0])
        row_idx = min(max(int(round(cy_ratio * h_f)), 0), h_f - 1)
    else:
        row_idx = h_f // 2

    line = act_2d[row_idx].cpu().numpy()
    x_src = np.linspace(0.0, 100.0, num=w_f)
    x_dst = np.linspace(0.0, 100.0, num=num_bins)
    profile = np.interp(x_dst, x_src, line)

    p_min, p_max = profile.min(), profile.max()
    if p_max - p_min > 1e-8:
        profile = (profile - p_min) / (p_max - p_min)
    return profile


@torch.no_grad()
def benchmark_inference_latency(
    model: torch.nn.Module,
    device: torch.device,
    image_size: int = 1024,
    prompt: str = "polyp",
    warmup_iters: int = 15,
    measure_iters: int = 50,
) -> Dict[str, float]:
    model.eval()
    dummy_img = torch.randn(1, 3, image_size, image_size, device=device)
    prompts = [prompt]

    for _ in range(warmup_iters):
        _ = model(dummy_img, prompts)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    timings_ms: List[float] = []
    for _ in range(measure_iters):
        if device.type == "cuda":
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            _ = model(dummy_img, prompts)
            end_evt.record()
            torch.cuda.synchronize(device)
            timings_ms.append(float(start_evt.elapsed_time(end_evt)))
        else:
            t0 = time.perf_counter()
            _ = model(dummy_img, prompts)
            t1 = time.perf_counter()
            timings_ms.append((t1 - t0) * 1000.0)

    return {
        "mean_ms": float(np.mean(timings_ms)),
        "std_ms": float(np.std(timings_ms)),
        "median_ms": float(np.median(timings_ms)),
    }


def generate_medgemma_clinical_prompts(
    image_dir: str,
    output_json: str,
    modality_hint: str = "gastrointestinal endoscopy polyp",
    medgemma_model_id: str = "google/medgemma-4b-it",
    device: str = "cuda",
    max_new_tokens: int = 64,
) -> None:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(medgemma_model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        medgemma_model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
    )
    model.eval()

    img_folder = Path(image_dir)
    results: Dict[str, str] = {}
    instruction = (
        f"Provide a concise, single-sentence clinical morphological description of the target "
        f"{modality_hint} in this medical image (focusing on shape, borders, and surface texture) "
        f"suitable as a segmentation text prompt."
    )

    for img_path in sorted(img_folder.iterdir()):
        if not (img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS):
            continue
        pil_img = Image.open(img_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        caption = processor.decode(gen_ids[0][input_len:], skip_special_tokens=True).strip()
        results[img_path.stem] = caption

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
