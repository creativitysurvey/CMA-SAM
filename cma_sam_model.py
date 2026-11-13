"""
CMA-SAM: Cross-Modal Anchor Generation for Text-Driven Medical Image Segmentation
Core Model Architecture, LoRA Adaptation, Fusion Strategies, AAG Module, and Losses.

Implements Section III of the paper:
- Section III-B.1: Visual Encoding with Parameter-Efficient Fine-Tuning (LoRA on SAM ViT Q/V projections)
- Section III-B.2: Implicit Semantic Text Encoding (Frozen CLIP / BiomedCLIP Text Encoder)
- Section III-C: Joint Feature Fusion (Bidirectional Cross-Attention + Concatenation & Hadamard ablations)
- Section III-D: Autonomous Anchor Generation (AAG) with Bounded Spatial Softmax, Dynamic Temperature, and Box MLP
- Section III-E: Differentiable Integration with SAM Native Prompt Encoder, Mask Decoder, and Joint Mask + Mask-Derived Anchor Loss
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Configuration Dataclass
# =============================================================================

@dataclass
class CMASAMConfig:
    """Configuration matching Section III & Section V-A hyperparameters."""

    # Foundation backbones
    sam_model_name: str = "facebook/sam-vit-base"
    sam_checkpoint: Optional[str] = None
    sam_model_type: str = "vit_b"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    image_size: int = 1024

    # LoRA configuration (Section III-B.1 & Section V-A: rank r=16 on Q and V)
    lora_rank: int = 16
    lora_alpha: float = 16.0

    # Feature dimensions
    visual_channels: int = 256
    fusion_dim: int = 256
    fusion_heads: int = 8
    fusion_ffn_dim: int = 1024
    fusion_layers: int = 2
    fusion_dropout: float = 0.1
    fusion_strategy: str = "bidirectional"  # {"bidirectional", "concat", "hadamard"}

    # Autonomous Anchor Generation (AAG) configuration (Section III-D & Ablations)
    num_points: int = 5         # K in {1, 3, 5, 10}; default K=5
    use_points: bool = True
    use_box: bool = True
    dense_decoding_only: bool = False
    saliency_heuristic_only: bool = False
    init_tau: float = 0.1
    learnable_tau: bool = True
    tau_eps: float = 1e-4
    logit_scale_cap: float = 3.0  # Bounded activation S in [-3, +3] per Proposition IV.2 & Theorem IV.3
    box_mlp_hidden: int = 256
    box_param_mode: str = "cxcywh"

    # Loss weights (Section III-E.2: Joint Mask + Mask-Derived Geometric Anchor Supervision)
    lambda_dice: float = 1.0
    lambda_bce: float = 1.0
    lambda_anchor: float = 2.0
    dice_smooth: float = 1e-5


# =============================================================================
# Section III-B.1: LoRA Injection for SAM ViT Query & Value Projections
# =============================================================================

class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for standard nn.Linear: h = X W_0 + (alpha/r) X A B."""

    def __init__(self, base_layer: nn.Linear, rank: int = 16, alpha: float = 16.0) -> None:
        super().__init__()
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.scaling = alpha / float(rank)

        self.lora_A = nn.Parameter(torch.empty(self.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features))
        nn.init.normal_(self.lora_A, mean=0.0, std=1.0 / math.sqrt(self.in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = (x @ self.lora_A) @ self.lora_B
        return base_out + self.scaling * lora_out


class LoRAFusedQKV(nn.Module):
    """
    LoRA wrapper for SAM's fused QKV linear projection (dim -> 3 * dim).
    Injects trainable low-rank matrices A, B exclusively into Query [:d] and Value [2d:]
    while leaving Key [d:2d] frozen, strictly matching Section III-B.1 & SAMed [11].
    """

    def __init__(self, qkv_layer: nn.Linear, rank: int = 16, alpha: float = 16.0) -> None:
        super().__init__()
        if qkv_layer.out_features != 3 * qkv_layer.in_features:
            raise ValueError(
                f"Expected fused QKV linear with out_features == 3 * in_features, "
                f"got {qkv_layer.in_features} -> {qkv_layer.out_features}"
            )
        self.base_qkv = qkv_layer
        for param in self.base_qkv.parameters():
            param.requires_grad = False

        self.dim = qkv_layer.in_features
        self.rank = rank
        self.scaling = alpha / float(rank)

        self.lora_A_q = nn.Parameter(torch.empty(self.dim, rank))
        self.lora_B_q = nn.Parameter(torch.zeros(rank, self.dim))
        self.lora_A_v = nn.Parameter(torch.empty(self.dim, rank))
        self.lora_B_v = nn.Parameter(torch.zeros(rank, self.dim))

        nn.init.normal_(self.lora_A_q, mean=0.0, std=1.0 / math.sqrt(self.dim))
        nn.init.normal_(self.lora_A_v, mean=0.0, std=1.0 / math.sqrt(self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv_out = self.base_qkv(x)
        delta_q = ((x @ self.lora_A_q) @ self.lora_B_q) * self.scaling
        delta_v = ((x @ self.lora_A_v) @ self.lora_B_v) * self.scaling
        delta_k = torch.zeros_like(delta_q)
        delta_qkv = torch.cat([delta_q, delta_k, delta_v], dim=-1)
        return qkv_out + delta_qkv


def inject_lora_into_sam_vision_encoder(
    vision_encoder: nn.Module,
    rank: int = 16,
    alpha: float = 16.0,
) -> int:
    """Freezes SAM vision encoder and injects LoRA into every MHSA Q/V layer safely."""
    for param in vision_encoder.parameters():
        param.requires_grad = False

    adapted_blocks = 0
    modules_snapshot = list(vision_encoder.named_modules())
    for _, module in modules_snapshot:
        if isinstance(module, (LoRAFusedQKV, LoRALinear)):
            continue
        if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
            if module.qkv.out_features == 3 * module.qkv.in_features:
                module.qkv = LoRAFusedQKV(module.qkv, rank=rank, alpha=alpha)
                adapted_blocks += 1
        elif hasattr(module, "q_proj") and hasattr(module, "v_proj"):
            if isinstance(module.q_proj, nn.Linear) and isinstance(module.v_proj, nn.Linear):
                module.q_proj = LoRALinear(module.q_proj, rank=rank, alpha=alpha)
                module.v_proj = LoRALinear(module.v_proj, rank=rank, alpha=alpha)
                adapted_blocks += 1

    if adapted_blocks == 0:
        raise RuntimeError("No QKV or Q/V attention projections found in vision_encoder.")
    return adapted_blocks


# =============================================================================
# Section III-C & Section V-F: Joint Feature Fusion Modules
# =============================================================================

class BidirectionalCrossAttentionLayer(nn.Module):
    """Single Bidirectional Cross-Attention layer implementing Equations (III.4)-(III.9)."""

    def __init__(
        self,
        channels: int = 256,
        hidden_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.w_qv = nn.Linear(channels, hidden_dim)
        self.w_kt = nn.Linear(channels, hidden_dim)
        self.w_vt = nn.Linear(channels, hidden_dim)
        self.w_ov = nn.Linear(hidden_dim, channels)

        self.w_qt = nn.Linear(channels, hidden_dim)
        self.w_kv = nn.Linear(channels, hidden_dim)
        self.w_vv = nn.Linear(channels, hidden_dim)
        self.w_ot = nn.Linear(hidden_dim, channels)

        self.ffn_v = nn.Sequential(
            nn.Linear(channels, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, channels),
            nn.Dropout(dropout),
        )
        self.ffn_t = nn.Sequential(
            nn.Linear(channels, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, channels),
            nn.Dropout(dropout),
        )
        self.norm_v1 = nn.LayerNorm(channels)
        self.norm_v2 = nn.LayerNorm(channels)
        self.norm_t1 = nn.LayerNorm(channels)
        self.norm_t2 = nn.LayerNorm(channels)
        self.attn_drop = nn.Dropout(dropout)

        self.last_t2v_affinity: Optional[torch.Tensor] = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_dim)

    def forward(
        self,
        x_v: torch.Tensor,
        x_t: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_t = self._split_heads(self.w_qt(x_t))
        k_v = self._split_heads(self.w_kv(x_v))
        v_v = self._split_heads(self.w_vv(x_v))

        scores_v2t = torch.matmul(q_t, k_v.transpose(-2, -1)) * self.scale
        attn_v2t = self.attn_drop(F.softmax(scores_v2t, dim=-1))
        ctx_v2t = self._merge_heads(torch.matmul(attn_v2t, v_v))

        x_t_mid = self.norm_t1(x_t + self.w_ot(ctx_v2t))
        x_t_tilde = self.norm_t2(x_t_mid + self.ffn_t(x_t_mid))

        q_v = self._split_heads(self.w_qv(x_v))
        k_t = self._split_heads(self.w_kt(x_t_tilde))
        v_t = self._split_heads(self.w_vt(x_t_tilde))

        scores_t2v = torch.matmul(q_v, k_t.transpose(-2, -1)) * self.scale
        if text_mask is not None:
            mask = (text_mask == 0).unsqueeze(1).unsqueeze(2)
            scores_t2v = scores_t2v.masked_fill(mask, float("-inf"))

        attn_t2v = F.softmax(scores_t2v, dim=-1)
        self.last_t2v_affinity = attn_t2v.detach()
        attn_t2v = self.attn_drop(attn_t2v)
        ctx_t2v = self._merge_heads(torch.matmul(attn_t2v, v_t))

        x_v_mid = self.norm_v1(x_v + self.w_ov(ctx_t2v))
        x_v_tilde = self.norm_v2(x_v_mid + self.ffn_v(x_v_mid))

        return x_v_tilde, x_t_tilde


class JointFeatureFusion(nn.Module):
    """Unified Joint Feature Fusion ('bidirectional', 'concat', 'hadamard')."""

    def __init__(
        self,
        visual_channels: int = 256,
        text_dim: int = 512,
        strategy: str = "bidirectional",
        hidden_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.visual_channels = visual_channels
        self.text_dim = text_dim
        self.strategy = strategy.lower()
        self.text_proj = nn.Linear(text_dim, visual_channels)

        if self.strategy == "bidirectional":
            self.layers = nn.ModuleList(
                [
                    BidirectionalCrossAttentionLayer(
                        channels=visual_channels,
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        ffn_dim=ffn_dim,
                        dropout=dropout,
                    )
                    for _ in range(num_layers)
                ]
            )
        elif self.strategy == "concat":
            self.concat_proj = nn.Sequential(
                nn.Conv2d(visual_channels * 2, visual_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(visual_channels),
                nn.ReLU(inplace=True),
            )
        elif self.strategy == "hadamard":
            self.hadamard_proj = nn.Sequential(
                nn.Conv2d(visual_channels, visual_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(visual_channels),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"Unsupported fusion strategy: {strategy}")

    def forward(
        self,
        f_v: torch.Tensor,
        f_t_seq: torch.Tensor,
        f_t_pooled: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, channels, h_f, w_f = f_v.shape

        if self.strategy == "bidirectional":
            x_v = f_v.flatten(2).transpose(1, 2)
            x_t = self.text_proj(f_t_seq)
            for layer in self.layers:
                x_v, x_t = layer(x_v, x_t, text_mask=text_mask)
            m = x_v.transpose(1, 2).contiguous().view(bsz, channels, h_f, w_f)
            return m

        t_global = self.text_proj(f_t_pooled).unsqueeze(-1).unsqueeze(-1)
        t_spatial = t_global.expand(-1, -1, h_f, w_f)

        if self.strategy == "concat":
            return self.concat_proj(torch.cat([f_v, t_spatial], dim=1))

        return self.hadamard_proj(f_v * t_spatial)


# =============================================================================
# Section III-D & Algorithm 1: Autonomous Anchor Generation (AAG) Module
# =============================================================================

class AutonomousAnchorGeneration(nn.Module):
    """
    Autonomous Anchor Generation (AAG) module (Algorithm 1, Equations III.10-III.15).
    Applies bounded spatial activations S = c * tanh(f_conv(M)) per Proposition IV.2 & Theorem IV.3
    so temperature tau acts as the true physical sharpness parameter of Spatial Softmax.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_points: int = 5,
        image_size: int = 1024,
        init_tau: float = 0.1,
        learnable_tau: bool = True,
        tau_eps: float = 1e-4,
        logit_scale_cap: float = 3.0,
        box_mlp_hidden: int = 256,
        box_param_mode: str = "cxcywh",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_points = num_points
        self.image_size = image_size
        self.learnable_tau = learnable_tau
        self.tau_eps = tau_eps
        self.logit_scale_cap = logit_scale_cap
        self.box_param_mode = box_param_mode

        # Eq. (III.10): f_conv = 3x3 Conv -> BN -> ReLU -> 1x1 Conv (C -> K)
        self.point_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_points, kernel_size=1, bias=True),
        )

        # Eq. (III.13): Dynamic temperature tau = Softplus(w_tau) + eps, initialized at init_tau
        effective_init = max(init_tau - tau_eps, 1e-4)
        init_w_tau = math.log(math.expm1(effective_init))
        if learnable_tau:
            self.w_tau = nn.Parameter(torch.tensor(init_w_tau, dtype=torch.float32))
        else:
            self.register_buffer("fixed_tau", torch.tensor(init_tau, dtype=torch.float32))

        # Eq. (III.14)-(III.15): GAP + LayerNorm + two-layer MLP + Sigmoid
        self.box_norm = nn.LayerNorm(in_channels)
        self.box_mlp = nn.Sequential(
            nn.Linear(in_channels, box_mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(box_mlp_hidden, 4),
        )

    def get_temperature(self) -> torch.Tensor:
        if self.learnable_tau:
            return F.softplus(self.w_tau.float()) + self.tau_eps
        return self.fixed_tau.float()

    def spatial_softmax_expectation(
        self,
        activation_maps: torch.Tensor,
        tau: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        act_fp32 = activation_maps.float()
        tau_fp32 = tau.float()
        bsz, k_pts, h_f, w_f = act_fp32.shape

        scaled_logits = (act_fp32 / tau_fp32).view(bsz, k_pts, h_f * w_f)
        prob_flat = F.softmax(scaled_logits, dim=-1)
        prob_maps = prob_flat.view(bsz, k_pts, h_f, w_f)

        device = act_fp32.device
        grid_x = (torch.arange(w_f, device=device, dtype=torch.float32) + 0.5) / float(w_f)
        grid_y = (torch.arange(h_f, device=device, dtype=torch.float32) + 0.5) / float(h_f)

        exp_x = (prob_maps.sum(dim=2) * grid_x.view(1, 1, w_f)).sum(dim=-1)
        exp_y = (prob_maps.sum(dim=3) * grid_y.view(1, 1, h_f)).sum(dim=-1)

        points_norm = torch.stack([exp_x, exp_y], dim=-1)
        return points_norm, prob_maps

    def regress_bounding_box(self, m: torch.Tensor) -> torch.Tensor:
        z = m.mean(dim=(-2, -1))
        z = self.box_norm(z)
        raw = torch.sigmoid(self.box_mlp(z).float())

        if self.box_param_mode == "cxcywh":
            cx, cy, w, h = raw.unbind(dim=-1)
            x_min = (cx - 0.5 * w).clamp(0.0, 1.0)
            y_min = (cy - 0.5 * h).clamp(0.0, 1.0)
            x_max = (cx + 0.5 * w).clamp(0.0, 1.0)
            y_max = (cy + 0.5 * h).clamp(0.0, 1.0)
            x_max = torch.maximum(x_max, (x_min + 1e-3).clamp(max=1.0))
            y_max = torch.maximum(y_max, (y_min + 1e-3).clamp(max=1.0))
            return torch.stack([x_min, y_min, x_max, y_max], dim=-1)

        x1, y1, x2, y2 = raw.unbind(dim=-1)
        x_min = torch.minimum(x1, x2)
        y_min = torch.minimum(y1, y2)
        x_max = torch.maximum(x1, x2)
        y_max = torch.maximum(y1, y2)
        return torch.stack([x_min, y_min, x_max, y_max], dim=-1)

    def forward(self, m: torch.Tensor) -> Dict[str, torch.Tensor]:
        tau = self.get_temperature()
        raw_maps = self.point_conv(m).float()
        # Bounded activation map S in [-logit_scale_cap, +logit_scale_cap] (Proposition IV.2)
        activation_maps = self.logit_scale_cap * torch.tanh(raw_maps)
        points_norm, prob_maps = self.spatial_softmax_expectation(activation_maps, tau)
        boxes_norm = self.regress_bounding_box(m)

        points_px = points_norm * float(self.image_size)
        boxes_px = boxes_norm * float(self.image_size)

        return {
            "points": points_px,
            "points_norm": points_norm,
            "boxes": boxes_px,
            "boxes_norm": boxes_norm,
            "activation_maps": activation_maps,
            "prob_maps": prob_maps,
            "tau": tau,
        }


# =============================================================================
# Baselines Built into the Unified Pipeline (Dense Decoding & Saliency Heuristic)
# =============================================================================

class DenseMaskPromptHead(nn.Module):
    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self.proj(m)


def extract_saliency_heuristic_anchors(
    visual_features: torch.Tensor,
    num_points: int = 5,
    image_size: int = 1024,
    threshold_quantile: float = 0.85,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, _, h_f, w_f = visual_features.shape
    device = visual_features.device
    saliency = torch.norm(visual_features.detach().float(), p=2, dim=1)

    flat_sal = saliency.view(bsz, -1)
    _, topk_idx = torch.topk(flat_sal, k=min(num_points, h_f * w_f), dim=-1)
    y_idx = (topk_idx // w_f).float()
    x_idx = (topk_idx % w_f).float()

    pts_x = ((x_idx + 0.5) / float(w_f)) * float(image_size)
    pts_y = ((y_idx + 0.5) / float(h_f)) * float(image_size)
    points_px = torch.stack([pts_x, pts_y], dim=-1)

    boxes_list: List[torch.Tensor] = []
    for b in range(bsz):
        sal_b = saliency[b]
        q_val = torch.quantile(sal_b.flatten(), threshold_quantile)
        fg_coords = torch.nonzero(sal_b >= q_val, as_tuple=False)
        if fg_coords.numel() == 0:
            box = torch.tensor([0.25, 0.25, 0.75, 0.75], device=device) * float(image_size)
        else:
            y_min = (fg_coords[:, 0].min().float() / float(h_f)) * float(image_size)
            y_max = ((fg_coords[:, 0].max().float() + 1.0) / float(h_f)) * float(image_size)
            x_min = (fg_coords[:, 1].min().float() / float(w_f)) * float(image_size)
            x_max = ((fg_coords[:, 1].max().float() + 1.0) / float(w_f)) * float(image_size)
            box = torch.stack([x_min, y_min, x_max, y_max])
        boxes_list.append(box)

    boxes_px = torch.stack(boxes_list, dim=0)
    return points_px, boxes_px


# =============================================================================
# Section III-E: End-to-End CMA-SAM Framework
# =============================================================================

class CMASAM(nn.Module):
    """End-to-End Cross-Modal Anchor Segment Anything Model (Algorithm 2)."""

    def __init__(self, config: CMASAMConfig) -> None:
        super().__init__()
        self.config = config
        self.backend = "segment_anything" if config.sam_checkpoint else "transformers"

        if self.backend == "transformers":
            from transformers import SamModel

            self.sam = SamModel.from_pretrained(config.sam_model_name)
            for param in self.sam.parameters():
                param.requires_grad = False
            self.num_lora_blocks = inject_lora_into_sam_vision_encoder(
                self.sam.vision_encoder,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
            )
            self.sam.prompt_encoder.eval()
            self.sam.mask_decoder.eval()
        else:
            from segment_anything import sam_model_registry

            self.sam = sam_model_registry[config.sam_model_type](checkpoint=config.sam_checkpoint)
            for param in self.sam.parameters():
                param.requires_grad = False
            self.num_lora_blocks = inject_lora_into_sam_vision_encoder(
                self.sam.image_encoder,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
            )
            self.sam.prompt_encoder.eval()
            self.sam.mask_decoder.eval()

        from transformers import AutoTokenizer, CLIPTextModel

        self.tokenizer = AutoTokenizer.from_pretrained(config.clip_model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(config.clip_model_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.eval()

        text_dim = self.text_encoder.config.hidden_size

        self.fusion = JointFeatureFusion(
            visual_channels=config.visual_channels,
            text_dim=text_dim,
            strategy=config.fusion_strategy,
            hidden_dim=config.fusion_dim,
            num_heads=config.fusion_heads,
            ffn_dim=config.fusion_ffn_dim,
            num_layers=config.fusion_layers,
            dropout=config.fusion_dropout,
        )

        self.aag = AutonomousAnchorGeneration(
            in_channels=config.visual_channels,
            num_points=config.num_points,
            image_size=config.image_size,
            init_tau=config.init_tau,
            learnable_tau=config.learnable_tau,
            tau_eps=config.tau_eps,
            logit_scale_cap=config.logit_scale_cap,
            box_mlp_hidden=config.box_mlp_hidden,
            box_param_mode=config.box_param_mode,
        )

        self.dense_head = (
            DenseMaskPromptHead(config.visual_channels) if config.dense_decoding_only else None
        )

    def train(self, mode: bool = True) -> "CMASAM":
        super().train(mode)
        self.text_encoder.eval()
        self.sam.prompt_encoder.eval()
        self.sam.mask_decoder.eval()
        return self

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """Returns compact state_dict (~18MB instead of 617MB) with trainable params & BN buffers."""
        trainable_names = {n for n, p in self.named_parameters() if p.requires_grad}
        compact: Dict[str, torch.Tensor] = {}
        for k, v in self.state_dict().items():
            if (
                k in trainable_names
                or k.startswith("fusion.")
                or k.startswith("aag.")
                or k.startswith("dense_head.")
            ):
                compact[k] = v.detach().cpu()
        return compact

    def encode_text(
        self,
        prompts: Union[List[str], Dict[str, torch.Tensor]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(prompts, list):
            tokens = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
        else:
            tokens = {k: v.to(device) for k, v in prompts.items()}

        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=tokens["input_ids"],
                attention_mask=tokens.get("attention_mask", None),
            )
        f_t_seq = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )
        f_t_pooled = (
            outputs.pooler_output
            if getattr(outputs, "pooler_output", None) is not None
            else f_t_seq.mean(dim=1)
        )
        attn_mask = tokens.get(
            "attention_mask",
            torch.ones(f_t_seq.shape[:2], device=device, dtype=torch.long),
        )
        return f_t_seq, f_t_pooled, attn_mask

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        if self.backend == "transformers":
            vis_out = self.sam.vision_encoder(pixel_values=images)
            return (
                vis_out.last_hidden_state
                if hasattr(vis_out, "last_hidden_state")
                else vis_out[0]
            )
        return self.sam.image_encoder(images)

    def _decode_transformers(
        self,
        f_v: torch.Tensor,
        f_fused: torch.Tensor,
        points_px: Optional[torch.Tensor],
        boxes_px: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz = f_v.shape[0]
        device = f_v.device

        dense_delta = None
        if self.config.dense_decoding_only and self.dense_head is not None:
            dense_delta = self.dense_head(f_fused).float()

        with torch.amp.autocast(device_type=device.type, enabled=False):
            f_v_fp32 = f_v.float()
            input_points = None
            input_labels = None
            input_boxes = None

            if points_px is not None:
                input_points = points_px.float().unsqueeze(1)
                input_labels = torch.ones(
                    (bsz, 1, points_px.shape[1]), dtype=torch.long, device=device
                )

            if boxes_px is not None:
                input_boxes = boxes_px.float().unsqueeze(1)

            if input_points is None and input_boxes is None:
                input_points = torch.zeros((bsz, 1, 1, 2), dtype=torch.float32, device=device)
                input_labels = torch.full((bsz, 1, 1), -1, dtype=torch.long, device=device)

            prompt_outputs = self.sam.prompt_encoder(
                input_points=input_points,
                input_labels=input_labels,
                input_boxes=input_boxes,
                input_masks=None,
            )
            sparse_embeddings = (
                prompt_outputs.sparse_embeddings
                if hasattr(prompt_outputs, "sparse_embeddings")
                else prompt_outputs[0]
            )
            dense_embeddings = (
                prompt_outputs.dense_embeddings
                if hasattr(prompt_outputs, "dense_embeddings")
                else prompt_outputs[1]
            )

            if dense_delta is not None:
                if dense_embeddings.dim() == 5:
                    dense_embeddings = dense_embeddings + dense_delta.unsqueeze(1)
                else:
                    dense_embeddings = dense_embeddings + dense_delta

            image_pe = self.sam.get_image_wide_positional_embeddings().float()
            image_pe = image_pe.repeat(bsz, 1, 1, 1)

            decoder_outputs = self.sam.mask_decoder(
                image_embeddings=f_v_fp32,
                image_positional_embeddings=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            low_res_masks = (
                decoder_outputs.pred_masks
                if hasattr(decoder_outputs, "pred_masks")
                else decoder_outputs[0]
            )

            if low_res_masks.dim() == 5:
                low_res_masks = low_res_masks.squeeze(1)
            return low_res_masks

    def _decode_segment_anything(
        self,
        f_v: torch.Tensor,
        f_fused: torch.Tensor,
        points_px: Optional[torch.Tensor],
        boxes_px: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz = f_v.shape[0]
        device = f_v.device
        dense_delta = (
            self.dense_head(f_fused).float()
            if (self.config.dense_decoding_only and self.dense_head is not None)
            else None
        )

        with torch.amp.autocast(device_type=device.type, enabled=False):
            f_v_fp32 = f_v.float()
            image_pe = self.sam.prompt_encoder.get_dense_pe().float()
            masks_per_batch: List[torch.Tensor] = []

            for b in range(bsz):
                pts_tuple = None
                if points_px is not None:
                    pts_b = points_px[b : b + 1].float()
                    lbl_b = torch.ones((1, pts_b.shape[1]), dtype=torch.int, device=device)
                    pts_tuple = (pts_b, lbl_b)

                box_b = boxes_px[b : b + 1].float() if boxes_px is not None else None

                sparse_emb, dense_emb = self.sam.prompt_encoder(
                    points=pts_tuple,
                    boxes=box_b,
                    masks=None,
                )
                if dense_delta is not None:
                    dense_emb = dense_emb + dense_delta[b : b + 1]

                dec_out = self.sam.mask_decoder(
                    image_embeddings=f_v_fp32[b : b + 1],
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )
                masks_per_batch.append(dec_out[0])

            return torch.cat(masks_per_batch, dim=0)

    def forward(
        self,
        images: torch.Tensor,
        prompts: Union[List[str], Dict[str, torch.Tensor]],
        output_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        device = images.device
        target_hw = output_size if output_size is not None else (images.shape[-2], images.shape[-1])

        f_v = self.encode_image(images)

        if self.config.saliency_heuristic_only:
            pts_px, box_px = extract_saliency_heuristic_anchors(
                f_v,
                num_points=self.config.num_points,
                image_size=self.config.image_size,
            )
            pts_norm = pts_px / float(self.config.image_size)
            box_norm = box_px / float(self.config.image_size)
            m = f_v
            tau = torch.tensor(self.config.init_tau, device=device, dtype=torch.float32)
            act_maps = torch.norm(f_v.float(), p=2, dim=1, keepdim=True)
        else:
            f_t_seq, f_t_pooled, text_mask = self.encode_text(prompts, device=device)
            f_t_seq = f_t_seq.to(dtype=f_v.dtype)
            f_t_pooled = f_t_pooled.to(dtype=f_v.dtype)
            m = self.fusion(f_v, f_t_seq, f_t_pooled, text_mask=text_mask)
            aag_out = self.aag(m)
            pts_px = aag_out["points"]
            pts_norm = aag_out["points_norm"]
            box_px = aag_out["boxes"]
            box_norm = aag_out["boxes_norm"]
            tau = aag_out["tau"]
            act_maps = aag_out["activation_maps"]

        use_pts_active = (
            self.config.use_points
            and not self.config.dense_decoding_only
            and not self.config.saliency_heuristic_only
        )
        use_box_active = (
            self.config.use_box
            and not self.config.dense_decoding_only
            and not self.config.saliency_heuristic_only
        )

        active_points = (
            pts_px if (self.config.use_points and not self.config.dense_decoding_only) else None
        )
        active_boxes = (
            box_px if (self.config.use_box and not self.config.dense_decoding_only) else None
        )

        if self.backend == "transformers":
            low_res_logits = self._decode_transformers(f_v, m, active_points, active_boxes)
        else:
            low_res_logits = self._decode_segment_anything(f_v, m, active_points, active_boxes)

        mask_logits = F.interpolate(
            low_res_logits,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
        mask_probs = torch.sigmoid(mask_logits)
        pred_masks = (mask_probs >= 0.5).float()

        return {
            "mask_logits": mask_logits,
            "mask_probs": mask_probs,
            "pred_masks": pred_masks,
            "points": pts_px,
            "boxes": box_px,
            "points_norm_active": pts_norm if use_pts_active else None,
            "boxes_norm_active": box_norm if use_box_active else None,
            "fused_features": m,
            "activation_maps": act_maps,
            "tau": tau,
        }

    def trainable_parameters_summary(self) -> Dict[str, Union[int, float]]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora_params = sum(
            p.numel()
            for n, p in self.named_parameters()
            if p.requires_grad and "lora_" in n
        )
        fusion_params = sum(p.numel() for p in self.fusion.parameters() if p.requires_grad)
        aag_params = sum(p.numel() for p in self.aag.parameters() if p.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "trainable_params_millions": trainable / 1e6,
            "lora_params_millions": lora_params / 1e6,
            "fusion_params_millions": fusion_params / 1e6,
            "aag_params_millions": aag_params / 1e6,
        }


# =============================================================================
# Section III-E.2: Joint Mask + Mask-Derived Geometric Anchor Supervision Loss
# =============================================================================

@torch.no_grad()
def extract_mask_derived_anchors(
    gt_masks: torch.Tensor,
    num_points: int,
    grid_size: int = 128,
    box_margin: float = 0.015,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Zero-annotation-cost extraction of canonical interior point anchors P_gt in [0, 1]^{B x K x 2}
    and tight bounding boxes B_gt in [0, 1]^{B x 4} directly from ground-truth binary masks Y.
    """
    bsz = gt_masks.shape[0]
    device = gt_masks.device

    gt_small = F.interpolate(gt_masks.float(), size=(grid_size, grid_size), mode="nearest")[:, 0]
    pts_targets = torch.full((bsz, num_points, 2), 0.5, dtype=torch.float32, device=device)
    box_targets = torch.tensor(
        [0.25, 0.25, 0.75, 0.75], dtype=torch.float32, device=device
    ).repeat(bsz, 1)
    valid_mask = torch.zeros(bsz, dtype=torch.bool, device=device)

    for b in range(bsz):
        fg_yx = torch.nonzero(gt_small[b] > 0.5, as_tuple=False).float()
        if fg_yx.shape[0] == 0:
            continue
        valid_mask[b] = True

        # Normalized foreground coordinates in [0, 1]
        ys = (fg_yx[:, 0] + 0.5) / float(grid_size)
        xs = (fg_yx[:, 1] + 0.5) / float(grid_size)
        fg_xy = torch.stack([xs, ys], dim=-1)  # (M, 2)

        # Tightest bounding box + slight margin
        x_min = (xs.min() - box_margin).clamp(0.0, 1.0)
        y_min = (ys.min() - box_margin).clamp(0.0, 1.0)
        x_max = (xs.max() + box_margin).clamp(0.0, 1.0)
        y_max = (ys.max() + box_margin).clamp(0.0, 1.0)
        x_max = torch.maximum(x_max, (x_min + 1e-3).clamp(max=1.0))
        y_max = torch.maximum(y_max, (y_min + 1e-3).clamp(max=1.0))
        box_targets[b] = torch.stack([x_min, y_min, x_max, y_max])

        # Central foreground anchor (centroid snapped to nearest interior foreground pixel)
        centroid = fg_xy.mean(dim=0, keepdim=True)  # (1, 2)
        d_center = torch.sum((fg_xy - centroid) ** 2, dim=-1)
        center_pt = fg_xy[torch.argmin(d_center)]
        pts_targets[b, 0] = center_pt

        if num_points > 1:
            # Partition foreground into K-1 ordered angular sectors around centroid
            angles = torch.atan2(ys - center_pt[1], xs - center_pt[0])
            sorted_idx = torch.argsort(angles)
            fg_sorted = fg_xy[sorted_idx]
            chunks = torch.tensor_split(fg_sorted, num_points - 1, dim=0)
            for k_idx, chunk in enumerate(chunks, start=1):
                if chunk.shape[0] == 0:
                    pts_targets[b, k_idx] = center_pt
                else:
                    sec_mean = 0.7 * chunk.mean(dim=0) + 0.3 * center_pt
                    d_sec = torch.sum((chunk - sec_mean.unsqueeze(0)) ** 2, dim=-1)
                    pts_targets[b, k_idx] = chunk[torch.argmin(d_sec)]

    return pts_targets, box_targets, valid_mask


def compute_generalized_box_iou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Computes 1 - GIoU loss for normalized (x_min, y_min, x_max, y_max) boxes."""
    px1, py1, px2, py2 = pred_boxes.unbind(dim=-1)
    tx1, ty1, tx2, ty2 = target_boxes.unbind(dim=-1)

    ix1 = torch.maximum(px1, tx1)
    iy1 = torch.maximum(py1, ty1)
    ix2 = torch.minimum(px2, tx2)
    iy2 = torch.minimum(py2, ty2)
    inter = (ix2 - ix1).clamp(min=0.0) * (iy2 - iy1).clamp(min=0.0)

    area_p = (px2 - px1).clamp(min=0.0) * (py2 - py1).clamp(min=0.0)
    area_t = (tx2 - tx1).clamp(min=0.0) * (ty2 - ty1).clamp(min=0.0)
    union = area_p + area_t - inter + eps
    iou = inter / union

    cx1 = torch.minimum(px1, tx1)
    cy1 = torch.minimum(py1, ty1)
    cx2 = torch.maximum(px2, tx2)
    cy2 = torch.maximum(py2, ty2)
    c_area = (cx2 - cx1).clamp(min=0.0) * (cy2 - cy1).clamp(min=0.0) + eps

    giou = iou - (c_area - union) / c_area
    return (1.0 - giou).mean()


class CMASAMJointLoss(nn.Module):
    """
    Joint end-to-end objective:
        L_total = lambda_dice * L_dice + lambda_bce * L_bce + lambda_anchor * (L_pts + L_box)
    Requires ONLY ground-truth binary masks Y during training (zero manual point/box labels).
    """

    def __init__(
        self,
        lambda_dice: float = 1.0,
        lambda_bce: float = 1.0,
        lambda_anchor: float = 2.0,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce
        self.lambda_anchor = lambda_anchor
        self.smooth = smooth

    def forward(
        self,
        outputs: Union[torch.Tensor, Dict[str, Any]],
        gt_masks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(outputs, dict):
            mask_logits = outputs["mask_logits"]
            pts_norm = outputs.get("points_norm_active", None)
            box_norm = outputs.get("boxes_norm_active", None)
        else:
            mask_logits = outputs
            pts_norm = None
            box_norm = None

        logits_fp32 = mask_logits.float()
        gt_float = gt_masks.float()
        probs = torch.sigmoid(logits_fp32)

        probs_flat = probs.view(probs.shape[0], -1)
        gt_flat = gt_float.view(gt_float.shape[0], -1)

        intersection = (probs_flat * gt_flat).sum(dim=-1)
        denom = probs_flat.sum(dim=-1) + gt_flat.sum(dim=-1)
        dice_score = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        loss_dice = (1.0 - dice_score).mean()

        loss_bce = F.binary_cross_entropy_with_logits(logits_fp32, gt_float, reduction="mean")
        loss_anchor = torch.zeros((), device=logits_fp32.device, dtype=torch.float32)

        if self.lambda_anchor > 0.0 and (pts_norm is not None or box_norm is not None):
            k_pts = pts_norm.shape[1] if pts_norm is not None else 1
            pts_gt, box_gt, valid = extract_mask_derived_anchors(gt_float, num_points=k_pts)
            if valid.any():
                if pts_norm is not None:
                    loss_pts = F.smooth_l1_loss(
                        pts_norm.float()[valid], pts_gt[valid], beta=0.02
                    )
                    loss_anchor = loss_anchor + loss_pts
                if box_norm is not None:
                    b_pred = box_norm.float()[valid]
                    b_targ = box_gt[valid]
                    loss_box = F.smooth_l1_loss(b_pred, b_targ, beta=0.02) + 0.5 * compute_generalized_box_iou_loss(b_pred, b_targ)
                    loss_anchor = loss_anchor + loss_box

        loss_total = (
            self.lambda_dice * loss_dice
            + self.lambda_bce * loss_bce
            + self.lambda_anchor * loss_anchor
        )

        return {
            "loss": loss_total,
            "loss_dice": loss_dice.detach(),
            "loss_bce": loss_bce.detach(),
            "loss_anchor": loss_anchor.detach(),
        }
