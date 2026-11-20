# CMA-SAM: Cross-Modal Anchor Generation for Text-Driven Medical Image Segmentation

Official PyTorch implementation, empirical benchmark suite, and camera-ready manuscript for **CMA-SAM** (*Cross-Modal Anchor Generation for Text-Driven Medical Image Segmentation*).

📄 **[Paper PDF (`paper/final.pdf`)](./paper/final.pdf)** | 📝 **[LaTeX Source (`paper/final.tex`)](./paper/final.tex)** | 📊 **[Measured Metrics (`results/`)](./results/)**

CMA-SAM bridges continuous vision-language representations and the discrete coordinate-driven prompt encoder of the Segment Anything Model (SAM). By fusing LoRA-adapted SAM visual features (`facebook/sam-vit-base`) with frozen CLIP text embeddings (`openai/clip-vit-base-patch32`) via Bidirectional Cross-Attention and regressing bounded Spatial Softmax point anchors ($S = c\tanh(f_{\text{conv}}(M))$) alongside a global center-size bounding box, CMA-SAM decouples semantic target localization from high-frequency boundary delineation. During training, zero-cost geometric anchor targets are extracted directly from binary ground-truth masks, while inference requires **only an input image and a text query** with zero manual prompts.

## Empirical Highlights

Evaluated across **Kvasir-SEG** (gastrointestinal endoscopy, `800 / 100 / 100`), **ISIC 2018 Task 1** (dermoscopy, `2075 / 259 / 260`), and external cross-cohort **HAM10000** (dermoscopy, `8012 / 1002 / 1001`) on 2x NVIDIA A800 (80 GB PCIe) GPUs:

| Evaluation Suite | Metric | Baseline | CMA-SAM (Ours) |
| :--- | :--- | :---: | :---: |
| **Anchor Localization Error (Kvasir-SEG)** | Point ALE (64x64 grid px) $\downarrow$ | `33.16` (Visual Saliency) | **`3.92`** (-88.2%) |
| **Anchor Localization Error (ISIC 2018)** | Point ALE (64x64 grid px) $\downarrow$ | `21.52` (Visual Saliency) | **`1.30`** (-94.0%) |
| **Anchor Localization Error (HAM10000, Zero-Shot)** | Point ALE (64x64 grid px) $\downarrow$ | `20.37` (Visual Saliency) | **`2.39`** (-88.3%) |
| **Cross-Domain Few-Shot (1 / 5 / 10-shot)** | Test DSC (%) $\uparrow$ | `9.27 / 27.28 / 48.91` | **`43.83 / 50.80 / 64.18`** |
| **Prompt Verbosity Stability (Simple vs. Complex)** | $|\Delta \text{DSC}|$ (pp) $\downarrow$ | `0.66` (Dense Decoding) | **`0.06`** (11.4x stabler) |
| **In-Domain Segmentation (Kvasir-SEG / ISIC 2018)** | Test DSC (%) $\uparrow$ | `89.26` (Dense Ceiling) | **`88.66` / `90.57`** |
| **Parameter & Latency Efficiency (1024x1024)** | Trainable Params / Latency | `5.20M` / `94.51 ms` | **`4.54M`** / **`94.13 ms`** |

All measured JSON metrics, auto-generated LaTeX snippets, and 300-DPI qualitative overlays are archived in [`results/`](./results/).

## Repository Structure

```text
.
├── cma_sam_model.py                      # LoRA-SAM + CLIP + JointFeatureFusion + Bounded AAG + JointLoss
├── datasets_and_metrics.py               # Deterministic splits, 16-core 1024x1024 cache, DSC/IoU/ALE/Latency
├── run_experiments.py                    # DDP & single-GPU trainer, ablations, plotting & LaTeX exporter
├── requirements.txt                      # Python dependencies
├── paper/
│   ├── final.tex                         # Camera-ready IEEEtran manuscript with TikZ/PGFPlots figures
│   ├── references.bib                    # Complete BibTeX bibliography
│   ├── final.pdf                         # Compiled camera-ready PDF manuscript
│   └── figures/                          # Embedded 300-DPI qualitative PDF overlays (Figs 9 & 13)
├── results/
│   ├── *.json                            # Measured experimental metrics (Tables II-VIII, Figs 7-12)
│   ├── real_experiment_tables_and_figures.tex
│   └── figures/                          # 300-DPI qualitative PDF/PNG segmentation overlays
└── scripts/
    ├── download_datasets.py              # Official downloader for Kvasir-SEG, ISIC 2018 & HAM10000
    ├── run_all_rtx3090x4.sh              # End-to-end SIGHUP/OOM-hardened DDP experiment launcher
    ├── safe_torchrun.py                  # SIGHUP-immune wrapper around torch.distributed.run
    └── verify_math_and_smoke_test.py     # Unit tests for Spatial Softmax Jacobians & end-to-end shapes
```

## Installation

```bash
conda create -n cmasam python=3.10 -y
conda activate cmasam
pip install -r requirements.txt
```

## Dataset Preparation

Download **Kvasir-SEG**, **ISIC 2018**, and **HAM10000** into `./data/`:

```bash
python3 scripts/download_datasets.py --data_dir ./data --datasets all
```

## Running the Full Experimental Suite

Launch the complete end-to-end training, ablation, efficiency, and figure-generation pipeline (supports automatic mid-epoch checkpoint resume and multi-GPU PyTorch DDP):

```bash
NUM_GPUS=2 BATCH_SIZE=2 GRAD_ACCUM=8 bash scripts/run_all_rtx3090x4.sh
```

Individual stages can also be executed standalone via `--mode` (`cache_resized`, `splits`, `zeroshot`, `fewshot`, `ablation_aag`, `ablation_tau`, `ablation_fusion`, `prompt_complexity`, `efficiency_and_ale`, `qualitative`, `export_latex`).

## Compiling the Manuscript

```bash
cd paper
pdflatex -interaction=nonstopmode final.tex
bibtex final
pdflatex -interaction=nonstopmode final.tex
pdflatex -interaction=nonstopmode final.tex
```

## License

Released under the [MIT License](./LICENSE).
