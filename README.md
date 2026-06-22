# DAGF-Swin

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Scientific Reports](https://img.shields.io/badge/Journal-Scientific%20Reports-blue)](https://www.nature.com/srep/)

**Dual-site Attention Fusion with Swin Transformer for detail-smearing image restoration.**

Official PyTorch implementation by **Jenshin Zhao**, released in connection with our submission to [*Scientific Reports*](https://www.nature.com/srep/) (Nature Portfolio).

---

## Downloads（百度网盘）

| Resource | Link | Code |
|----------|------|------|
| **TrainDataset.zip** | [Download](https://pan.baidu.com/s/1lrpjz0RDhEOBCZ6eIpkzUA?pwd=pdmk) | `pdmk` |
| **TestDataset.zip** | [Download](https://pan.baidu.com/s/1oL_ipYsWL06gDxGFrwkKnA?pwd=q4j5) | `q4j5` |
| **Checkpoints** (`best_*.pth`) | [Download](https://pan.baidu.com/s/1AOiekmrW7FrBnfMEdW-9qw?pwd=ah3p) | `ah3p` |

| After download | Extract to |
|----------------|------------|
| TrainDataset.zip | `data/train/` (`gt/`, `low/`) |
| TestDataset.zip | `data/test/` (`gt/`, `low/`) |
| checkpoint | `weights/` |

Checkpoint guide: **[weights/CHECKPOINTS.md](weights/CHECKPOINTS.md)** · Dataset guide: **[data/README.md](data/README.md)**

---

## Quick start

```bash
# 1. Clone (code lives in the DAGF-swin/ subfolder on GitHub)
git clone https://github.com/zjx7453/DAGF-Swin.git
cd DAGF-Swin/DAGF-swin

# 2. Environment
conda create -n dagf-swin python=3.10 -y && conda activate dagf-swin
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 3. Download dataset & checkpoints from Baidu Netdisk (table above), then:
python inference.py \
  --input ./data/test/low \
  --output ./outputs \
  --checkpoint ./weights/best_score.pth \
  --gt_dir ./data/test/gt \
  --compute_metrics \
  --device cuda \
  --use_amp
```

> Default checkpoint: **`best_score.pth`** (best composite validation score). See [CHECKPOINTS.md](weights/CHECKPOINTS.md) for `best_psnr.pth`, `best_lpips.pth`, etc.

---

## Highlights

- **SDAF / DMSA-Fusion** — dual-site shallow–deep adaptive fusion
- **Swin RSTB** backbone with shifted-window attention and **GConvFF**
- **YCbCr-Y** restoration (Cb/Cr preserved)
- **3:3:3:1** detail-smearing degradation for training (`degradation.py`)
- Reproducible **inference**, **benchmark**, and **data preparation** scripts

---

## Repository layout

```
DAGF-swin/
├── model_v3.py              # Main model (inference default)
├── model_v2.py              # V2 baseline
├── inference.py             # Restore images + PSNR/SSIM/LPIPS/NIQE
├── benchmark.py             # Params / FLOPs / latency / GPU memory
├── degradation.py           # Degradation synthesis (3:3:3:1)
├── utils_metrics.py         # PSNR / SSIM helpers
├── scripts/
│   ├── prepare_dataset.py   # Build gt/ + low/ pairs from clean images
│   └── download_pretrained.py
├── data/README.md           # Dataset download & layout
├── weights/CHECKPOINTS.md   # Meaning of best_*.pth files
├── docs/
│   ├── BENCHMARK.md         # Formal efficiency protocol
│   ├── DATA.md              # Data preparation guide
│   └── RELEASE.md           # DOI & code-availability text
├── environment.yml
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

---

## Installation

**pip**

```bash
conda create -n dagf-swin python=3.10 -y
conda activate dagf-swin
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**conda**

```bash
conda env create -f environment.yml
conda activate dagf-swin
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Inference

**Single folder / batch restore**

```bash
python inference.py \
  --input ./data/test/low \
  --output ./outputs \
  --checkpoint ./weights/best_score.pth \
  --device cuda \
  --use_amp
```

**With metrics** (GT filenames must match inputs)

```bash
python inference.py \
  --input ./data/test/low \
  --output ./outputs \
  --checkpoint ./weights/best_score.pth \
  --gt_dir ./data/test/gt \
  --compute_metrics \
  --device cuda
```

**Large images** (avoid OOM)

```bash
python inference.py ... --tile_size 512 --overlap 32
```

---

## Data

**Option A — Download** (recommended): see [data/README.md](data/README.md) and Baidu links above.

**Option B — Synthesize** from clean images:

```bash
python scripts/prepare_dataset.py \
  --gt_dir path/to/clean_images \
  --out_dir ./data/train \
  --mode random \
  --seed 42
```

Full protocol: **[docs/DATA.md](docs/DATA.md)**

---

## Benchmark

```bash
python benchmark.py --model ycbcr --H 512 --W 512 --device cuda \
  --warmup 10 --repeats 100 --mem_repeats 3
```

Formal protocol: **[docs/BENCHMARK.md](docs/BENCHMARK.md)**

---

## Model configuration

| Parameter | Default |
|-----------|---------|
| `dim` | 64 |
| `num_rstb` | 4 |
| `blocks_per_rstb` | 6 |
| `num_heads` | 4 |
| `window_size` | 8 |

- **YCbCr (default):** `YCbCr_RestorationSwinNetV3` — restores luminance Y only  
- **RGB:** `RestorationSwinNetV3(in_channels=3, ...)`

---

## Permanent links

| Item | Link |
|------|------|
| Source code | [github.com/zjx7453/DAGF-Swin](https://github.com/zjx7453/DAGF-Swin) |
| Code subdirectory | [DAGF-swin/](https://github.com/zjx7453/DAGF-Swin/tree/main/DAGF-swin) |
| Release tag | [v1.0.0](https://github.com/zjx7453/DAGF-Swin/releases/tag/v1.0.0) |
| Zenodo (DOI) | `https://doi.org/10.5281/zenodo.XXXXXXX
        
        ` *(pending)* |

See **[docs/RELEASE.md](docs/RELEASE.md)** for archival instructions.

---

## Citation

```bibtex
@software{dagf_swin2026,
  title   = {DAGF-Swin: Dual-site Attention Fusion with Swin Transformer for Detail-Smearing Image Restoration},
  author  = {Zhao, Jenshin},
  year    = {2026},
  journal = {Scientific Reports},
  note    = {Software v1.0.0. Code: https://github.com/zjx7453/DAGF-Swin. DOI: 10.5281/zenodo.XXXXXXX},
  doi     = {10.5281/zenodo.XXXXXXX}
}
```

Machine-readable: **[CITATION.cff](CITATION.cff)**

---

## Code & data availability (for the paper)

> Source code, pretrained checkpoints, train/test datasets, data-preparation scripts, and the evaluation protocol are available at [https://github.com/zjx7453/DAGF-Swin](https://github.com/zjx7453/DAGF-Swin) and via Baidu Netdisk (links in README). Datasets: TrainDataset.zip (code `pdmk`), TestDataset.zip (code `q4j5`); checkpoints (code `ah3p`). This release accompanies our submission to *Scientific Reports*.

---

## License

[MIT License](LICENSE) · Copyright (c) 2026 Jenshin Zhao

---

## Contact

GitHub Issues: [zjx7453/DAGF-Swin](https://github.com/zjx7453/DAGF-Swin/issues)  
Corresponding author: **Jenshin Zhao**
