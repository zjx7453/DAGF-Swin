# DAGF-Swin

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Scientific Reports](https://img.shields.io/badge/Journal-Scientific%20Reports-blue)](https://www.nature.com/srep/)

**Dual-site Attention Fusion with Swin Transformer for detail-smearing image restoration.**

Official PyTorch implementation released in connection with our submission to *[Scientific Reports](https://www.nature.com/srep/)* (Nature Portfolio).

---

## Permanent links

| Item | Link |
|------|------|
| Source code | [https://github.com/zjx7453/DAGF-Swin](https://github.com/zjx7453/DAGF-Swin) (code in [`DAGF-swin/`](https://github.com/zjx7453/DAGF-Swin/tree/main/DAGF-swin)) |
| Release tag | [v1.0.0](https://github.com/zjx7453/DAGF-Swin/releases/tag/v1.0.0) *(create when publishing weights)* |
| Zenodo (DOI) | `https://doi.org/10.5281/zenodo.XXXXXXX` ← **fill after Zenodo archive** |

See **[docs/RELEASE.md](docs/RELEASE.md)** for DOI registration steps.

---

## Downloads（百度网盘）

| Resource | Baidu Netdisk | Code |
|----------|---------------|------|
| **Dataset** (`Data.zip`) | [Download](https://pan.baidu.com/s/1W_UJRRs2z0CFrzXwBqqwKQ?pwd=bsmr) | `bsmr` |
| **Checkpoints** (`best_*.pth`) | [Download](https://pan.baidu.com/s/1AOiekmrW7FrBnfMEdW-9qw?pwd=ah3p) | `ah3p` |

- Dataset → extract to [`data/`](data/README.md)  
- Checkpoints → extract to [`weights/`](weights/README.md)（默认推理用 `best_score.pth`）  
- Details: [weights/CHECKPOINTS.md](weights/CHECKPOINTS.md)

---

## Highlights

- YCbCr luminance restoration with **dual-site shallow–deep fusion (SDAF / DMSA-Fusion)**
- Swin RSTB backbone with **gated depthwise FFN (GConvFF)** and shifted-window attention
- Training-time **3:3:3:1 detail-smearing degradation** synthesis
- Scripts for **inference**, **benchmark**, **data preparation**, and **pretrained weight download**

---

## Repository layout

```
DAGF-swin/
├── model_v3.py              # Main model (inference default)
├── model_v2.py              # V2 variant
├── inference.py             # Restore images + optional metrics
├── benchmark.py             # Params / FLOPs / latency / GPU memory
├── calc_params_flops.py     # Params & FLOPs only
├── degradation.py           # Degradation synthesis (3:3:3:1)
├── utils_metrics.py         # PSNR / SSIM
├── scripts/
│   ├── prepare_dataset.py   # Build gt/ + low/ pairs
│   └── download_pretrained.py
├── weights/                 # Place dagf_swin_best.pth here
├── docs/
│   ├── BENCHMARK.md         # Formal benchmark protocol
│   ├── DATA.md              # Data preparation guide
│   └── RELEASE.md           # DOI & Scientific Reports code-availability text
├── environment.yml
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

---

## Environment setup

### Option A — pip

```bash
conda create -n dagf-swin python=3.10 -y
conda activate dagf-swin
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Option B — conda

```bash
conda env create -f environment.yml
conda activate dagf-swin
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Pretrained weights

Download from **[Baidu Netdisk](https://pan.baidu.com/s/1AOiekmrW7FrBnfMEdW-9qw?pwd=ah3p)** (extract code: `ah3p`), extract to `weights/`, then:

```bash
python inference.py --checkpoint weights/best_score.pth ...
```

Or:

```bash
python scripts/download_pretrained.py
# manual: see weights/README.md
```

Details: **[weights/README.md](weights/README.md)**

---

## Quick start — inference

```bash
python inference.py \
  --input path/to/low_images \
  --output path/to/restored \
  --checkpoint weights/dagf_swin_best.pth \
  --device cuda \
  --use_amp
```

With metrics (requires paired GT):

```bash
python inference.py \
  --input path/to/low \
  --output path/to/restored \
  --checkpoint weights/dagf_swin_best.pth \
  --gt_dir path/to/gt \
  --compute_metrics \
  --device cuda
```

Large images: `--tile_size 512 --overlap 32`

---

## Data preparation

```bash
python scripts/prepare_dataset.py \
  --gt_dir path/to/clean_images \
  --out_dir ./data/train \
  --mode random \
  --seed 42
```

Full guide: **[docs/DATA.md](docs/DATA.md)**  
Pre-built dataset: **[data/README.md](../data/README.md)** (Baidu Netdisk `Data.zip`, code `bsmr`)

---

## Benchmark protocol

Efficiency (latency, FPS, ms/Mpixel, peak memory):

```bash
python benchmark.py --model ycbcr --H 512 --W 512 --device cuda \
  --warmup 10 --repeats 100 --mem_repeats 3
```

Formal protocol (for paper reproduction): **[docs/BENCHMARK.md](docs/BENCHMARK.md)**

---

## Model configuration

| Parameter | Default |
|-----------|---------|
| `dim` | 64 |
| `num_rstb` | 4 |
| `blocks_per_rstb` | 6 |
| `num_heads` | 4 |
| `window_size` | 8 |

- **YCbCr (default):** `YCbCr_RestorationSwinNetV3` — restores Y; Cb/Cr unchanged  
- **RGB:** `RestorationSwinNetV3(in_channels=3, ...)`

---

## Citation

If you use this code, please cite:

```bibtex
@software{dagf_swin2026,
  title   = {DAGF-Swin: Dual-site Attention Fusion with Swin Transformer for Detail-Smearing Image Restoration},
  author  = {Zhao, Jenshin},
  year    = {2026},
  journal = {Scientific Reports},
  note    = {Software release v1.0.0 accompanying submission to Scientific Reports. Code: \url{https://github.com/zjx7453/DAGF-Swin}. DOI: 10.5281/zenodo.XXXXXXX},
  doi     = {10.5281/zenodo.XXXXXXX}
}
```

Also cite the paper when published. Machine-readable metadata: **[CITATION.cff](CITATION.cff)**.

---

## Code availability (for the paper)

> The source code, pretrained weights, data preparation scripts, and benchmark protocol are available at [https://github.com/zjx7453/DAGF-Swin](https://github.com/zjx7453/DAGF-Swin) (release `v1.0.0`, subdirectory `DAGF-swin/`) and permanently archived on Zenodo (`https://doi.org/10.5281/zenodo.XXXXXXX
        
        `). This release accompanies our submission to *Scientific Reports*.

---

## License

[MIT License](LICENSE)

---

## Contact

For questions about this code release, open a GitHub issue or contact **Jenshin Zhao** (corresponding author).
