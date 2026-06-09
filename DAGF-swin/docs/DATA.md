# Data preparation

## 1. Directory layout

### Training (synthetic degradation from GT)

```
data/train/
  gt/    # clean images
  low/   # degraded inputs (same filenames as gt/)
```

Generate with:

```bash
python scripts/prepare_dataset.py \
  --gt_dir /path/to/clean_images \
  --out_dir ./data/train \
  --mode random \
  --seed 42
```

### Evaluation (paired test set)

```
data/test/
  low/   # degraded inputs
  gt/    # ground truth (optional, for metrics)
```

Place your own benchmark images here, or use `prepare_dataset.py` with a fixed `--mode` for controlled degradation ablations.

## 2. Degradation types

| Type | Description | Parameters |
|------|-------------|------------|
| `bilateral` | Edge-preserving smoothing, texture smearing | d∈{5,7,9}, σ∈[30,80] |
| `jpeg` | Compression artifacts | quality∈[50,85] |
| `downsampling` | Down-up sampling detail loss | scale∈{0.5,…,0.8} |
| `mixed` | Downsample → bilateral → JPEG | combined |
| `random` | **3:3:3:1** mix of the four above | training default |

Implementation: `degradation.py` (same logic as the paper training pipeline).

## 3. Examples

Single degradation type:

```bash
python scripts/prepare_dataset.py --gt_dir ./gt --out_dir ./data/jpeg_only --mode jpeg
```

Reproducible random mix:

```bash
python scripts/prepare_dataset.py --gt_dir ./gt --out_dir ./data/train --mode random --seed 0
```

## 4. Notes

- Input images: RGB, any common format (PNG/JPEG).  
- Output: PNG in `gt/` and `low/`.  
- Validation sets in the paper should **not** apply random degradation at test time; use fixed pairs in `low/` and `gt/`.  
- Large corpora: run per subfolder or extend the script for sharding if needed.
