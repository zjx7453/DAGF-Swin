# Benchmark protocol (DAGF-Swin)

This document defines the **reproducible efficiency benchmark** shipped with this repository.
Report all numbers in papers using this protocol.

## 1. Scope

| Item | Included | Excluded |
|------|----------|----------|
| Forward pass | `net(x)` in eval mode, `torch.no_grad()` | I/O, tiling, metric computation |
| Input | Single tensor, batch size **B=1** | DataLoader |
| Memory | PyTorch allocator peak | `nvidia-smi` process total |

## 2. Software & hardware (report in paper)

Record and report:

- GPU model & driver version
- PyTorch / CUDA version (`python -c "import torch; print(torch.__version__, torch.version.cuda)"`)
- Commit hash or release tag of this repository
- Command line used

## 3. Model & input settings (defaults)

| Parameter | Default |
|-----------|---------|
| Model | `YCbCr_RestorationSwinNetV3` (`--model ycbcr`) |
| `dim` | 64 |
| `num_rstb` | 4 |
| `blocks_per_rstb` | 6 |
| `num_heads` | 4 |
| `window_size` | 8 |
| Input resolution | `--H 512 --W 512` (also report 1080×1920 if needed) |
| Precision | FP32; optional `--fp16` (autocast) |

**Note:** Swin padding may increase effective size to `H'×W'` (multiples of 8). The script reports both nominal and padded megapixels.

## 4. Commands

### 4.1 Full benchmark (params + MACs + latency + memory)

```bash
python benchmark.py \
  --model ycbcr \
  --H 512 --W 512 \
  --device cuda \
  --warmup 10 \
  --repeats 100 \
  --mem_repeats 3
```

FP16 variant:

```bash
python benchmark.py --model ycbcr --H 512 --W 512 --device cuda --fp16 \
  --warmup 10 --repeats 100 --mem_repeats 3
```

### 4.2 Params / MACs only

```bash
python calc_params_flops.py --model_file model_v3 --net ycbcr --H 512 --W 512 --device cuda
```

Requires: `pip install thop`

## 5. Timing protocol

1. `torch.cuda.synchronize()` after each warmup and timed iteration  
2. Timer: **`torch.cuda.Event`** (fallback: `time.perf_counter` on CPU)  
3. Warmup: **`--warmup` iterations** (default 10), **excluded** from statistics  
4. Timed runs: **`--repeats`** (default 100)  
5. Report: **mean ± std** latency (ms), **FPS** = 1000 / mean_ms, **ms/Mpixel** = mean_ms / (H×W/1e6)

## 6. Memory protocol

1. `torch.cuda.empty_cache()`  
2. Same warmup as timing  
3. Each repeat: `reset_peak_memory_stats` → forward → `synchronize` → `max_memory_allocated`  
4. Report **max over `--mem_repeats`** (default 3), unit **MiB**

## 7. Quality evaluation (separate from efficiency)

For PSNR / SSIM / LPIPS / NIQE on a test set:

```bash
python inference.py \
  --input path/to/low \
  --output path/to/restored \
  --checkpoint weights/dagf_swin_best.pth \
  --gt_dir path/to/gt \
  --compute_metrics \
  --device cuda
```

- Metrics computed on restored outputs vs GT (same filenames).  
- Use **the same checkpoint and model config** as in the paper.  
- For images larger than GPU memory, add `--tile_size 512 --overlap 32` and report tiling settings.

## 8. Paper checklist

- [ ] GitHub release tag (e.g. `v1.0.0`)  
- [ ] Zenodo DOI (see `docs/RELEASE.md`)  
- [ ] GPU / PyTorch versions  
- [ ] Nominal input size + padded size if H or W ≢ 0 (mod 8)  
- [ ] Warmup & repeat counts  
- [ ] FP32 or FP16 autocast  
- [ ] Checkpoint filename & SHA256 (see `weights/README.md`)
