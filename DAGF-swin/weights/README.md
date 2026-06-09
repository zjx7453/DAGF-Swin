# Pretrained weights

## Expected file

| File | Description |
|------|-------------|
| `dagf_swin_best.pth` | Best checkpoint (YCbCr V3, default architecture) |

Default architecture (must match inference):

- `dim=64`, `num_rstb=4`, `blocks_per_rstb=6`, `num_heads=4`, `window_size=8`, `use_dwconv=True`

## Download

After the authors publish the release:

```bash
python scripts/download_pretrained.py
```

Or manual download from **GitHub Releases** / **Zenodo** (see `docs/RELEASE.md`).

## SHA256 (fill after upload)

```
dagf_swin_best.pth  <PASTE_SHA256_HERE>
```

Verify:

```bash
# Linux / macOS
sha256sum weights/dagf_swin_best.pth

# Windows PowerShell
Get-FileHash weights\dagf_swin_best.pth -Algorithm SHA256
```

## Checkpoint format

Supported keys in `.pth`:

- Raw `state_dict`
- `{'model_state_dict': ...}` or `{'state_dict': ...}`

EMA weights are used automatically if present (`ema_state_dict` / `ema`).

## Placeholder

If the file is missing, inference will fail with `FileNotFoundError`.  
Upload the checkpoint to Releases before announcing the DOI.
