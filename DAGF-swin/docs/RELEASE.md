# Release, permanent link & DOI

This code release is associated with a submission to **Scientific Reports** (Nature Portfolio).

## Permanent links (fill after publishing)

| Resource | URL |
|----------|-----|
| **GitHub repository** | [https://github.com/zjx7453/DAGF-Swin](https://github.com/zjx7453/DAGF-Swin) |
| **Source subdirectory** | [DAGF-swin/](https://github.com/zjx7453/DAGF-Swin/tree/main/DAGF-swin) |
| **Latest release (tag)** | [v1.0.0](https://github.com/zjx7453/DAGF-Swin/releases/tag/v1.0.0) |
| **Zenodo archive (DOI)** | `https://doi.org/10.5281/zenodo.XXXXXXX` |

> Use the **Zenodo DOI** as the primary permanent identifier in the paper.  
> The GitHub release tag ensures reproducibility; Zenodo provides long-term archival.

## Steps to obtain a DOI (recommended)

1. **Create a GitHub release**  
   - Tag: `v1.0.0`  
   - Attach: `dagf_swin_best.pth` (or link to it in release notes if too large)  
   - Release notes: cite paper title, Scientific Reports submission, and checkpoint SHA256  

2. **Enable Zenodo–GitHub integration**  
   - Log in at [https://zenodo.org](https://zenodo.org) with GitHub  
   - Enable sync for `DAGF-swin`  
   - Create release on GitHub → Zenodo builds an archive automatically  

3. **Copy the DOI** from the Zenodo record into:  
   - `README.md` (badges + citation)  
   - `CITATION.cff` (`doi` field)  
   - Paper “Code availability” / “Data availability” section  

4. **Update pretrained download**  
   - Set `PRETRAINED_URL` in `scripts/download_pretrained.py`  
   - Document SHA256 in `weights/README.md`  

## Code availability statement (for the paper)

Suggested text:

> The source code, pretrained weights, data preparation scripts, and benchmark protocol for DAGF-Swin are publicly available at [https://github.com/zjx7453/DAGF-Swin](https://github.com/zjx7453/DAGF-Swin) (release `v1.0.0`, subdirectory `DAGF-swin/`) and archived on Zenodo (`https://doi.org/10.5281/zenodo.XXXXXXX`). This software release accompanies our submission to *Scientific Reports*.

## Reproducibility checklist

- [ ] Git tag matches paper (“code version v1.0.0”)  
- [ ] Zenodo DOI live and cited  
- [ ] `requirements.txt` / `environment.yml` tested on a clean machine  
- [ ] Pretrained weights downloadable and SHA256 documented  
- [ ] `docs/BENCHMARK.md` commands reproduce reported efficiency numbers  
- [ ] `docs/DATA.md` commands reproduce training degradation setup  
