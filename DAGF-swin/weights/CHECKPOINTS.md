# Pretrained Checkpoints 说明

训练过程中，验证集上每出现一次**新的最佳指标**，就会单独保存一个 `.pth` 文件。  
五个文件**结构相同**（均含 `model_state_dict` / EMA 权重、`model_config`、epoch 等），区别仅在于**按哪一项指标选为“最佳”**。

> 默认推理推荐：**`best_score.pth`**（综合评分最优，与 `inference.py` 默认一致）。

---

## 文件一览

| 文件名 | 保存条件 | 指标方向 | 含义 |
|--------|----------|----------|------|
| **`best_score.pth`** | 验证集 **综合 Score** 创新高 | 越高越好 ↑ | **默认推荐**。多指标加权后的整体最优模型 |
| **`best_psnr.pth`** | 验证集 **PSNR** 创新高 | 越高越好 ↑ | 像素误差最小，数值保真度最高 |
| **`best_ssim.pth`** | 验证集 **SSIM** 创新高 | 越高越好 ↑ | 结构相似度最高，轮廓/布局最接近 GT |
| **`best_lpips.pth`** | 验证集 **LPIPS** 创新低 | 越低越好 ↓ | 感知距离最小，人眼观感通常更自然 |
| **`best_niqe.pth`** | 验证集 **NIQE** 创新低 | 越低越好 ↓ | 无参考自然度最好（不依赖 GT 的质量估计） |

---

## 各指标简要说明

### PSNR（`best_psnr.pth`）

- **Peak Signal-to-Noise Ratio**，峰值信噪比，单位 dB  
- 衡量恢复图与 GT 的**像素级误差**  
- **优点**：客观、可复现  
- **局限**：与主观视觉不完全一致，对纹理/感知差异不敏感  

### SSIM（`best_ssim.pth`）

- **Structural Similarity Index**  
- 衡量亮度、对比度、结构的相似性（0–1）  
- **优点**：比 PSNR 更关注结构  
- **局限**：仍可能忽略某些感知细节  

### LPIPS（`best_lpips.pth`）

- **Learned Perceptual Image Patch Similarity**  
- 基于深度特征的**感知距离**，越小越好  
- **优点**：更接近人眼对模糊、纹理、伪影的判断  
- **适用**：强调视觉质量、细节自然度时  

### NIQE（`best_niqe.pth`）

- **Naturalness Image Quality Evaluator**  
- **无参考**指标：不需要 GT，估计图像“自然度”，越小越好  
- **优点**：反映真实感、噪声/失真程度  
- **局限**：与 GT 一致性无直接关系  

### Score（`best_score.pth`）

- 训练脚本中的**综合评分**，由 PSNR、SSIM、LPIPS、NIQE 加权得到（默认 `score_mode=balanced`）  
- **balanced 模式权重**（见 `train_v2.py` → `calculate_score`）：

  ```
  Score = 0.1 × PSNR_norm + 0.45 × SSIM + 0.25 × LPIPS_norm + 0.2 × NIQE_norm
  ```

  其中 PSNR/LPIPS/NIQE 会先归一化到 [0,1]，LPIPS/NIQE 为“越小越好”故用 `1 - norm`。

- **优点**：单一 checkpoint 在多项指标间折中  
- **论文 / 开源默认**：建议使用 **`best_score.pth`**

---

## 五个文件为何不同？

训练时**每个 epoch 验证一次**，各指标**最佳时刻往往不在同一 epoch**，因此会得到 5 个不同的权重文件。例如：

- `best_psnr.pth` 可能 PSNR 最高，但 LPIPS 不是最低  
- `best_lpips.pth` 观感可能更好，但 PSNR 略低  

这是正常现象，不是错误。

---

## 如何使用

```bash
# 默认综合最优
python inference.py --input ./data/test/low --output ./outputs \
  --checkpoint weights/best_score.pth --device cuda

# 若论文表格报的是 PSNR 最优结果
python inference.py --checkpoint weights/best_psnr.pth ...

# 若强调感知质量
python inference.py --checkpoint weights/best_lpips.pth ...
```

---

## 文件内容结构（每个 .pth 相同格式）

每个 checkpoint 通常包含：

| 键名 | 说明 |
|------|------|
| `model_state_dict` / `state_dict` | 模型权重 |
| `ema_state_dict`（若有） | EMA 权重；`inference.py` 会优先加载 |
| `model_config` | `dim`, `num_rstb`, `blocks_per_rstb` 等 |
| `epoch` | 保存时的训练轮次 |
| `best_*` | 该文件对应指标的历史最佳值 |
| `metrics` | 保存当次的 PSNR / SSIM / LPIPS / NIQE / Score |

---

## 开源 / Release 建议

| 场景 | 建议上传 |
|------|----------|
| 一般用户复现 | **`best_score.pth`**（可另存为 `dagf_swin_best.pth`） |
| 论文主表报 PSNR | 正文说明指标，必要时额外提供 `best_psnr.pth` |
| 消融 / 补充材料 | 可按需提供全部 5 个文件 |

GitHub 仓库**不要**直接 commit 大体积 `.pth`；请使用 **Baidu Netdisk**（见 [README.md](README.md)）或 **GitHub Releases** / **Zenodo**，并填写 SHA256（可选）。

---

## 联系

作者：Jenshin Zhao · 仓库：[zjx7453/DAGF-Swin](https://github.com/zjx7453/DAGF-Swin)
