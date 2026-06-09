# -*- coding: utf-8 -*-
"""
DAGF-Swin 推理脚本 — 细节涂抹/图像恢复

支持：
1. 单张或文件夹批量推理
2. EMA 权重自动加载
3. 大图 tile 切片（防 OOM）
4. 可选 PSNR / SSIM / LPIPS / NIQE
5. FP16 混合精度
"""

import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from tqdm import tqdm
import glob
import time
import lpips
import pyiqa
from pathlib import Path

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_v3 as model
import utils_metrics


# ==============================================================
#  图像预处理和后处理
# ==============================================================

def preprocess_image(img_path, device):
    """
    预处理图像：PIL -> Tensor [0, 1]
    
    Args:
        img_path: 图片路径（str或PIL.Image）
        device: 设备
    
    Returns:
        tensor: [1, 3, H, W]，值域[0, 1]
    """
    if isinstance(img_path, str):
        img = Image.open(img_path).convert('RGB')
    else:
        img = img_path
    
    # 转换为tensor，值域[0, 1]
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    tensor = transform(img).unsqueeze(0).to(device)
    return tensor


def postprocess_image(tensor):
    """
    后处理：Tensor -> PIL Image
    
    Args:
        tensor: [1, 3, H, W]，值域[0, 1]
    
    Returns:
        PIL.Image
    """
    # 确保值域在[0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    # 转换为PIL
    to_pil = transforms.ToPILImage()
    return to_pil(tensor.squeeze(0).cpu())


def apply_residual_boost(input_tensor, output_tensor, boost=1.0):
    """
    放大模型相对输入的校正量（免重训）：out = clamp(in + boost * (model_out - in))。
    boost>1 时去模糊/细节更明显，过大易产生振铃与过冲。
    """
    if boost is None or abs(float(boost) - 1.0) < 1e-6:
        return output_tensor
    b = float(boost)
    return torch.clamp(input_tensor + b * (output_tensor - input_tensor), 0.0, 1.0)


def enhance_pil_after_model(
    pil_rgb,
    unsharp_radius=0.0,
    unsharp_percent=0,
    unsharp_threshold=0,
    color_saturation=1.0,
):
    """
    推理后 PIL 域增强（免重训）：
    - UnsharpMask：提升边缘/纹理观感（细节更明显）
    - color_saturation：轻微>1 可使色彩更“实”，对去模糊观感有时有帮助

    unsharp_percent==0 时不做锐化。
    """
    out = pil_rgb
    if unsharp_percent and int(unsharp_percent) > 0:
        r = max(0.1, float(unsharp_radius))
        out = out.filter(
            ImageFilter.UnsharpMask(
                radius=r,
                percent=int(unsharp_percent),
                threshold=int(unsharp_threshold),
            )
        )
    if color_saturation is not None and abs(float(color_saturation) - 1.0) > 1e-6:
        out = ImageEnhance.Color(out).enhance(float(color_saturation))
    return out


# ==============================================================
#  大图切片处理（避免显存溢出）
# ==============================================================

def split_image_into_tiles(pil_img, tile_size=512, overlap=32):
    """
    将大图切分成小块
    
    Args:
        pil_img: PIL Image
        tile_size: 切片大小
        overlap: 重叠区域大小（用于避免边界伪影）
    
    Returns:
        list: [(tile_pil, (x, y)), ...]
    """
    width, height = pil_img.size
    tiles = []
    
    y = 0
    while y < height:
        x = 0
        while x < width:
            # 计算实际切片区域（考虑重叠）
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)
            
            # 提取切片
            tile = pil_img.crop((x, y, x_end, y_end))
            tiles.append((tile, (x, y)))
            
            # 移动位置（考虑重叠）
            if x_end == width:
                break
            x += tile_size - overlap
        
        if y_end == height:
            break
        y += tile_size - overlap
    
    return tiles


def stitch_tiles_into_image(tiles_with_pos, original_size):
    """
    将切片拼接回原图
    
    Args:
        tiles_with_pos: [(tile_pil, (x, y)), ...]
        original_size: (width, height)
    
    Returns:
        PIL.Image
    """
    width, height = original_size
    result = Image.new('RGB', (width, height))
    
    for tile, (x, y) in tiles_with_pos:
        result.paste(tile, (x, y))
    
    return result


def process_image_tiled(
    net,
    pil_img,
    tile_size=512,
    overlap=32,
    device='cuda',
    use_amp=False,
    show_progress=True,
    residual_boost=1.0,
    unsharp_radius=0.0,
    unsharp_percent=0,
    unsharp_threshold=0,
    color_saturation=1.0,
):
    """
    使用切片方式处理大图
    
    Args:
        net: 模型
        pil_img: PIL Image
        tile_size: 切片大小
        overlap: 重叠区域
        device: 设备
        use_amp: 是否使用混合精度
        residual_boost: 放大模型校正量（>1 更锐，易振铃）
        unsharp_* / color_saturation: 见 enhance_pil_after_model
    
    Returns:
        PIL.Image: 恢复后的图像
    """
    width, height = pil_img.size
    
    # 如果图片不大，直接处理
    if width <= tile_size and height <= tile_size:
        input_tensor = preprocess_image(pil_img, device)
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast('cuda'):
                    output_tensor = net(input_tensor)
            else:
                output_tensor = net(input_tensor)
        output_tensor = apply_residual_boost(input_tensor, output_tensor, residual_boost)
        pil_out = postprocess_image(output_tensor)
        return enhance_pil_after_model(
            pil_out,
            unsharp_radius=unsharp_radius,
            unsharp_percent=unsharp_percent,
            unsharp_threshold=unsharp_threshold,
            color_saturation=color_saturation,
        )
    
    # 大图切片处理
    tiles = split_image_into_tiles(pil_img, tile_size, overlap)
    processed_tiles = []
    
    net.eval()
    with torch.no_grad():
        tile_iter = tqdm(tiles, desc="处理切片", leave=False) if show_progress else tiles
        for tile, (x, y) in tile_iter:
            input_tensor = preprocess_image(tile, device)
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    output_tensor = net(input_tensor)
            else:
                output_tensor = net(input_tensor)
            output_tensor = apply_residual_boost(input_tensor, output_tensor, residual_boost)
            processed_tile = postprocess_image(output_tensor)
            processed_tile = enhance_pil_after_model(
                processed_tile,
                unsharp_radius=unsharp_radius,
                unsharp_percent=unsharp_percent,
                unsharp_threshold=unsharp_threshold,
                color_saturation=color_saturation,
            )
            processed_tiles.append((processed_tile, (x, y)))
    
    # 拼接结果
    result = stitch_tiles_into_image(processed_tiles, (width, height))
    return result


# ==============================================================
#  加载模型
# ==============================================================

def detect_model_config_from_checkpoint(checkpoint_path):
    """
    从checkpoint中尝试检测模型配置
    
    Args:
        checkpoint_path: checkpoint路径
    
    Returns:
        dict: 检测到的配置，如果无法检测则返回None
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        # 优先检查是否有保存的配置信息
        if 'model_config' in ckpt:
            print("  [自动检测] 从checkpoint中读取模型配置")
            return ckpt['model_config']
        
        # 尝试从权重键名推断blocks_per_rstb
        state_dict = ckpt.get('ema_state_dict', ckpt.get('model_state_dict', ckpt))
        
        # 查找body.0.blocks.X的最大索引
        max_block_idx = -1
        for key in state_dict.keys():
            if 'body.0.blocks.' in key:
                # 提取block索引，例如 body.0.blocks.6.norm1.weight -> 6
                parts = key.split('.')
                if len(parts) >= 3 and parts[0] == 'body' and parts[2] == 'blocks':
                    try:
                        block_idx = int(parts[3])
                        max_block_idx = max(max_block_idx, block_idx)
                    except (ValueError, IndexError):
                        continue
        
        if max_block_idx >= 0:
            detected_blocks = max_block_idx + 1  # 索引从0开始，所以+1
            print(f"  [自动检测] 从权重键名推断 blocks_per_rstb={detected_blocks}")
            return {'blocks_per_rstb': detected_blocks}
        
    except Exception as e:
        print(f"  [警告] 无法自动检测配置: {e}")
    
    return None


def load_model(checkpoint_path, device, model_config):
    """
    加载模型权重
    
    Args:
        checkpoint_path: checkpoint路径
        device: 设备
        model_config: 模型配置字典
    
    Returns:
        model: 加载好权重的模型
    """
    print(f"===> 正在加载模型权重: {checkpoint_path}")
    
    # 尝试从checkpoint自动检测配置
    detected_config = detect_model_config_from_checkpoint(checkpoint_path)
    if detected_config:
        # 更新配置（优先使用检测到的值）
        for key, value in detected_config.items():
            if key in model_config:
                old_value = model_config[key]
                model_config[key] = value
                if old_value != value:
                    print(f"  [配置更新] {key}: {old_value} -> {value}")
    
    # 创建模型（YCbCr版本，保持色彩不变）
    print("  [模型类型] YCbCr版本（只处理Y通道，Cb/Cr直接保留）")
    net = model.YCbCr_RestorationSwinNetV3(
        dim=model_config['dim'],
        num_rstb=model_config['num_rstb'],
        blocks_per_rstb=model_config['blocks_per_rstb'],
        num_heads=model_config['num_heads'],
        window_size=model_config['window_size'],
        use_dwconv=model_config.get('use_dwconv', True)
    ).to(device)
    
    # 加载权重
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到权重文件: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # 优先使用EMA权重（如果存在），否则使用普通权重
    if 'ema_state_dict' in ckpt:
        print("  [检测到EMA权重] 使用EMA权重进行推理（通常效果更好）")
        state_dict = ckpt['ema_state_dict']
    elif 'model_state_dict' in ckpt:
        print("  [使用普通权重] 使用model_state_dict")
        state_dict = ckpt['model_state_dict']
    else:
        print("  [使用直接权重] checkpoint中直接包含权重")
        state_dict = ckpt
    
    # 加载权重（允许部分匹配，兼容不同配置）
    missing_keys, unexpected_keys = net.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"  [警告] 以下权重未加载（{len(missing_keys)}个）: {missing_keys[:3]}...")
        print(f"  [提示] 这通常表示模型配置与训练时不一致，请检查 --blocks_per_rstb 等参数")
    if unexpected_keys:
        print(f"  [警告] 以下权重未使用（{len(unexpected_keys)}个）: {unexpected_keys[:3]}...")
        print(f"  [提示] 这通常表示模型配置与训练时不一致")
        if detected_config:
            print(f"  [建议] 请使用 --blocks_per_rstb {detected_config.get('blocks_per_rstb', '?')} 参数（如果已自动检测）")
    
    net.eval()
    print("  ✅ 模型加载成功")
    
    return net


# ==============================================================
#  计算评估指标
# ==============================================================

def compute_metrics(restored_path, gt_path, lpips_fn, niqe_fn, device):
    """
    计算PSNR、SSIM、LPIPS、NIQE指标
    
    Args:
        restored_path: 恢复图像路径
        gt_path: 真实图像路径
        lpips_fn: LPIPS模型
        niqe_fn: NIQE模型
        device: 设备
    
    Returns:
        dict: {'psnr': float, 'ssim': float, 'lpips': float, 'niqe': float}
    """
    try:
        restored_img = Image.open(restored_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')
        
        # 转换为tensor用于计算PSNR和SSIM
        restored_tensor = preprocess_image(restored_img, device)
        gt_tensor = preprocess_image(gt_img, device)
        
        # PSNR和SSIM（需要tensor格式 [B, C, H, W]）
        with torch.no_grad():
            psnr = utils_metrics.calculate_psnr(restored_tensor, gt_tensor).item()
            ssim = utils_metrics.calculate_ssim(restored_tensor, gt_tensor).item()
            
            # LPIPS
            lpips_score = lpips_fn(restored_tensor, gt_tensor).item()
        
        # NIQE（无参考指标，使用文件路径）
        niqe_score = niqe_fn(restored_path).item()
        
        return {
            'psnr': psnr,
            'ssim': ssim,
            'lpips': lpips_score,
            'niqe': niqe_score
        }
    except Exception as e:
        print(f"  [错误] 计算指标失败: {e}")
        return None


# ==============================================================
#  主推理函数
# ==============================================================

def main():
    parser = argparse.ArgumentParser(description='V2模型推理脚本')
    
    # 输入输出
    parser.add_argument('--input', type=str, required=True,
                        help='输入图片路径或文件夹路径')
    parser.add_argument('--output', type=str, default='./outputs',
                        help='输出文件夹路径')
    _default_ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'weights', 'dagf_swin_best.pth')
    parser.add_argument('--checkpoint', type=str, default=_default_ckpt,
                        help='模型权重路径（.pth）；见 weights/README.md')
    
    # 模型配置（需要与训练时一致）
    parser.add_argument('--model_dim', type=int, default=64,
                        help='模型维度（默认64）')
    parser.add_argument('--num_rstb', type=int, default=4,
                        help='RSTB块数量（默认4）')
    parser.add_argument('--blocks_per_rstb', type=int, default=6,
                        help='每个RSTB的块数量（默认7，需与训练时一致）')
    parser.add_argument('--model_heads', type=int, default=4,
                        help='注意力头数（默认4）')
    parser.add_argument('--model_window_size', type=int, default=8,
                        help='窗口大小（默认8）')
    parser.add_argument('--use_dwconv', action='store_true', default=True,
                        help='使用深度可分离卷积')
    
    # 推理选项
    parser.add_argument('--tile_size', type=int, default=0,
                        help='大图切片大小（默认0表示不切片）')
    parser.add_argument('--overlap', type=int, default=32,
                        help='切片重叠区域大小（默认32）')
    parser.add_argument('--use_amp', action='store_true',
                        help='使用混合精度推理（FP16，节省显存）')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备（cuda/cpu，默认cuda）')
    
    # 免重训：观感增强（残差放大 + PIL 反锐化）
    parser.add_argument('--residual_boost', type=float, default=1.0,
                        help='放大模型相对输入的校正量，默认1；约1.1~1.35 可更明显去模糊（易振铃）')
    parser.add_argument('--unsharp_radius', type=float, default=0.0,
                        help='反锐化半径（PIL UnsharpMask），0 表示关闭')
    parser.add_argument('--unsharp_percent', type=int, default=0,
                        help='反锐化强度 0~300+，0 关闭；约 80~180 较常用')
    parser.add_argument('--unsharp_threshold', type=int, default=0,
                        help='反锐化阈值（抑制平坦区噪声）')
    parser.add_argument('--color_saturation', type=float, default=1.0,
                        help='输出色彩饱和度，1.0 不变；略>1 更饱和')
    
    # 评估选项
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='真实图像文件夹路径（可选，用于计算指标）')
    parser.add_argument('--compute_metrics', action='store_true',
                        help='计算评估指标（需要提供--gt_dir）')
    
    args = parser.parse_args()
    
    # 设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("  [警告] CUDA不可用，使用CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    print(f"===> 使用设备: {device}")
    
    # 模型配置
    model_config = {
        'dim': args.model_dim,
        'num_rstb': args.num_rstb,
        'blocks_per_rstb': args.blocks_per_rstb,
        'num_heads': args.model_heads,
        'window_size': args.model_window_size,
        'use_dwconv': args.use_dwconv
    }
    
    # 加载模型
    net = load_model(args.checkpoint, device, model_config)
    
    # 评估指标模型（如果需要）
    lpips_fn = None
    niqe_fn = None
    if args.compute_metrics:
        print("===> 加载评估指标模型...")
        lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
        niqe_fn = pyiqa.create_metric('niqe', device=device)
        print("  ✅ 评估指标模型加载成功")
    
    # 创建输出文件夹
    os.makedirs(args.output, exist_ok=True)
    
    # 获取输入图片列表
    if os.path.isfile(args.input):
        img_paths = [args.input]
    elif os.path.isdir(args.input):
        img_paths = []
        # 只使用小写扩展名（Windows不区分大小写，会自动匹配）
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        
        for ext in extensions:
            pattern = os.path.join(args.input, ext)
            found_files = glob.glob(pattern)
            # 只保留文件，排除目录和隐藏文件
            img_paths.extend([
                f for f in found_files 
                if os.path.isfile(f) and not os.path.basename(f).startswith('.')
            ])
        
        # 去重：Windows系统不区分大小写
        if os.name == 'nt':
            # Windows: 使用normcase规范化后去重
            seen = set()
            unique_paths = []
            for p in img_paths:
                norm_p = os.path.normcase(p)
                if norm_p not in seen:
                    seen.add(norm_p)
                    unique_paths.append(p)
            img_paths = unique_paths
        else:
            img_paths = list(dict.fromkeys(img_paths))
        
        img_paths = sorted(img_paths)
    else:
        raise ValueError(f"输入路径不存在: {args.input}")
    
    if len(img_paths) == 0:
        raise ValueError(f"未找到图片文件: {args.input}")
    
    print(f"===> 找到 {len(img_paths)} 张图片")
    enhance_kw = dict(
        residual_boost=args.residual_boost,
        unsharp_radius=args.unsharp_radius,
        unsharp_percent=args.unsharp_percent,
        unsharp_threshold=args.unsharp_threshold,
        color_saturation=args.color_saturation,
    )
    if args.residual_boost != 1.0 or args.unsharp_percent > 0 or args.color_saturation != 1.0:
        print(f"===> 免重训增强: {enhance_kw}")
    
    # 推理循环
    all_metrics = {'psnr': [], 'ssim': [], 'lpips': [], 'niqe': []}
    total_time = 0
    
    print(f"\n===> 开始推理...")
    for idx, img_path in enumerate(tqdm(img_paths, desc="推理进度")):
        img_name = os.path.basename(img_path)
        output_path = os.path.join(args.output, img_name)
        
        try:
            start_time = time.time()
            
            # 读取图片
            pil_img = Image.open(img_path).convert('RGB')
            
            # 推理
            if args.tile_size > 0 and (pil_img.width > args.tile_size or pil_img.height > args.tile_size):
                # 大图切片处理
                restored_img = process_image_tiled(
                    net, pil_img, args.tile_size, args.overlap, device, args.use_amp,
                    **enhance_kw,
                )
            else:
                # 直接处理
                input_tensor = preprocess_image(pil_img, device)
                with torch.no_grad():
                    if args.use_amp:
                        with torch.amp.autocast('cuda'):
                            output_tensor = net(input_tensor)
                    else:
                        output_tensor = net(input_tensor)
                output_tensor = apply_residual_boost(input_tensor, output_tensor, args.residual_boost)
                restored_img = postprocess_image(output_tensor)
                restored_img = enhance_pil_after_model(
                    restored_img,
                    unsharp_radius=args.unsharp_radius,
                    unsharp_percent=args.unsharp_percent,
                    unsharp_threshold=args.unsharp_threshold,
                    color_saturation=args.color_saturation,
                )
            
            # 保存结果
            restored_img.save(output_path)
            
            processing_time = time.time() - start_time
            total_time += processing_time
            
            # 计算指标（如果提供GT）
            if args.compute_metrics and args.gt_dir:
                gt_path = os.path.join(args.gt_dir, img_name)
                if os.path.exists(gt_path):
                    metrics = compute_metrics(output_path, gt_path, lpips_fn, niqe_fn, device)
                    if metrics:
                        all_metrics['psnr'].append(metrics['psnr'])
                        all_metrics['ssim'].append(metrics['ssim'])
                        all_metrics['lpips'].append(metrics['lpips'])
                        all_metrics['niqe'].append(metrics['niqe'])
                        print(f"  [{idx+1}/{len(img_paths)}] {img_name} | "
                              f"PSNR: {metrics['psnr']:.2f}dB | "
                              f"SSIM: {metrics['ssim']:.4f} | "
                              f"LPIPS: {metrics['lpips']:.4f} | "
                              f"NIQE: {metrics['niqe']:.4f} | "
                              f"时间: {processing_time:.2f}s")
                    else:
                        print(f"  [{idx+1}/{len(img_paths)}] {img_name} | 时间: {processing_time:.2f}s")
                else:
                    print(f"  [{idx+1}/{len(img_paths)}] {img_name} | 时间: {processing_time:.2f}s")
            else:
                print(f"  [{idx+1}/{len(img_paths)}] {img_name} | 时间: {processing_time:.2f}s")
        
        except Exception as e:
            print(f"  [错误] 处理 {img_name} 失败: {e}")
            continue
    
    # 打印统计信息
    print(f"\n===> 推理完成！")
    print(f"  总耗时: {total_time:.2f}s")
    print(f"  平均耗时: {total_time/len(img_paths):.2f}s/张")
    print(f"  结果保存在: {args.output}")
    
    # 打印平均指标
    if args.compute_metrics and len(all_metrics['psnr']) > 0:
        print(f"\n===> 平均评估指标:")
        print(f"  PSNR: {np.mean(all_metrics['psnr']):.2f}dB")
        print(f"  SSIM: {np.mean(all_metrics['ssim']):.4f}")
        print(f"  LPIPS: {np.mean(all_metrics['lpips']):.4f}")
        print(f"  NIQE: {np.mean(all_metrics['niqe']):.4f}")


if __name__ == '__main__':
    main()

