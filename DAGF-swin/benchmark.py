# -*- coding: utf-8 -*-
"""
DAGF-Swin 性能测试：参数量、FLOPs/MACs、推理耗时、FPS、ms/Mpixel、峰值显存。

用法:
  python benchmark.py
  python benchmark.py --H 512 --W 512 --dim 64 --model ycbcr
  python benchmark.py --device cuda --fp16
  python benchmark.py --H 1080 --W 1920 --warmup 10 --repeats 100

依赖:
  - FLOPs: pip install thop   （未安装时仅打印参数量与显存）
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model_v3 as model


def count_parameters(m: nn.Module):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


def format_large(n: float) -> str:
    if n >= 1e9:
        return f"{n/1e9:.4f} G"
    if n >= 1e6:
        return f"{n/1e6:.4f} M"
    if n >= 1e3:
        return f"{n/1e3:.4f} K"
    return f"{n:.2f}"


def padded_resolution(h: int, w: int, window_size: int) -> tuple[int, int]:
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    return h + pad_h, w + pad_w


def profile_flops_thop(m: nn.Module, x: torch.Tensor):
    """thop 统计的是乘加次数（MACs），与论文中常写的 FLOPs（一次乘加常算 2 FLOPs）可能差约 2 倍。"""
    from thop import profile

    m = m.eval()
    with torch.no_grad():
        macs, _params = profile(m, inputs=(x,), verbose=False)
    return float(macs), float(_params)


def benchmark_latency(
    run_forward,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> list[float]:
    """返回每次 forward 的耗时列表（毫秒）。"""
    for _ in range(max(0, warmup)):
        run_forward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    times_ms: list[float] = []
    for _ in range(max(1, repeats)):
        if device.type == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            run_forward()
            ender.record()
            torch.cuda.synchronize(device)
            times_ms.append(starter.elapsed_time(ender))
        else:
            t0 = time.perf_counter()
            run_forward()
            times_ms.append((time.perf_counter() - t0) * 1000.0)
    return times_ms


def benchmark_peak_memory(
    run_forward,
    device: torch.device,
    warmup: int,
    mem_repeats: int,
) -> float:
    """峰值显存（MiB），多次取 max。"""
    torch.cuda.empty_cache()
    for _ in range(max(0, warmup)):
        run_forward()
        torch.cuda.synchronize(device)

    peak_mb = 0.0
    for _ in range(max(1, mem_repeats)):
        torch.cuda.reset_peak_memory_stats(device)
        run_forward()
        torch.cuda.synchronize(device)
        cur = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_mb = max(peak_mb, cur)
    return peak_mb


def print_latency_report(
    times_ms: list[float],
    h: int,
    w: int,
    hp: int,
    wp: int,
    warmup: int,
    repeats: int,
    device: torch.device,
    dtype_label: str,
):
    mean_ms = statistics.mean(times_ms)
    std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    min_ms = min(times_ms)
    max_ms = max(times_ms)
    mp_nominal = h * w / 1e6
    mp_padded = hp * wp / 1e6
    fps = 1000.0 / mean_ms if mean_ms > 0 else float("inf")
    ms_per_mp_nominal = mean_ms / mp_nominal if mp_nominal > 0 else float("inf")
    ms_per_mp_padded = mean_ms / mp_padded if mp_padded > 0 else float("inf")

    timer = "CUDA Event" if device.type == "cuda" else "perf_counter"
    print(f"  device:              {device}")
    if device.type == "cuda":
        print(f"  GPU:                 {torch.cuda.get_device_name(device)}")
    print(f"  dtype:               {dtype_label}")
    print(f"  batch:               1")
    print(f"  input (nominal):     {h} x {w}  ({mp_nominal:.4f} Mpix)")
    print(f"  input (padded):      {hp} x {wp}  ({mp_padded:.4f} Mpix, window 对齐后)")
    print(f"  warmup:              {warmup}")
    print(f"  timed repeats:       {repeats}")
    print(f"  timer:               {timer}")
    print(f"  latency mean ± std:  {mean_ms:.3f} ± {std_ms:.3f} ms")
    print(f"  latency min / max:   {min_ms:.3f} / {max_ms:.3f} ms")
    print(f"  FPS (1/latency):     {fps:.2f}")
    print(f"  ms/Mpix (nominal):   {ms_per_mp_nominal:.4f}")
    print(f"  ms/Mpix (padded):    {ms_per_mp_padded:.4f}")
    print("  说明: 仅 net.forward，不含读图/存盘；CUDA 使用 Event 计时并 synchronize。")


def main():
    p = argparse.ArgumentParser(description="DAGF-Swin 参数量 / FLOPs / 推理耗时 / 峰值显存")
    p.add_argument("--model", choices=("ycbcr", "rgb"), default="ycbcr", help="与推理一致默认 ycbcr（core 单通道）")
    p.add_argument("--H", type=int, default=512)
    p.add_argument("--W", type=int, default=512)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--num_rstb", type=int, default=4)
    p.add_argument("--blocks_per_rstb", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--window_size", type=int, default=8)
    p.add_argument("--no_dwconv", action="store_true", help="关闭深度可分离卷积（默认开启）")
    p.add_argument("--device", type=str, default="cuda", help="cuda 或 cpu")
    p.add_argument("--fp16", action="store_true", help="CUDA 下用 autocast（与 AMP 推理接近）")
    p.add_argument("--warmup", type=int, default=10, help="正式测速/显存前的 warmup 次数")
    p.add_argument("--repeats", type=int, default=100, help="测速重复次数，报告 mean±std")
    p.add_argument("--mem_repeats", type=int, default=3, help="测峰值显存重复次数，取 max")
    args = p.parse_args()

    cfg = dict(
        dim=args.dim,
        num_rstb=args.num_rstb,
        blocks_per_rstb=args.blocks_per_rstb,
        num_heads=args.num_heads,
        window_size=args.window_size,
        use_dwconv=not args.no_dwconv,
    )

    if args.model == "ycbcr":
        net = model.YCbCr_RestorationSwinNetV2(**cfg)
        dummy = torch.rand(1, 3, args.H, args.W)
    else:
        net = model.RestorationSwinNetV2(in_channels=3, **cfg)
        dummy = torch.rand(1, 3, args.H, args.W)

    hp, wp = padded_resolution(args.H, args.W, args.window_size)
    dtype_label = "fp16 autocast" if args.fp16 else "fp32"

    total, trainable = count_parameters(net)
    print("=== 参数量 ===")
    print(f"  Total params:   {total:,}  ({format_large(total)})")
    print(f"  Trainable:      {trainable:,}")
    print(f"  Model:          {args.model}, HxW={args.H}x{args.W}, cfg={cfg}")

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("  [警告] CUDA 不可用，改用 CPU")

    net = net.to(device)
    dummy = dummy.to(device)
    net.eval()

    def run_forward():
        if args.fp16 and device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                with torch.no_grad():
                    return net(dummy)
        with torch.no_grad():
            return net(dummy)

    # ---------- FLOPs / MACs ----------
    print("\n=== 计算量（依赖 thop）===")
    try:
        macs, thop_params = profile_flops_thop(net, dummy)
        print(f"  MACs (thop):    {macs:,.0f}  (~{format_large(macs)})")
        print(f"  约 FLOPs(×2):   {2*macs:,.0f}  （若按「1 次乘加 = 2 FLOPs」换算，仅供参考）")
        if abs(thop_params - total) > 1:
            print(f"  [提示] thop 参数量 {int(thop_params):,} 与直接统计略有差异属常见现象")
    except ImportError:
        print("  未安装 thop，跳过。安装: pip install thop")
    except Exception as e:
        print(f"  thop 统计失败: {e}")

    # ---------- 推理耗时 / FPS / ms/Mpix ----------
    print("\n=== 推理耗时 / FPS / ms/Mpixel ===")
    times_ms = benchmark_latency(run_forward, device, args.warmup, args.repeats)
    print_latency_report(
        times_ms, args.H, args.W, hp, wp,
        args.warmup, args.repeats, device, dtype_label,
    )

    # ---------- 峰值显存 ----------
    print("\n=== 峰值显存（仅 CUDA）===")
    if device.type != "cuda":
        print("  非 CUDA 设备，跳过。")
        return

    peak_mb = benchmark_peak_memory(run_forward, device, args.warmup, args.mem_repeats)
    print(f"  device:         {torch.cuda.get_device_name(device)}")
    print(f"  dtype:          {dtype_label}")
    print(f"  warmup:         {args.warmup}")
    print(f"  mem_repeats:    {args.mem_repeats}  (取峰值最大)")
    print(f"  peak allocated: {peak_mb:.2f} MiB  (torch.cuda.max_memory_allocated)")
    print("  说明: 不含 DataLoader/其它张量；与 batch、驱动、缓存有关，略低于 nvidia-smi 进程占用属正常。")


if __name__ == "__main__":
    main()
