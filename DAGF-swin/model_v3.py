# -*- coding: utf-8 -*-
"""
模型V3版本 - 在 V2 骨架上强化高频/细节重建
相对 V2 的改动：
1. DMSA_Fusion：门控输出 + 可学习浅层注入，减轻乘性注意力对纹理的压制
2. 浅层：双卷积 + GELU，加强边缘与细纹理表征
3. conv_after_body：全卷积 3x3（替代深度可分离），增强通道混合
4. tail：双层全卷积 + GELU（use_dwconv=True 时），提升残差高频重建能力
其余（Swin/RSTB/窗口注意力等）与 V2 一致。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_


# ==============================================================================
#  0. 深度可分离卷积（优化版）
# ==============================================================================

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积，参数量约减少8-9倍"""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1, bias=True):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size, 
            padding=padding, stride=stride, 
            groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
        
    def forward(self, x):
        return self.pointwise(self.depthwise(x))


# ==============================================================================
#  0.5. 注意力模块（集成DMSA_Fusion）
# ==============================================================================

class ChannelAttention(nn.Module):
    """通道注意力模块（SENet风格）"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DMSA_Fusion(nn.Module):
    """
    双注意力多尺度融合模块（Dual Multi-Scale Attention Fusion）
    结合通道注意力和空间注意力，效果很好，已被集成
    
    功能：
    - 融合不同层级的特征（如浅层和深层）
    - 使用通道注意力，关注重要通道
    - 使用空间注意力，关注重要空间位置
    - 浅层可学习注入：减轻 fused_ca*fused_sa 对高频的压制，利于胡须/发丝等细节
    """
    def __init__(self, dim):
        super().__init__()
        self.conv1x1 = nn.Conv2d(dim * 2, dim, 1)
        self.channel_att = ChannelAttention(dim)
        # 简化空间注意力，使用更高效
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        # 显式拉回浅层结构信息（纹理/边缘），训练时可自适应强度
        self.shallow_inject = nn.Parameter(torch.tensor(0.22))

    def forward(self, x_low, x_high, return_intermediates=False):
        """
        融合两个特征图
        
        Args:
            x_low: 低层特征 (B, C, H, W) - 通常指浅层特征
            x_high: 高层特征 (B, C, H, W) - 通常指深层特征
            return_intermediates: 为 True 时额外返回可视化用中间量
        
        Returns:
            fused: 融合后特征 (B, C, H, W)；若 return_intermediates 则为 (fused, dict)
        """
        # 1. 拼接两个特征
        fused = torch.cat([x_low, x_high], dim=1)  # (B, 2C, H, W)
        fused = self.conv1x1(fused)  # (B, C, H, W)

        # 2. 通道注意力，关注重要通道
        fused_ca = self.channel_att(fused)

        # 3. 空间注意力，关注重要空间位置（使用fused的副本，避免inplace操作）
        fused_for_spatial = fused  # 使用副本
        avg_out = torch.mean(fused_for_spatial, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(fused_for_spatial, dim=1, keepdim=True)  # (B, 1, H, W)
        sa_feature = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        fused_sa = self.spatial_att(sa_feature)  # (B, 1, H, W)

        # 4. 结合通道和空间注意力，并注入浅层（避免纯乘积过度削弱高频）
        attn_out = fused_ca * fused_sa
        out = attn_out + self.shallow_inject * x_low
        if return_intermediates:
            inter = {
                "fused_1x1": fused,
                "after_channel_att": fused_ca,
                "spatial_att_map": fused_sa,
                "attn_product": attn_out,
                "shallow_inject": self.shallow_inject * x_low,
            }
            return out, inter
        return out

# ==============================================================================
#  1. 窗口分割 & Mask 缓存（优化避免重复计算）
# ==============================================================================

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class MaskCache:
    """Mask缓存类，避免重复计算"""
    def __init__(self):
        self.cache = {}
    
    def get_mask(self, H, W, window_size, shift_size, device):
        key = (H, W, window_size, shift_size, device)
        if key not in self.cache:
            img_mask = torch.zeros((1, H, W, 1), device=device)
            h_slices = (slice(0, -window_size),
                        slice(-window_size, -shift_size),
                        slice(-shift_size, None))
            w_slices = (slice(0, -window_size),
                        slice(-window_size, -shift_size),
                        slice(-shift_size, None))
            
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            
            mask_windows = window_partition(img_mask, window_size)
            mask_windows = mask_windows.view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.cache[key] = attn_mask
        
        return self.cache[key]


# ==============================================================================
#  2. Window Attention（优化版：减少计算）
# ==============================================================================

class WindowAttention(nn.Module):
    """优化的窗口注意力，减少不必要的计算"""
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        # �7�3 优化：不使用einops，直接reshape，减少中间计算
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


# ==============================================================================
#  3. 优化的Gated Feed-Forward Network
# ==============================================================================

class OptimizedGatedFeedForward(nn.Module):
    """
    优化的门控前馈网络：
    1. 减少reshape操作
    2. 使用更高效的实现
    """
    def __init__(self, dim, mult=4):
        super().__init__()
        hidden_features = int(dim * mult)
        self.project_in = nn.Linear(dim, hidden_features * 2)
        # �7�3 优化：使用深度卷积，groups=hidden_features*2
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, 3, 1, 1, groups=hidden_features * 2)
        self.project_out = nn.Linear(hidden_features, dim)
        self.act = nn.GELU()  # �7�3 使用GELU激活函数

    def forward(self, x, resolution=None):
        B, L, C = x.shape
        
        # 如果提供了resolution，使用它；否则尝试从L推断
        if resolution is not None:
            H, W = resolution
        else:
            # 尝试推断H和W（假设是正方形）
            H = int(L ** 0.5)
            W = H
            # 如果推断失败（L不是完全平方数），尝试其他方法
            if H * W != L:
                # 尝试常见的宽高比
                for h in range(int(L ** 0.5), 0, -1):
                    if L % h == 0:
                        H, W = h, L // h
                        break
        
        # �7�3 优化：减少一次permute
        x = self.project_in(x)  # [B, L, hidden*2]
        x = x.transpose(1, 2).view(B, -1, H, W)  # [B, hidden*2, H, W]
        x = self.dwconv(x)  # 深度卷积
        x = x.view(B, -1, L).transpose(1, 2)  # [B, L, hidden*2]
        
        x1, x2 = x.chunk(2, dim=-1)
        x = self.act(x1) * x2  # �7�3 使用GELU
        x = self.project_out(x)
        return x


# ==============================================================================
#  4. Swin Transformer Block（优化版）
# ==============================================================================

class OptimizedSwinBlock(nn.Module):
    """优化的Swin Block，减少不必要的操作"""
    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, to_2tuple(window_size), num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = OptimizedGatedFeedForward(dim, mult=mlp_ratio)

    def forward(self, x, resolution, mask=None):
        H, W = resolution
        B, L, C = x.shape
        shortcut = x
        
        # �7�3 优化：减少norm和view，只做一次view
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=mask)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, L, C)
        x = shortcut + x  # �7�3 第一次残差连接
        x = x + self.mlp(self.norm2(x), resolution)  # �7�3 第二次残差连接
        return x


# ==============================================================================
#  5. RSTB: Residual Swin Transformer Block（与 V2 相同结构，配合 V3 版 DMSA）
# ==============================================================================

class OptimizedRSTB(nn.Module):
    """
    优化的RSTB：
    1. 使用缓存的mask
    2. 优化的卷积层
    3. 集成的特征融合
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=2., use_dwconv=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = window_size // 2
        
        self.blocks = nn.ModuleList([
            OptimizedSwinBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio
            )
            for i in range(depth)
        ])
        
        # �7�3 优化：使用深度可分离卷积
        if use_dwconv:
            self.conv = DepthwiseSeparableConv(dim, dim, kernel_size=3, padding=1)
        else:
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        
        # �7�3 集成：使用DMSA_Fusion进行特征融合（比简单的1x1卷积）
        self.fusion = DMSA_Fusion(dim)
        
        # �7�3 优化：Mask缓存
        self.mask_cache = MaskCache()

    def forward(self, x, resolution):
        H, W = resolution
        shortcut = x
        
        # �7�3 优化：使用缓存的mask（仅当shift_size>0时）
        attn_mask = None
        if self.shift_size > 0:
            attn_mask = self.mask_cache.get_mask(H, W, self.window_size, self.shift_size, x.device)

        for blk in self.blocks:
            x = blk(x, resolution, mask=attn_mask if blk.shift_size > 0 else None)

        # �7�3 优化：减少一次transpose
        x = x.transpose(1, 2).view(x.shape[0], -1, H, W)
        x_conv = self.conv(x)
        
        # �7�3 使用DMSA_Fusion融合原始特征和卷积特征
        shortcut_2d = shortcut.transpose(1, 2).view(x.shape[0], -1, H, W)
        x = self.fusion(shortcut_2d, x_conv)
        x = x.flatten(2).transpose(1, 2)
        return x + shortcut


# ==============================================================================
#  6. 主恢复网络（V3：细节增强版）
# ==============================================================================

class RestorationSwinNetV3(nn.Module):
    """
    V3 图像恢复网络：在 V2 基础上强化细纹理与高频残差重建（见文件头说明）。
    """
    def __init__(self, in_channels=3, dim=64, num_rstb=4, blocks_per_rstb=6, 
                 num_heads=4, window_size=8, use_dwconv=True):
        super().__init__()
        self.window_size = window_size
        self.use_dwconv = use_dwconv
        
        # �7�3 优化：head使用深度可分离卷积
        if use_dwconv:
            self.head = DepthwiseSeparableConv(in_channels, dim, kernel_size=3, padding=1)
        else:
            self.head = nn.Conv2d(in_channels, dim, 3, 1, 1)
        
        # 浅层多一层卷积 + 非线性，强化纹理/边缘表征（利于胡须等细节）
        self.shallow_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )

        self.body = nn.ModuleList([
            OptimizedRSTB(dim, (None, None), blocks_per_rstb, num_heads, window_size, use_dwconv=use_dwconv)
            for _ in range(num_rstb)
        ])

        # Body 后使用全卷积混通道，比深度可分离更有利于恢复细纹理
        self.conv_after_body = nn.Conv2d(dim, dim, 3, 1, 1)
        
        # �7�3 集成：使用DMSA_Fusion进行浅层+深层特征融合（比简单的concat+conv）
        self.feature_fusion = DMSA_Fusion(dim)
            
        # 双层全卷积输出头：比单深度可分离尾更强的高频重建能力
        if use_dwconv:
            self.tail = nn.Sequential(
                nn.Conv2d(dim, dim, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(dim, in_channels, 3, 1, 1),
            )
        else:
            self.tail = nn.Conv2d(dim, in_channels, 3, 1, 1)
            
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: 
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, DepthwiseSeparableConv)):
            if hasattr(m, 'pointwise'):
                # 深度可分离卷积
                nn.init.kaiming_normal_(m.depthwise.weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(m.pointwise.weight, mode='fan_out', nonlinearity='relu')
            else:
                # 普通卷积
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_image_size(self, x):
        """确保图像尺寸是 window_size 的倍数，并返回填充后的空间尺寸供 Swin 使用。"""
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, hp, wp = x.size()
        # 必须返回 (hp, wp)：RSTB/window_partition 需要与 x 张量空间维一致；若误用原始 h,w 会在非整除时 view 报错
        return x, hp, wp

    def forward(self, x, return_fusion_vis=False):
        B, C, H_orig, W_orig = x.shape
        x_padded, H, W = self.check_image_size(x)

        # 优化：提取浅层特征（显存优化：只在需要时保存）
        x_feat = self.head(x_padded)
        shallow_feat = self.shallow_conv(x_feat)  # 浅层特征
        shortcut = x_feat
        
        # 转换为序列格式进行Transformer处理
        x_feat_seq = x_feat.flatten(2).transpose(1, 2)  # [B, H*W, dim]

        # 优化：RSTB处理使用缓存的mask
        for rstb in self.body:
            x_feat_seq = rstb(x_feat_seq, (H, W))

        # 转换回空间格式
        x_feat = x_feat_seq.transpose(1, 2).view(B, -1, H, W)  # [B, dim, H, W]
        x_feat = self.conv_after_body(x_feat)
        deep_feat = x_feat

        if return_fusion_vis:
            fused_dmsa, fusion_inter = self.feature_fusion(
                shallow_feat, deep_feat, return_intermediates=True
            )
        else:
            fused_dmsa = self.feature_fusion(shallow_feat, deep_feat)

        x_feat = fused_dmsa + shortcut  # 残差连接
        
        out = self.tail(x_feat)
        out = out[:, :, :H_orig, :W_orig]
        # 修复：使用x_padded而不是x，避免inplace操作问题
        x_orig = x_padded[:, :, :H_orig, :W_orig]
        restored = torch.clamp(x_orig + out, 0, 1)
        if return_fusion_vis:
            vis = {
                "shallow": shallow_feat,
                "deep": deep_feat,
                "fused_dmsa": fused_dmsa,
                "fusion_inter": fusion_inter,
                "pre_tail": x_feat,
            }
            return restored, vis
        return restored


# ==============================================================================
#  YCbCr版本封装（保持色彩不变）
# ==============================================================================

class YCbCr_RestorationSwinNetV3(nn.Module):
    """
    YCbCr 版 V3：只处理 Y 通道，Cb/Cr 保留，色彩与 V2 YCbCr 策略一致。
    """
    def __init__(self, dim=64, num_rstb=4, blocks_per_rstb=6, 
                 num_heads=4, window_size=8, use_dwconv=True):
        super().__init__()
        self.core = RestorationSwinNetV3(
            in_channels=1,  # 改为1通道
            dim=dim,
            num_rstb=num_rstb,
            blocks_per_rstb=blocks_per_rstb,
            num_heads=num_heads,
            window_size=window_size,
            use_dwconv=use_dwconv
        )
    
    def rgb_to_ycbcr(self, x):
        """RGB转YCbCr（标准BT.601）"""
        r, g, b = x.chunk(3, dim=1)
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b
        return y, cb, cr
    
    def ycbcr_to_rgb(self, y, cb, cr):
        """YCbCr转RGB（标准BT.601）"""
        r = y + 1.402 * cr
        g = y - 0.344136 * cb - 0.714136 * cr
        b = y + 1.772 * cb
        rgb = torch.cat([r, g, b], dim=1)
        return torch.clamp(rgb, 0.0, 1.0)
    
    def forward(self, x_rgb, return_fusion_vis=False):
        """
        前向传播：只处理Y通道，Cb/Cr直接保留
        
        Args:
            x_rgb: RGB输入图像 (B, 3, H, W)
            return_fusion_vis: 为 True 时由 core 返回融合过程特征（基于 Y 通道特征图）
        
        Returns:
            restored_rgb: 恢复后的RGB图像 (B, 3, H, W)，色彩与输入一致；
            return_fusion_vis 时为 (restored_rgb, vis_dict)
        """
        # 1. RGB转YCbCr
        y, cb, cr = self.rgb_to_ycbcr(x_rgb)
        
        # 2. 只处理Y通道（亮度恢复）
        if return_fusion_vis:
            y_restored, vis = self.core(y, return_fusion_vis=True)
        else:
            y_restored = self.core(y)
            vis = None
        
        # 3. YCbCr转回RGB（使用原始Cb/Cr，保持色彩不变）
        rgb = self.ycbcr_to_rgb(y_restored, cb, cr)
        if return_fusion_vis:
            return rgb, vis
        return rgb


# 别名：与 V3 为同一类，兼容仍写 V2 类名的脚本（权重须与 V3 结构一致）
YCbCr_RestorationSwinNetV2 = YCbCr_RestorationSwinNetV3
RestorationSwinNetV2 = RestorationSwinNetV3
