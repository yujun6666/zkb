# depart/model.py
import torch
import torch.nn as nn
import math
import warnings
from drop import DropPath  # 调用学长的 DropPath

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # 学长的截断正态初始化黑科技
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. ")
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.)).add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, dim_in=1, dim_feat=256, patch_size=5, t_patch_size=24):
        super().__init__()
        self.proj = nn.Conv2d(dim_in, dim_feat, kernel_size=[t_patch_size, patch_size], stride=[t_patch_size, patch_size])
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class IndustryMAE(nn.Module):
    def __init__(self, num_signal=65, num_frames=360, patch_size=5, t_patch_size=24, dim_feat=256, depth=8, num_heads=8):
        super().__init__()
        self.patch_embed = PatchEmbed(dim_in=1, dim_feat=dim_feat, patch_size=patch_size, t_patch_size=t_patch_size)
        num_patches = (num_signal // patch_size) * (num_frames // t_patch_size)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim_feat))
        trunc_normal_(self.pos_embed, std=.02) # 学长级初始化
        
        # 引入学长的 Stochastic Depth 衰减率
        dpr = [x.item() for x in torch.linspace(0, 0.1, depth)] 
        self.blocks = nn.ModuleList([
            Block(dim=dim_feat, num_heads=num_heads, drop_path=dpr[i]) for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim_feat)
        
        # 针对 2026 赛题：预测未来 24 步的 3 个信号
        self.predict_head = nn.Linear(num_patches * dim_feat, 24 * 3)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = x.unsqueeze(1) # [B, 1, T, V]
        x = self.patch_embed(x) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x).reshape(B, -1)
        return self.predict_head(x).reshape(B, 24, 3)