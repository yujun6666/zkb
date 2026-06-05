# model.py
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features or in_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features or in_features, out_features or in_features)
    def forward(self, x): 
        return self.fc2(self.act(self.fc1(x)))

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class Block(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=dim * 4)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class SkeleEmbed(nn.Module):
    def __init__(self, dim_in=1, dim_feat=256, patch_size=5, t_patch_size=24):
        super().__init__()
        self.proj = nn.Conv2d(dim_in, dim_feat, kernel_size=[t_patch_size, patch_size], stride=[t_patch_size, patch_size])
    def forward(self, x):
        x = x.permute(0, 3, 1, 2) 
        x = self.proj(x) 
        return x.flatten(2).transpose(1, 2)

class IndustryMAE(nn.Module):
    def __init__(self, num_signal=65, num_frames=360, patch_size=5, t_patch_size=24, dim_feat=256):
        super().__init__()
        self.signal_embed = SkeleEmbed(dim_in=1, dim_feat=dim_feat, patch_size=patch_size, t_patch_size=t_patch_size)
        num_patches = (num_signal // patch_size) * (num_frames // t_patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim_feat))
        self.blocks = nn.ModuleList([Block(dim=dim_feat) for _ in range(4)])
        self.norm = nn.LayerNorm(dim_feat)
        self.predict_head = nn.Linear(num_patches * dim_feat, 24 * 3)

    def forward(self, x):
        B, T, V = x.shape
        x_in = x.unsqueeze(-1) 
        x_patches = self.signal_embed(x_in) + self.pos_embed
        for blk in self.blocks: 
            x_patches = blk(x_patches)
        feat = self.norm(x_patches) 
        feat_flat = feat.reshape(B, -1)
        pred = self.predict_head(feat_flat)
        return pred.reshape(B, 24, 3)