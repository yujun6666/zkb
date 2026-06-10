import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader

print("🔥 V2 启动：Xicor 非线性注意力引擎 + 高级清洗 + 早停...")

def advanced_clean_data(raw_data):
    df = pd.DataFrame(raw_data)
    df.replace([np.inf, -np.inf, 0.0], np.nan, inplace=True)
    df = df.interpolate(method='linear', limit_direction='both')
    return df.bfill().ffill().fillna(0.0).values

# ==================== [2] 前沿突破：Xicor Attention ====================
class XicorAttention(nn.Module):
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
        
        # 💡 核心机制：结合 Chatterjee Xicor 的非线性特征空间
        # 通过对 Q 和 K 进行 L2 归一化，并计算非线性距离惩罚矩阵，拟合秩相关性
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        
        # 标准线性点乘注意力
        linear_attn = q @ k.transpose(-2, -1)
        
        # Xicor 非线性距离惩罚项 (L1 Distance Proxy)
        nonlinear_dist = torch.cdist(q_norm, k_norm, p=1)
        
        # 融合非线性关系，计算最终的 Attention Weights
        attn_logits = (linear_attn - nonlinear_dist) * self.scale
        attn = attn_logits.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features or in_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features or in_features, in_features)
    def forward(self, x): return self.fc2(self.act(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        # 替换为 XicorAttention
        self.attn = XicorAttention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * 4)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class SkeleEmbed(nn.Module):
    def __init__(self, dim_in=1, dim_feat=256, patch_size=5, t_patch_size=24):
        super().__init__()
        self.proj = nn.Conv2d(dim_in, dim_feat, kernel_size=[t_patch_size, patch_size], stride=[t_patch_size, patch_size])
    def forward(self, x):
        return self.proj(x.permute(0, 3, 1, 2)).flatten(2).transpose(1, 2)

class IndustryMAE_Xicor(nn.Module):
    def __init__(self, num_signal=65, num_frames=360, dim_feat=256):
        super().__init__()
        self.signal_embed = SkeleEmbed(dim_in=1, dim_feat=dim_feat)
        num_patches = (num_signal // 5) * (num_frames // 24)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim_feat))
        self.blocks = nn.ModuleList([Block(dim=dim_feat) for _ in range(4)])
        self.norm = nn.LayerNorm(dim_feat)
        self.predict_head = nn.Linear(num_patches * dim_feat, 24 * 3)

    def forward(self, x):
        B = x.shape[0]
        x_patches = self.signal_embed(x.unsqueeze(-1)) + self.pos_embed
        for blk in self.blocks: x_patches = blk(x_patches)
        return self.predict_head(self.norm(x_patches).reshape(B, -1)).reshape(B, 24, 3)

class IndustrialDataset(Dataset):
    def __init__(self, norm_data):
        self.data = norm_data
        self.samples = [idx for idx in range(0, len(norm_data) - 360 - 24, 200)]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        start = self.samples[idx]
        return torch.FloatTensor(self.data[start:start+360]), torch.FloatTensor(self.data[start+360:start+384, -3:])

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_data = np.load("“中控杯”工业大模型初赛训练集.npy").astype(np.float32)
    if raw_data.ndim == 2 and raw_data.shape[0] == 65: raw_data = raw_data.T
    
    clean_data = advanced_clean_data(raw_data)
    mean, std = np.mean(clean_data, axis=0), np.std(clean_data, axis=0) + 1e-5
    norm_data = (clean_data - mean) / std
    
    train_loader = DataLoader(IndustrialDataset(norm_data), batch_size=32, shuffle=True)
    model = IndustryMAE_Xicor().to(device)
    
    MAX_EPOCHS = 50
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)
    
    print(f"\n=================== ⚙️ 启动 Xicor 深空炼丹 ({MAX_EPOCHS} 轮) ===================")
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"📈 Epoch {epoch+1}/{MAX_EPOCHS} | Xicor MSE Loss: {avg_loss:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_v2_xicor.pth")
        else:
            patience_counter += 1
            if patience_counter >= 8:
                print(f"🛑 Xicor 早停触发，最优 MSE 已锁定在第 {epoch+1} 轮。")
                break

    model.load_state_dict(torch.load("best_model_v2_xicor.pth"))
    
    print("\n=================== 📦 正在执行 V2 Xicor 滚动推理收网 ===================")
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for i in range(10):
            print(f"⏳ Xicor 外推预测第 {i+1}/10 个切片...")
            batch_data = norm_data[i * 60000 : (i + 1) * 60000].copy()
            for step in range(500):
                pred_start = 48000 + step * 24
                input_tensor = torch.FloatTensor(batch_data[pred_start - 360 : pred_start]).unsqueeze(0).to(device)
                batch_data[pred_start : pred_start + 24, -3:] = model(input_tensor).squeeze(0).cpu().numpy()
            pred_real = (batch_data[48000 : 60000, -3:] * std[-3:]) + mean[-3:]
            all_predictions.append(pred_real.T)

    final_output = np.stack(all_predictions).transpose(1, 0, 2)
    np.save("predictions_v2_xicor_submit.npy", final_output)
    print("\n🎉 V2 激进冲榜版完工！交卷文件: predictions_v2_xicor_submit.npy")