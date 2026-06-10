import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

print("🔥 V1 启动：高级数据清洗 + 深度百轮特训 + 早停机制...")

# ==================== [1] 高级精细化数据清洗 ====================
def advanced_clean_data(raw_data):
    print("🧽 正在执行工业级精细化数据清洗 (剔除毛刺 + 线性插值)...")
    df = pd.DataFrame(raw_data)
    
    # 1. 将绝对值为 0.0 或无穷大的异常点暂时设为 NaN
    df.replace([np.inf, -np.inf, 0.0], np.nan, inplace=True)
    
    # 2. 时序双向线性插值 (平滑缝合丢失的数据)
    df = df.interpolate(method='linear', limit_direction='both')
    
    # 3. 兜底填充 (防止开头结尾全是 NaN)
    df = df.bfill().ffill()
    
    # 如果还有残留的 NaN，补 0 兜底
    return df.fillna(0.0).values

# ==================== [2] 标准 Transformer 架构 ====================
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

class IndustrialDataset(Dataset):
    def __init__(self, norm_data):
        self.data = norm_data
        self.samples = []
        for idx in range(0, len(norm_data) - 360 - 24, 200):
            self.samples.append(idx)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        start = self.samples[idx]
        return torch.FloatTensor(self.data[start : start + 360]), torch.FloatTensor(self.data[start + 360 : start + 360 + 24, -3:])

# ==================== [3] 主程序 ====================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_PATH = "“中控杯”工业大模型初赛训练集.npy"
    
    raw_data = np.load(DATA_PATH).astype(np.float32)
    if raw_data.ndim == 2 and raw_data.shape[0] == 65: raw_data = raw_data.T
    
    # 启用高级清洗
    clean_data = advanced_clean_data(raw_data)
    mean, std = np.mean(clean_data, axis=0), np.std(clean_data, axis=0) + 1e-5
    norm_data = (clean_data - mean) / std
    
    train_loader = DataLoader(IndustrialDataset(norm_data), batch_size=32, shuffle=True)
    model = IndustryMAE().to(device)
    
    # 引入高级优化器与学习率衰减
    MAX_EPOCHS = 50
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)
    
    print(f"\n=================== ⚙️ 正在执行大模型深度训练 ({MAX_EPOCHS} 轮) ===================")
    epoch_losses = []
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
        epoch_losses.append(avg_loss)
        print(f"📈 Epoch {epoch+1}/{MAX_EPOCHS} | LR: {scheduler.get_last_lr()[0]:.6f} | MSE Loss: {avg_loss:.6f}")
        
        # 早停与最佳模型保存机制
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_v1.pth")
        else:
            patience_counter += 1
            if patience_counter >= 8:
                print(f"🛑 触发早停机制！MSE已在第 {epoch+1} 轮极致收敛。")
                break

    model.load_state_dict(torch.load("best_model_v1.pth"))
    
    print("\n=================== 📦 正在执行 V1 滚动推理收网 ===================")
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for i in range(10):
            print(f"⏳ 外推预测第 {i+1}/10 个切片...")
            batch_data = norm_data[i * 60000 : (i + 1) * 60000].copy()
            for step in range(500):
                pred_start = 48000 + step * 24
                input_tensor = torch.FloatTensor(batch_data[pred_start - 360 : pred_start]).unsqueeze(0).to(device)
                batch_data[pred_start : pred_start + 24, -3:] = model(input_tensor).squeeze(0).cpu().numpy()
            pred_real = (batch_data[48000 : 60000, -3:] * std[-3:]) + mean[-3:]
            all_predictions.append(pred_real.T)

    final_output = np.stack(all_predictions).transpose(1, 0, 2)
    np.save("predictions_v1_submit.npy", final_output)
    print("\n🎉 V1 纯净进阶版完工！交卷文件: predictions_v1_submit.npy")