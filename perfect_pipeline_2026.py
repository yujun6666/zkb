import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

print("🔥 启动中控杯工业大模型【学长原版自注意力+500轮自回归滚动推理】终极稳定引擎...")

# ==================== [1] 学长原版高精 Transformer 架构 ====================
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
        
        # 💡 绝杀内存崩溃：只预测未来短长 24 步 × 3 个信号 = 72 维，完美卸载百兆内存负载！
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

# ==================== [2] 稳健的数据批次化切片 Dataset ====================
class IndustrialDataset(Dataset):
    def __init__(self, norm_data):
        self.data = norm_data
        self.samples = []
        # 以 200 为滑动间隔切出大量训练子序列，保障 15 个 Epoch 的充分学习
        for idx in range(0, len(norm_data) - 360 - 24, 200):
            self.samples.append(idx)
            
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        start = self.samples[idx]
        x = self.data[start : start + 360]
        y = self.data[start + 360 : start + 360 + 24, -3:]
        return torch.FloatTensor(x), torch.FloatTensor(y)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_PATH = "“中控杯”工业大模型初赛训练集.npy"
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ 错误：找不到原始数据集 {DATA_PATH}")
        exit()
        
    raw_data = np.load(DATA_PATH).astype(np.float32)
    if raw_data.ndim == 2 and raw_data.shape[0] == 65: raw_data = raw_data.T
    
    # 全自动数据清洗过滤
    if np.isnan(raw_data).any():
        print("⚠️ 警报：检测到原始数据中包含缺失值！正在智能清洗...")
        raw_data = np.nan_to_num(raw_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    mean = np.mean(raw_data, axis=0)
    std = np.std(raw_data, axis=0) + 1e-5
    norm_data = (raw_data - mean) / std
    
    # 启用标准小批次装载，彻底锁死内存泄露
    train_dataset = IndustrialDataset(norm_data)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = IndustryMAE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    # ------------------ 阶段 1：深度预训练 + 微调一体化演练 (15 Epochs) ------------------
    print(f"\n=================== ⚙️ 正在执行大模型基础训练 (15 轮) ===================")
    epoch_losses = []
    
    for epoch in range(15):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = F.mse_loss(preds, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 控住梯度
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"📈 Epoch {epoch+1}/15 | 工业指标预测核心误差 (MSE Loss): {avg_loss:.4f}")
        
    # 📊 生成高精度的 MSE 下降收敛轨迹图
    plt.figure(figsize=(9, 4.5))
    plt.plot(epoch_losses, marker='s', color='#1f77b4', linewidth=2, label='Model Training Loss')
    plt.title('Industrial Transformer MSE Loss Curve (15 Epochs)', fontsize=12)
    plt.xlabel('Training Epoch', fontsize=10)
    plt.ylabel('MSE Loss Value', fontsize=10)
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.savefig("mse_loss_curve.png", dpi=300)
    print("✅ 成果一：误差收敛曲线图已成功导出为: mse_loss_curve.png")

    torch.save(model.state_dict(), "final_perfect_model_2026.pth")
    print("✅ 成果二：复核备战存档成功生成: final_perfect_model_2026.pth")

    # ------------------ 阶段 2：高精尖 500 次自回归滚动推理 ------------------
    print("\n=================== 📦 正在执行阶段三：500轮自回归滚动推理收网 ===================")
    model.eval()
    batch_size = 60000 # 对应官方6万行大切片
    all_predictions = []
    
    with torch.no_grad():
        for i in range(10):
            print(f"⏳ 正在滚动外推预测第 {i+1}/10 个工业切片...")
            batch_data = norm_data[i * batch_size : (i + 1) * batch_size].copy() # 拷贝当前大Batch
            
            # 严格滚动外推 500 次（500次 * 24步 = 12000步）
            for step in range(500):
                pred_start = 48000 + step * 24
                # 抽出已知时序最后的 360 步历史窗口
                history_window = batch_data[pred_start - 360 : pred_start]
                input_tensor = torch.FloatTensor(history_window).unsqueeze(0).to(device)
                
                # 模型精准吐出未来 24 步 [1, 24, 3]
                pred_norm = model(input_tensor).squeeze(0).cpu().numpy()
                
                # 🔥 滚动回填核心：把这一轮预测的24步，实时填入未来未知区域，作为下一步滚动的历史！
                batch_data[pred_start : pred_start + 24, -3:] = pred_norm
                
            # 500轮滚雪球结束，把这 12000 步的最终纯净预测结果提取并执行反归一化
            final_norm_12000 = batch_data[48000 : 60000, -3:]
            pred_real = (final_norm_12000 * std[-3:]) + mean[-3:]
            all_predictions.append(pred_real.T) # 揉成 [3, 12000]

    # 严格揉合为初赛平台刚性规定的极致交卷形态
    final_output = np.stack(all_predictions) # [10, 3, 12000]
    if final_output.shape == (10, 3, 12000):
        final_output = final_output.transpose(1, 0, 2) # 完美兑现 (3, 10, 12000)
        
    np.save("predictions_2026_submit.npy", final_output)
    print("\n🎉====================================================================")
    print("💾 战役全面打通！！初赛全量大模型闭环控制网络已顺利收网！")
    print("📂 最终预测提交文件已安稳降落：predictions_2026_submit.npy")
    print(f"📊 矩阵规格检验：{final_output.shape} (百分之百完美咬合平台初赛硬性指标！)")
    print("=======================================================================")