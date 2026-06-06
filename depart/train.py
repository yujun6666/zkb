# train.py
import torch
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import DataLoader
import config
from data_utils import advanced_clean_data, IndustrialDataset
from model import IndustryMAE

def main():
    print(f"🔥 开始训练！使用的设备: {config.DEVICE}")
    
    # 1. 锁死直接读取你准备好的 cleaned_data.csv
    import pandas as pd
    CSV_PATH = "cleaned_data.csv"
    print(f"📂 正在直接读取你指定的清洗后数据集: {CSV_PATH}...")
    
    df = pd.read_csv(CSV_PATH)
    
    # 💡 安全机制 1：剔除文本时间列
    if '时间' in df.columns:
        df = df.drop(columns=['时间'])
    if 'Time_Step(时间步)' in df.columns:
        df = df.drop(columns=['Time_Step(时间步)'])
        
    clean_data = df.values.astype(np.float32)
    print(f"📊 CSV数据加载成功！当前矩阵形状为: {clean_data.shape}")
    
    # 💡 安全机制 2：动态兼容列数（对齐模型底层的刚性结构）
    current_cols = clean_data.shape[1]
    if current_cols != 65:
        print(f"⚠️ 警告：你的CSV只有 {current_cols} 列（可能因为data_clean.py删除了死通道）。")
        print(f"🛠️ 正在自动补齐至 65 列，以防大模型报错崩溃...")
        # 如果列数不够65，自动在右侧用0补齐到65列
        padding = np.zeros((clean_data.shape[0], 65 - current_cols), dtype=np.float32)
        clean_data = np.hstack([clean_data, padding])
        print(f"✅ 补齐完成！最终矩阵形状已对齐为: {clean_data.shape}")

    # 2. 归一化并【保存归一化参数】
    mean = np.mean(clean_data, axis=0)
    std = np.std(clean_data, axis=0) + 1e-5
    np.save(config.SCALER_SAVE_PATH, np.stack([mean, std]))
    
    norm_data = (clean_data - mean) / std
    train_loader = DataLoader(IndustrialDataset(norm_data), batch_size=config.BATCH_SIZE, shuffle=True)
    
    # 3. 初始化模型与优化器
    model = IndustryMAE().to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.MAX_EPOCHS, eta_min=1e-6)
    
    # 4. 训练循环
    print(f"\n=================== ⚙️ 正在执行大模型深度训练 ({config.MAX_EPOCHS} 轮) ===================")
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.MAX_EPOCHS):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(config.DEVICE), batch_y.to(config.DEVICE)
            optimizer.zero_grad()
            loss = F.mse_loss(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"📈 Epoch {epoch+1}/{config.MAX_EPOCHS} | LR: {scheduler.get_last_lr()[0]:.6f} | MSE Loss: {avg_loss:.6f}")
        
        # 早停与最佳模型保存机制
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"🛑 触发早停机制！MSE已在第 {epoch+1} 轮极致收敛。")
                break
                
    print("✅ 训练完毕！金丹已炼成！")

if __name__ == '__main__':
    main()
