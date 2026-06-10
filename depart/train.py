# depart/train.py
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast  # 学长的混合精度加速

import config
from data_utils import advanced_clean_data, IndustrialDataset
from model import IndustryMAE

def main():
    print(f"🔥 开始使用学长版 MAE(Depth=8) + 混合精度闭关炼丹！设备: {config.DEVICE}")
    
    raw_data = np.load(config.TRAIN_DATA_PATH).astype(np.float32)
    if raw_data.ndim == 2 and raw_data.shape[0] == 65: raw_data = raw_data.T
    clean_data = advanced_clean_data(raw_data)
    
    mean = np.mean(clean_data, axis=0)
    std = np.std(clean_data, axis=0) + 1e-5
    np.save(config.SCALER_SAVE_PATH, np.stack([mean, std]))
    
    norm_data = (clean_data - mean) / std
    train_loader = DataLoader(IndustrialDataset(norm_data), batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    
    model = IndustryMAE().to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.MAX_EPOCHS, eta_min=1e-6)
    
    scaler = GradScaler() # 激活学长的加速器
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.MAX_EPOCHS):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(config.DEVICE), batch_y.to(config.DEVICE)
            optimizer.zero_grad()
            
            # 使用半精度计算，极大降低显存并提速
            with autocast():
                preds = model(batch_x)
                loss = F.mse_loss(preds, batch_y)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"📈 Epoch {epoch+1}/{config.MAX_EPOCHS} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print("   ✨ 发现更优模型，已保存！")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"🛑 触发早停机制！")
                break
                
    print("✅ 训练完毕！学长级金丹已炼成！")

if __name__ == '__main__':
    main()