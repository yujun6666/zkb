# inference.py
import torch
import numpy as np
import config
from data_utils import advanced_clean_data
from model import IndustryMAE

def main():
    print("=================== 📦 正在执行滚动推理收网 ===================")
    
    # 1. 读数据并强制对齐你训练时的 65 维尺度
    raw_data = np.load(config.DATA_PATH).astype(np.float32)
    if raw_data.ndim == 2 and raw_data.shape[0] == 65: 
        raw_data = raw_data.T
    
    # 2. 读取之前保存的归一化参数（保证尺度和训练时 100% 一致）
    scaler = np.load(config.SCALER_SAVE_PATH)
    mean, std = scaler[0], scaler[1]
    norm_data = (clean_data - mean) / std
    
    # 3. 加载训练好的金丹模型
    model = IndustryMAE().to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    model.eval()
    
    # 4. 滚动推理
    all_predictions = []
    with torch.no_grad():
        for i in range(10):
            print(f"⏳ 外推预测第 {i+1}/10 个切片...")
            batch_data = norm_data[i * 60000 : (i + 1) * 60000].copy()
            for step in range(500):
                pred_start = 48000 + step * 24
                input_tensor = torch.FloatTensor(batch_data[pred_start - 360 : pred_start]).unsqueeze(0).to(config.DEVICE)
                batch_data[pred_start : pred_start + 24, -3:] = model(input_tensor).squeeze(0).cpu().numpy()
            
            # 物理反归一化还原
            pred_real = (batch_data[48000 : 60000, -3:] * std[-3:]) + mean[-3:]
            all_predictions.append(pred_real.T)

    final_output = np.stack(all_predictions).transpose(1, 0, 2)
    np.save(config.SUBMIT_SAVE_PATH, final_output)
    print(f"\n🎉 完工！交卷文件已生成: {config.SUBMIT_SAVE_PATH}")

if __name__ == '__main__':
    main()
