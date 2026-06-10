import torch
import numpy as np
import config
from data_utils import advanced_clean_data
from model import IndustryMAE

def main():
    print("=================== 📦 正在对【验证集】执行动态滚动推理 ===================")
    
    # 1. 读取【考试题目】
    # 官方发卷形状: (num_features, num_samples, seq_len) -> (65, 10, 60000)
    raw_data = np.load(config.VAL_DATA_PATH).astype(np.float32)
    print(f"📥 原始验证集形状读取成功: {raw_data.shape}")
    
    num_features, num_samples, seq_len = raw_data.shape
    
    # 2. 读取黄金尺子 (用训练集的标准量测验证集)
    scaler = np.load(config.SCALER_SAVE_PATH)
    mean, std = scaler[0], scaler[1]
    
    # 3. 加载炼好的 MAE 金丹
    model = IndustryMAE().to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model.eval()
    
    # 4. 🔥 动态自适应滚动推理引擎 (按样本独立预测)
    all_predictions = []
    future_steps = 12000            # 我们需要外推的未来步数 (500次 * 24步 = 12000)
    
    with torch.no_grad():
        # 遍历 10 个独立样本
        for i in range(num_samples):
            print(f"\n⏳ 正在攻克第 {i+1}/{num_samples} 个考题样本 (官方给出历史长度: {seq_len} 步)...")
            
            # 【关键修复】剥离出第 i 个样本，并转置为 2D: (seq_len, num_features) -> (60000, 65)
            # 这样就能完美适配组员的 advanced_clean_data
            sample_raw = raw_data[:, i, :].T 
            
            # 单样本清洗
            sample_clean = advanced_clean_data(sample_raw)
            
            # 单样本归一化
            sample_norm = (sample_clean - mean) / std
            
            # 创建动态画板
            batch_data = np.zeros((seq_len + future_steps, num_features), dtype=np.float32)
            batch_data[:seq_len, :] = sample_norm
            
            # 疯狂滚动外推 500 次
            for step in range(500):
                pred_start = seq_len + step * 24
                # 向前看 360 步
                input_tensor = torch.FloatTensor(batch_data[pred_start - 360 : pred_start]).unsqueeze(0).to(config.DEVICE)
                
                # 往后填 24 步的 3 个目标变量
                pred_out = model(input_tensor).squeeze(0).cpu().numpy()
                batch_data[pred_start : pred_start + 24, -3:] = pred_out
            
            # 物理反归一化 (仅针对外推出来的 future_steps 和最后 3 个目标特征)
            pred_real = (batch_data[seq_len : seq_len + future_steps, -3:] * std[-3:]) + mean[-3:]
            
            # pred_real 形状为 (12000, 3)
            all_predictions.append(pred_real)

    # 5. 打包输出
    # 此时 all_predictions 包含了 10 个 (12000, 3) 的矩阵
    # stack 后形状变为 (10, 12000, 3) -> (num_samples, pred_len, num_targets)
    final_output = np.stack(all_predictions)
    
    # 【关键修复】根据平台要求重塑形状: (num_targets, num_samples, pred_len)
    # 利用 transpose 将第 2 维(targets)放到最前面，第 0 维(samples)放中间，第 1 维(seq_len)放最后
    final_submit = final_output.transpose(2, 0, 1)
    
    print(f"\n📐 最终战果形状验证: {final_submit.shape} (必须严格等同于 (3, 10, 12000))")
    np.save(config.SUBMIT_SAVE_PATH, final_submit)
    print(f"🎉 考卷填写完毕！交卷文件已生成在根目录: {config.SUBMIT_SAVE_PATH}")

if __name__ == '__main__':
    main()