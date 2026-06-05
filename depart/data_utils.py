# data_utils.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def advanced_clean_data(raw_data):
    """工业级精细化数据清洗 (剔除毛刺 + 线性插值)"""
    print("🧽 正在执行数据清洗...")
    df = pd.DataFrame(raw_data)
    # 将绝对值为 0.0 或无穷大的异常点暂时设为 NaN
    df.replace([np.inf, -np.inf, 0.0], np.nan, inplace=True)
    # 时序双向线性插值 (平滑缝合丢失的数据)
    df = df.interpolate(method='linear', limit_direction='both')
    # 兜底填充 (防止开头结尾全是 NaN)
    df = df.bfill().ffill()
    return df.fillna(0.0).values

class IndustrialDataset(Dataset):
    """工业时序数据集装载器"""
    def __init__(self, norm_data):
        self.data = norm_data
        self.samples = []
        for idx in range(0, len(norm_data) - 360 - 24, 200):
            self.samples.append(idx)
            
    def __len__(self): 
        return len(self.samples)
        
    def __getitem__(self, idx):
        start = self.samples[idx]
        history = torch.FloatTensor(self.data[start : start + 360])
        future_target = torch.FloatTensor(self.data[start + 360 : start + 360 + 24, -3:])
        return history, future_target