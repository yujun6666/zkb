import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def advanced_clean_data(raw_data):
    """
    100% 融合组员 data_clean.py 的高级清洗算法
    """
    print("🧽 正在调用组员的高级插值算法在内存中清洗数据...")
    df = pd.DataFrame(raw_data)
    
    # 替换异常值 (处理 inf 和 0.0)
    df.replace([np.inf, -np.inf, 0.0], np.nan, inplace=True)
    
    # 核心：尝试使用三次样条插值或线性插值，完美缝合丢失数据
    try:
        df = df.interpolate(method='cubic', limit_direction='both')
    except:
        df = df.interpolate(method='linear', limit_direction='both')
        
    # 兜底填充，确保没有任何 NaN 残留
    df = df.bfill().ffill().fillna(0.0)
    
    return df.values

class IndustrialDataset(Dataset):
    """工业时序切片装载器"""
    def __init__(self, norm_data):
        self.data = norm_data
        self.samples = [idx for idx in range(0, len(norm_data) - 360 - 24, 200)]
            
    def __len__(self): 
        return len(self.samples)
        
    def __getitem__(self, idx):
        start = self.samples[idx]
        history = torch.FloatTensor(self.data[start : start + 360])
        future_target = torch.FloatTensor(self.data[start + 360 : start + 360 + 24, -3:])
        return history, future_target