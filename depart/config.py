# config.py
import torch

# 路径配置
DATA_PATH = "“中控杯”工业大模型初赛训练集.npy"
MODEL_SAVE_PATH = "best_model_v1.pth"
SCALER_SAVE_PATH = "data_scaler_v1.npy"     # 拆分代码必须保存的归一化参数
SUBMIT_SAVE_PATH = "predictions_v1_submit.npy"

# 硬件配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型与训练超参数
MAX_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05
PATIENCE = 8  # 早停容忍轮数