import torch

# ====== 1. 路径配置 ======
TRAIN_DATA_PATH = "“中控杯”工业大模型初赛训练集.npy"  # 你的原始题库
VAL_DATA_PATH = "验证集-题目集.npy"                   # 官方发的新考卷
MODEL_SAVE_PATH = "best_model_mae.pth"                   # 炼出的学长标准金丹
SCALER_SAVE_PATH = "data_scaler_mae.npy"                 # 保存的归一化尺子
SUBMIT_SAVE_PATH = "predictions_val_submit.npy"       # 最终交卷的 NPY

# ====== 2. 硬件与超参数配置 ======
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05
PATIENCE = 8  # 触发早停的忍耐轮数