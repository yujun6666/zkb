import numpy as np
import pandas as pd
import os

print("🔄 正在加载 60 万行原始工业数据集...")
FILE_PATH = "“中控杯”工业大模型初赛训练集.npy"

if not os.path.exists(FILE_PATH):
    print(f"❌ 找不到文件 {FILE_PATH}，请检查路径。")
    exit()

# 1. 加载数据并对齐形状
raw_data = np.load(FILE_PATH).astype(np.float32)
if raw_data.ndim == 2 and raw_data.shape[0] == 65:
    raw_data = raw_data.T  # 转置为 (600000, 65)

print(f"📊 数据加载成功！矩阵形状为: {raw_data.shape}")

# 2. 生成清晰的列名
# 前 62 个是已知特征，最后 3 个是我们需要预测的目标
columns = [f"Signal_{i}" for i in range(62)] + [
    "Signal_62 (Target 1)", 
    "Signal_63 (Target 2)", 
    "Signal_64 (Target 3)"
]

# 3. 转换为 DataFrame 并插入时间步轴
df = pd.DataFrame(raw_data, columns=columns)
df.insert(0, 'Time_Step(时间步)', np.arange(1, len(df) + 1))

# 4. 导出为 CSV
output_file = "原始训练集_全量60万行.csv"
print(f"⏳ 正在导出为 CSV 文件 (数据量达 60 万行，硬盘写入大概需要 10~30 秒，请稍候)...")
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("\n🎉==================================================")
print(f"✅ 导出成功！所有传感器数据已解包保存为: {output_file}")
print("💡 温馨提示：")
print("   1. 文件较大(约 400MB~500MB)。")
print("   2. 强烈建议使用 VS Code 自带的 CSV 插件，或者用 Excel 打开。")
print("   3. 如果用 Excel 打开，加载时可能会卡顿几秒钟，属于正常现象。")
print("====================================================")