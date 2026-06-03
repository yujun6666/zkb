import numpy as np
import pandas as pd
import os

print("🔄 正在为你拼装原始历史与预测未来数据...")

ORIGINAL_PATH = "“中控杯”工业大模型初赛训练集.npy"
SUBMIT_PATH = "predictions_2026_submit.npy"

if not os.path.exists(SUBMIT_PATH):
    print(f"❌ 错误：找不到预测文件 {SUBMIT_PATH}，请确认你已经跑完了训练。")
    exit()

# 1. 读取数据
raw_data = np.load(ORIGINAL_PATH).astype(np.float32)
if raw_data.ndim == 2 and raw_data.shape[0] == 65: 
    raw_data = raw_data.T
submit_data = np.load(SUBMIT_PATH) # 形状: (3, 10, 12000)

# 2. 提取第1个大切片(Batch 0)的历史已知数据（前48000步），我们只关心最后3个需要预测的信号
history_62 = raw_data[0:48000, -3]
history_63 = raw_data[0:48000, -2]
history_64 = raw_data[0:48000, -1]

# 3. 提取我们模型预测的未来 12000 步数据
pred_62 = submit_data[0, 0, :] # 信号0, Batch 0
pred_63 = submit_data[1, 0, :] # 信号1, Batch 0
pred_64 = submit_data[2, 0, :] # 信号2, Batch 0

# 4. 物理拼接：历史 + 未来
full_62 = np.concatenate([history_62, pred_62])
full_63 = np.concatenate([history_63, pred_63])
full_64 = np.concatenate([history_64, pred_64])

# 5. 标记数据类型，方便你在 Excel 里筛选查看
data_type = ["历史已知"] * 48000 + ["大模型预测未来"] * 12000

# 6. 打包成表格
df = pd.DataFrame({
    "时间步长(Row)": np.arange(1, 60001),
    "数据类型": data_type,
    "信号_62(预测目标1)": full_62,
    "信号_63(预测目标2)": full_63,
    "信号_64(预测目标3)": full_64
})

# 保存文件
output_file = "预测结果与历史拼接对照表.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("\n🎉==================================================")
# 修正了之前的占位符显示，确保输出信息完全对应你今年的65特征和12000步要求
print(f"💾 表格生成成功！已保存为: {output_file}")
print("💡 提示：你可以直接在 VS Code 里双击它，或者用 Excel 打开查看！")
print("   前 48000 行是真实历史，第 48001 到 60000 行是我们预测的 12000 步！")
print("====================================================")