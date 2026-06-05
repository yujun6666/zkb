import numpy as np
import matplotlib.pyplot as plt
import os

print("📊 正在为你绘制时序大模型预测轨迹图...")

# 💡 在这里修改你想查看的版本！
# 如果你想看 V1 进阶版，填 "predictions_v1_submit.npy"
# 如果你想看 V2 激进版，填 "predictions_v2_xicor_submit.npy"
SUBMIT_PATH = "predictions_v1_submit.npy"  

if not os.path.exists(SUBMIT_PATH):
    print(f"❌ 错误：找不到文件 {SUBMIT_PATH}。请确认你已经运行了对应的训练脚本！")
    exit()

raw_data = np.load("“中控杯”工业大模型初赛训练集.npy").astype(np.float32)
if raw_data.ndim == 2 and raw_data.shape[0] == 65: raw_data = raw_data.T
submit_data = np.load(SUBMIT_PATH)

# 我们选取 信号62（即第一个预测目标）的第一个 Batch 来画图展示
# 截取历史最后 1000 步用于画图对比
history_plot = raw_data[47000:48000, -3]
pred_plot = submit_data[0, 0, :] # 未来 12000 步

# 创建时间轴
time_history = np.arange(47000, 48000)
time_pred = np.arange(48000, 48000 + 12000)

plt.figure(figsize=(12, 6))
# 画出已知的历史（用蓝色）
plt.plot(time_history, history_plot, color='#1f77b4', linewidth=2, label='True History (Last 1000 steps)')
# 画出模型外推预测的未来（用橙色虚线，代表这是未来）
plt.plot(time_pred, pred_plot, color='#ff7f0e', linestyle='--', linewidth=1.5, label=f'Model Forecast Future ({SUBMIT_PATH})')

plt.axvline(x=48000, color='red', linestyle=':', linewidth=2, label='Prediction Start Boundary')
plt.title(f"Industrial Time-Series Forecasting Visualization [{SUBMIT_PATH}]", fontsize=12)
plt.xlabel("Time Step (Index)", fontsize=10)
plt.ylabel("Signal 62 Sensor Value", fontsize=10)
plt.grid(True, linestyle='--')
plt.legend()

# 保存图片时，自动把文件名带上版本号，防止覆盖！
pic_name = f"prediction_future_vision_{SUBMIT_PATH.split('_')[1]}.png"
plt.savefig(pic_name, dpi=300)
print("\n🎉==================================================")
print(f"💾 可视化曲线图绘制成功！已保存为: {pic_name}")
print("====================================================")