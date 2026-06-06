import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ====================== 1. 读取数据 ======================
df = pd.read_csv("data/zk_competition_data.csv", encoding="utf-8-sig")

signal_cols = [c for c in df.columns if "信号" in c]
time_col = df["时间"]

# 强制数值化（防脏数据）
df[signal_cols] = df[signal_cols].apply(pd.to_numeric, errors="coerce")

# ====================== 2. Step 1: 删除死通道 ======================
stds = df[signal_cols].std()

dead_cols = stds[stds == 0].index.tolist()
print(f"❌ 删除死通道数量: {len(dead_cols)}")

df = df.drop(columns=dead_cols)
signal_cols = [c for c in signal_cols if c not in dead_cols]

# ====================== 3. Step 2: Hampel去异常（替代3σ） ======================
def hampel_filter(series, k=7, t0=3):
    x = series.copy()
    rolling_median = x.rolling(k, center=True).median()
    diff = np.abs(x - rolling_median)
    mad = diff.rolling(k, center=True).median()

    threshold = t0 * 1.4826 * mad
    outliers = diff > threshold

    x[outliers] = np.nan
    return x

print("🔧 Hampel filtering...")

for col in signal_cols:
    df[col] = hampel_filter(df[col])

# ====================== 4. Step 3: 限制插值（防造假） ======================
df[signal_cols] = df[signal_cols].interpolate(limit=5, limit_direction="both")

# ====================== 5. Step 4: 轻度平滑（只建议可选） ======================
def ema(series, alpha=0.2):
    return series.ewm(alpha=alpha).mean()

df_smoothed = df.copy()
for col in signal_cols:
    df_smoothed[col] = ema(df[col])

# ====================== 6. Step 5: MinMax（按列！非常重要） ======================
scaler = MinMaxScaler()

df_normalized = pd.DataFrame(
    scaler.fit_transform(df_smoothed[signal_cols]),
    columns=signal_cols
)

# ====================== 7. 拼接输出 ======================
df_final = pd.concat([time_col, df_normalized], axis=1)

df_final.to_csv("data/cleaned_data.csv", index=False, encoding="utf-8-sig")

print("✅ 清洗完成 -> cleaned_data.csv")
