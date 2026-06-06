import numpy as np
import pandas as pd

# 1. 加载官方原始数据
print("正在加载官方原始数据 zk_competition.npy...")
raw_data = np.load("data/zk_competition.npy")

# 2. 设置列名（第0列=时间，第1-64列=信号0~信号63）
col_names = ["时间"] + [f"信号{i}" for i in range(64)]

# 3. 转换为DataFrame
df = pd.DataFrame(raw_data, columns=col_names)

# 4. 保存为CSV文件（大文件适配，无乱码）
output_path = "data/zk_competition_data.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig", chunksize=10000)

print(f"✅ 原始数据CSV生成完成！文件路径：{output_path}")
print(f"✅ 数据总行数：{len(df)}，总列数：{len(df.columns)}")
