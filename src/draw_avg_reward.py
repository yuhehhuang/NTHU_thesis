import os
import glob
import pandas as pd

# === 參數設定 ===
W = 2
alpha = 1
folder_path = "results"

# 找到所有 data_rates.csv
pattern_data = f"**/*_W{W}_alpha{alpha}_*data_rates.csv"
files_data = glob.glob(os.path.join(folder_path, pattern_data), recursive=True)

results = []

for file_data in files_data:
    method = os.path.basename(file_data).split(f"_W{W}_")[0]

    # 找對應的 load_by_time.csv
    load_file_pattern = file_data.replace("data_rates.csv", "load_by_time.csv")
    if not os.path.exists(load_file_pattern):
        print(f"[警告] {method} 缺少 load_by_time.csv，跳過")
        continue

    # 讀取
    df_data = pd.read_csv(file_data)
    df_load = pd.read_csv(load_file_pattern)

    # 合併 (time, sat)
    merged = pd.merge(
        df_data,
        df_load,
        on=["time", "sat"],
        how="left"
    )

    # 計算 Reward = (1 - alpha * L) * data_rate
    merged["reward"] = (1 - alpha * merged["load"]) * merged["data_rate"]

    # 平均 reward（全部數據直接平均）
    avg_reward = merged["reward"].mean()

    results.append({"method": method, "avg_reward": avg_reward})

# 轉成 DataFrame
df_results = pd.DataFrame(results)
print(df_results)
