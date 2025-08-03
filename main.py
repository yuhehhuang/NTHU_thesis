import os
import copy
import pandas as pd
from src.init import load_system_data
from src.greedy import run_greedy_per_W
from src.utils import recompute_all_data_rates

# === 1️⃣ 載入系統資料 ===
system = load_system_data(regenerate_sat_channels=False)  # 想重新生成隨機頻道 → 改成 True
df_users = system["users"]
df_access = system["access_matrix"]
path_loss = system["path_loss"]
params = system["params"]
sat_channel_dict_backup = system["sat_channel_dict_backup"]

# 設定 W 與 alpha
W = 2
alpha = params["alpha"]  # 從 system_params.json 讀取

# === 2️⃣ 執行 Greedy 演算法 ===
results_df, all_user_paths, load_by_time, df_data_rates = run_greedy_per_W(
    user_df=df_users,
    access_matrix=df_access.to_dict(orient="records"),
    path_loss=path_loss,
    sat_load_dict_backup=copy.deepcopy(sat_channel_dict_backup),
    params=params,
    W=W
)

# === 3️⃣ 重新計算正確的 data rate (考慮初始隨機分配的干擾) ===
df_correct_rates = recompute_all_data_rates(
    all_user_paths, path_loss, params, sat_channel_dict_backup
)

# 讓排序跟 df_data_rates 一致
df_correct_rates = df_correct_rates.set_index(["user_id", "time"]).reindex(
    df_data_rates.set_index(["user_id", "time"]).index
).reset_index()

# === 4️⃣ 建立 results 資料夾 ===
os.makedirs("results", exist_ok=True)

# === 5️⃣ 儲存 Greedy 演算法結果 ===
results_df.to_csv(f"results/greedy_results_W{W}_alpha{alpha}.csv", index=False)
pd.DataFrame(all_user_paths).to_csv(f"results/greedy_paths_W{W}_alpha{alpha}.csv", index=False)
df_data_rates.to_csv(f"results/greedy_data_rates_W{W}_alpha{alpha}.csv", index=False)

# === 6️⃣ 輸出包含 initial 隨機分配負載的 load_by_time ===
records = []
df_access = df_access.reset_index(drop=True)  # 確保 index 對應時間

for t in range(len(df_access)):
    visible_sats = df_access.loc[t, "visible_sats"]
    for sat in visible_sats:
        greedy_load = load_by_time[t].get(sat, 0)
        random_load = sum(sat_channel_dict_backup[sat].values()) if sat in sat_channel_dict_backup else 0
        total_load = greedy_load + random_load
        records.append({"time": t, "sat": sat, "load": total_load})

df_load = pd.DataFrame(records)
df_load.to_csv(f"results/greedy_load_by_time_W{W}_alpha{alpha}.csv", index=False)

# === 7️⃣ 輸出重新計算後的正確 data rate ===
df_correct_rates.to_csv(f"results/greedy_real_data_rates_W{W}_alpha{alpha}.csv", index=False)

# === 8️⃣ 完成訊息 ===
print("✅ Greedy 方法完成！")
print(f"📄 已輸出結果到 results/greedy_results_W{W}_alpha{alpha}.csv")
print(f"📄 已輸出路徑到 results/greedy_paths_W{W}_alpha{alpha}.csv")
print(f"📄 已輸出 Data Rate (分配時序列式計算) 到 results/greedy_data_rates_W{W}_alpha{alpha}.csv")
print(f"📄 已輸出 Load 狀態到 results/greedy_load_by_time_W{W}_alpha{alpha}.csv")
print(f"📄 已輸出正確 Data Rate (重新計算干擾) 到 results/greedy_real_data_rates_W{W}_alpha{alpha}.csv")
