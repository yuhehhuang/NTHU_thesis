import numpy as np
import os
import random
import pandas as pd
from collections import defaultdict
from src.init import load_system_data
from src.greedy import run_greedy_per_W 
from src.utils import (
    compute_sinr_and_rate,
    compute_score,
    update_m_s_t_from_channels,
    check_visibility
)
import copy
# 1️⃣ 載入系統資料
system = load_system_data()
df_users = system["users"]
df_access = system["access_matrix"]
sat_positions = system["sat_positions"]
path_loss = system["path_loss"]  # (sat, t)
params = system["params"]
num_channels = params["num_channels"]

# 2️⃣ 初始化所有衛星
# 紀錄T的所有出現衛星
all_satellites = set()
for sats in df_access["visible_sats"]:
    all_satellites.update(sats)


# 初始化衛星負載function
def init_sat_channel_all(satellites, num_channels=25, randomize=True, max_background_users=10):
    """
    初始化每顆衛星每個 channel 的使用狀態
    return: dict[sat_name][channel] = 0/1
    """
    sat_channel_dict = {}

    for sat in satellites:
        # key: satellite ,value:{0:0,1:0,...,24:0} 左邊的數字代表第幾個channel，右邊的數字代表是否使用
        sat_channel_dict[sat] = {ch: 0 for ch in range(num_channels)}

        if randomize:
            # 從[0,25]編號的channel，隨機取k個 channel 作為背景使用 
            used_channels = random.sample(range(num_channels),  
                                          k=random.randint(0, min(max_background_users, num_channels)))
            for ch in used_channels:
                sat_channel_dict[sat][ch] = 1

    return sat_channel_dict


# 3️⃣ 初始化衛星頻道狀態
sat_channel_dict_backup  = init_sat_channel_all(
    satellites=all_satellites,
    num_channels=num_channels,
    randomize=True,
    max_background_users=10
)
# 4️⃣ 執行 Greedy 演算法
results_df, all_user_paths, load_by_time, df_data_rates = run_greedy_per_W(
    user_df=df_users,
    access_matrix=df_access.to_dict(orient="records"),
    path_loss=path_loss,
    sat_load_dict_backup=copy.deepcopy(sat_channel_dict_backup),
    params=params,   # ✅ 改這裡
    W=4
)
# 5️⃣ 輸出結果
os.makedirs("results", exist_ok=True)

# 儲存結果
results_df.to_csv("results/greedy_results.csv", index=False)
pd.DataFrame(all_user_paths).to_csv("results/greedy_paths.csv", index=False)
df_data_rates.to_csv("results/greedy_data_rates.csv", index=False)

print("✅ Greedy 方法完成！")
print(f"📄 已輸出結果到 results/greedy_results.csv")
print(f"📄 已輸出路徑到 results/greedy_paths.csv")
print(f"📄 已輸出 Data Rate 到 results/greedy_data_rates.csv")