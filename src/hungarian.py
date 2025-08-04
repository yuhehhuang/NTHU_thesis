from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd
import ast
from collections import defaultdict
from src.utils import (
    compute_sinr_and_rate,
)

def compute_sat_load(channel_status_dict):
    """簡易負載計算：使用中頻道數 / 總頻道數"""
    total_channels = len(channel_status_dict)
    used_channels = sum(channel_status_dict.values())
    return used_channels / total_channels if total_channels > 0 else 0

def run_hungarian_per_W(df_users, df_access, path_loss, sat_channel_dict_backup, sat_positions, params, W):
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    import pandas as pd
    import ast
    from collections import defaultdict
    from src.utils import compute_sinr_and_rate
    import copy

    time_slots = len(df_access)
    alpha = params["alpha"]
    sat_load_dict = {sat: chs.copy() for sat, chs in sat_channel_dict_backup.items()}
    user_assignments = defaultdict(list)
    data_rate_records = []
    load_by_time = defaultdict(lambda: defaultdict(int))

    t_global = 0  # ✅ 外層時間，從頭到尾跑一次
    while t_global < time_slots:

        # ✅ Step 1: 擷取這個時間 t_global 進場的 user
        batch_users = df_users[df_users["t_start"] == t_global].index.tolist()
        if not batch_users:
            t_global += 1
            continue

        # ✅ Step 2: 處理這批 user 從 t_global 到他們的 t_end 範圍內的所有時間點
        t = t_global
        next_available_time = {uid: t for uid in batch_users}
        remaining_users = set(batch_users)

        while len(remaining_users) > 0 and t < time_slots:
            # ✅ Step 3: 釋放那些已完成任務的 user 所佔資源
            for uid in list(user_assignments):
                t_end = df_users.loc[uid, "t_end"]
                if t == t_end + 1:
                    for _, sat, ch in user_assignments[uid]:
                        sat_load_dict[sat][ch] = 0

            # ✅ Step 4: 找出這一批中，現在可以做分配的 user
            candidate_users = [
                uid for uid in remaining_users
                if next_available_time[uid] == t and df_users.loc[uid, "t_end"] >= t
            ]
            if not candidate_users:
                t += 1
                continue

            # ✅ Step 5: 取得目前 t 可見衛星清單
            visible_sats_str = df_access[df_access["time_slot"] == t]["visible_sats"].iloc[0]
            visible_sats = ast.literal_eval(visible_sats_str) if isinstance(visible_sats_str, str) else visible_sats_str

            # ✅ Step 6: 建立所有可用的 (sat, ch) 配對及其 reward
            candidate_pairs = []
            pair_scores = {}
            for sat in visible_sats:
                if sat not in sat_load_dict:
                    continue
                for ch, occupied in sat_load_dict[sat].items():
                    if occupied != 0:
                        continue
                    if (sat, t) not in path_loss:
                        continue
                    _, rate = compute_sinr_and_rate(params, path_loss, sat, t, sat_load_dict, ch)
                    load_score = 1 - compute_sat_load(sat_load_dict[sat])
                    score = load_score * rate * alpha
                    candidate_pairs.append((sat, ch))
                    pair_scores[(sat, ch)] = {
                        "score": score,
                        "data_rate": rate
                    }

            if not candidate_pairs:
                t += 1
                continue

            # ✅ Step 7: 建立 cost matrix 並做匹配
            n_users = len(candidate_users)
            n_pairs = len(candidate_pairs)
            cost_matrix = np.full((n_users, n_pairs), 1e9)

            for i, uid in enumerate(candidate_users):
                for j, (sat, ch) in enumerate(candidate_pairs):
                    cost_matrix[i, j] = -pair_scores[(sat, ch)]["score"]

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # ✅ Step 8: 執行實際分配
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] > 1e8:
                    continue

                uid = candidate_users[i]
                sat, ch = candidate_pairs[j]
                info = pair_scores[(sat, ch)]
                t_end = df_users.loc[uid, "t_end"]
                last_used_sat = user_assignments[uid][-1][1] if user_assignments[uid] else None
                handover = (last_used_sat != sat)
                t_last = min(t + W - 1, t_end) if handover else t

                for t_used in range(t, t_last + 1):
                    sat_load_dict[sat][ch] = 1
                    user_assignments[uid].append((t_used, sat, ch))
                    data_rate_records.append({
                        "user_id": uid,
                        "time": t_used,
                        "sat": sat,
                        "channel": ch,
                        "data_rate": info["data_rate"]
                    })
                    load_by_time[t_used][sat] += 1

                next_available_time[uid] = t_last + 1
                if next_available_time[uid] > t_end:
                    remaining_users.remove(uid)

            t += 1  # ➕ 繼續處理這批 user 的下一個時間

        # ✅ 全部完成後，才前進下一批使用者（新的 t_start）
        t_global += 1

    # ✅ 整理成你希望的輸出格式
    df_results = pd.DataFrame(data_rate_records)
    formatted_paths = []
    for uid, entries in user_assignments.items():
        if not entries:
            continue
        entries = sorted(entries, key=lambda x: x[0])
        path_list = [(sat, ch, t) for (t, sat, ch) in entries]
        t_begin = entries[0][0]
        t_end = entries[-1][0]
        success = (t_end - t_begin + 1) == (df_users.loc[uid, "t_end"] - df_users.loc[uid, "t_start"] + 1)
        total_rate = sum(d["data_rate"] for d in data_rate_records if d["user_id"] == uid)
        formatted_paths.append([uid, str(path_list), t_begin, t_end, success, total_rate])

    df_paths = pd.DataFrame(formatted_paths, columns=["user_id", "path", "t_begin", "t_end", "success", "reward"])
    return df_results, df_paths, load_by_time
