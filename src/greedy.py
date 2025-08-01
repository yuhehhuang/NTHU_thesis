import json
from collections import defaultdict, Counter
import pandas as pd
from src.utils import (
    compute_sinr_and_rate,
    compute_score,
    update_m_s_t_from_channels,
    check_visibility
)

####################################################################################################
from collections import Counter
import pandas as pd
from src.utils import compute_sinr_and_rate, compute_score, update_m_s_t_from_channels, check_visibility

def run_greedy_path_for_user(
    user_id: int,
    t_start: int,
    t_end: int,
    W: int,
    access_matrix: list,
    path_loss: dict,
    sat_channel_dict: dict,
    params: dict
):
    num_channels = params["num_channels"]

    current_sat, current_ch = None, None
    last_ho_time = t_start
    is_first_handover = True
    path = []
    total_reward = 0
    data_rate_records = []

    # ==========================
    # 1️⃣ 第一次選擇衛星與 channel
    # ==========================
    best_sat, best_ch, best_score, best_data_rate = None, None, -1, 0

    for sat in access_matrix[t_start]["visible_sats"]:
        for ch in sat_channel_dict[sat]:
            if sat_channel_dict[sat][ch] == 1:
                continue
            SINR, data_rate = compute_sinr_and_rate(params, path_loss, sat, t_start, sat_channel_dict, ch)
            if data_rate is None:
                continue
            m_s_t = update_m_s_t_from_channels(sat_channel_dict, sat_channel_dict.keys())
            score = compute_score(params, m_s_t, data_rate, sat)

            if score > best_score:
                best_score = score
                best_sat = sat
                best_ch = ch
                best_data_rate = data_rate

    if best_sat is None:
        return [], 0, False, []

    # ✅ 紀錄第一次選擇結果
    current_sat, current_ch = best_sat, best_ch
    path.append((current_sat, current_ch, t_start))
    data_rate_records.append((user_id, t_start, current_sat, current_ch, best_data_rate))
    total_reward += best_score

    # ==========================
    # 2️⃣ 從下一個 slot 開始
    # ==========================
    t = t_start + 1
    while t <= t_end:
        # 計算當前衛星的 score
        _, data_rate_curr = compute_sinr_and_rate(params, path_loss, current_sat, t, sat_channel_dict, current_ch)
        if data_rate_curr is None:
            current_score = -1
        else:
            m_s_t = update_m_s_t_from_channels(sat_channel_dict, sat_channel_dict.keys())
            current_score = compute_score(params, m_s_t, data_rate_curr, current_sat)

        best_sat, best_ch, best_score, best_data_rate = current_sat, current_ch, current_score, data_rate_curr

        # ==========================
        # 需要換手時重新找衛星
        # ==========================
        if is_first_handover or (t - last_ho_time >= W) or (current_score <= 0):
            for sat in access_matrix[t]["visible_sats"]:
                if sat == current_sat:
                    continue

                # 檢查未來 W slot 是否可見
                future_visible = check_visibility(pd.DataFrame(access_matrix), sat, t, min(t_end, t + W - 1))
                if not future_visible:
                    continue

                for ch in sat_channel_dict[sat]:
                    if sat_channel_dict[sat][ch] == 1:
                        continue
                    SINR, data_rate = compute_sinr_and_rate(params, path_loss, sat, t, sat_channel_dict, ch)
                    if data_rate is None:
                        continue
                    m_s_t = update_m_s_t_from_channels(sat_channel_dict, sat_channel_dict.keys())
                    score = compute_score(params, m_s_t, data_rate, sat)

                    if score > best_score:
                        best_score = score
                        best_sat = sat
                        best_ch = ch
                        best_data_rate = data_rate

        # ==========================
        # 如果換了衛星，更新 handover 時間
        # ==========================
        if best_sat != current_sat or best_ch != current_ch:
            current_sat, current_ch = best_sat, best_ch
            last_ho_time = t
            is_first_handover = False

        # ==========================
        # 固定使用這個衛星與 channel W 個 slot
        # ==========================
        for w in range(W):
            if t + w > t_end:
                break
            path.append((current_sat, current_ch, t + w))
            _, dr = compute_sinr_and_rate(params, path_loss, current_sat, t + w, sat_channel_dict, current_ch)
            data_rate_records.append((user_id, t + w, current_sat, current_ch, dr if dr else 0))
            total_reward += best_score

        t += W

    return path, total_reward, True, data_rate_records


##################################################################################
def run_greedy_per_W(
    user_df: pd.DataFrame,
    access_matrix: list,
    path_loss: dict,
    sat_load_dict_backup: dict,
    params: dict,        # ✅ 接收 main 傳進來的 params
    W: int = 4
):
    sat_load_dict = {sat: chs.copy() for sat, chs in sat_load_dict_backup.items()}

    active_user_paths = []
    all_user_paths = []
    results = []
    load_by_time = defaultdict(lambda: defaultdict(int))
    all_user_data_rates = []

    for _, user in user_df.iterrows():
        user_id = int(user["user_id"])
        t_start = int(user["t_start"])
        t_end = int(user["t_end"])

        # 釋放已完成的使用者
        to_delete = []
        for old_user in active_user_paths:
            if old_user["t_end"] < t_start:
                path = old_user.get("path")
                if path:
                    usage_count = Counter((s, c) for s, c, _ in path)
                    for (sat, ch), count in usage_count.items():
                        sat_load_dict[sat][ch] = max(0, sat_load_dict[sat][ch] - count)

                    for s, c, t in path:
                        if t in load_by_time and s in load_by_time[t]:
                            load_by_time[t][s] = max(0, load_by_time[t][s] - 1)

                to_delete.append(old_user)
        for u in to_delete:
            active_user_paths.remove(u)

        # 計算路徑
        path, reward, success, data_rate_records = run_greedy_path_for_user(
            user_id=user_id,
            t_start=t_start,
            t_end=t_end,
            W=W,
            access_matrix=access_matrix,
            path_loss=path_loss,
            sat_channel_dict=sat_load_dict,
            params=params       # ✅ 這裡使用傳進來的 params
        )

        # 更新負載
        if path:
            usage_count = Counter((s, c) for s, c, _ in path)
            for (sat, ch), count in usage_count.items():
                sat_load_dict[sat][ch] += count
            for s, c, t in path:
                load_by_time[t][s] = load_by_time[t].get(s, 0) + 1

        all_user_data_rates.extend(data_rate_records)

        # 紀錄結果
        if success:
            active_user_paths.append({
                "user_id": user_id,
                "path": path,
                "t_begin": t_start,
                "t_end": t_end,
                "success": success,
                "reward": reward
            })

        all_user_paths.append({
            "user_id": user_id,
            "path": path,
            "t_begin": t_start,
            "t_end": t_end,
            "success": success,
            "reward": reward
        })

        results.append({
            "user_id": user_id,
            "reward": reward if success else None,
            "success": success
        })

    df_data_rates = pd.DataFrame(all_user_data_rates, columns=["user_id", "time", "sat", "channel", "data_rate"])
    return pd.DataFrame(results), all_user_paths, load_by_time, df_data_rates

#user_data_rates的結構:
#{
#  1: {0: 15.3, 1: 16.8, 2: 14.7},   # user 1 在每個時間的 data rate
#  2: {3: 12.1, 4: 10.5, 5: 13.0}    # user 2 在每個時間的 data rate
#}