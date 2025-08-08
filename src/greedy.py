import json
import copy
from collections import defaultdict, Counter
import pandas as pd
from src.utils import (
    compute_sinr_and_rate,
    compute_score,
    update_m_s_t_from_channels,
    check_visibility
)

####################################################################################################

class Individual:
    """
    一個個體（染色體），表示所有使用者的 path 分配方案。
    包含分配路徑與 reward。
    """
    def __init__(self, user_df, access_matrix, W, path_loss, sat_channel_dict, params):
        # 初始化使用者資料與系統參數
        self.user_df = user_df
        self.access_matrix = access_matrix                  # list[dict] 或等價
        self.df_access = pd.DataFrame(access_matrix)        # 給 check_visibility 用，只建立一次
        self.W = W
        self.path_loss = path_loss
        self.sat_channel_dict = sat_channel_dict
        self.params = params

        self.position = {}   # {user_id: [(sat, ch, t), ...]}
        self.data_rates = [] # [(user_id, t, sat, ch, data_rate), ...]
        self.reward = 0      # 全部使用者的總 reward

        self.generate_fast_path()

    def generate_fast_path(self):
        # 重置結果
        self.position = {}
        self.data_rates = []
        self.reward = 0

        tmp_sat_dict = copy.deepcopy(self.sat_channel_dict)  # 以佔用計數表示
        total_reward = 0

        # === 逐一處理每個使用者 ===
        for _, user in self.user_df.iterrows():
            user_id = int(user["user_id"])
            t_start = int(user["t_start"])
            t_end = int(user["t_end"])

            t = t_start
            current_sat, current_ch = None, None
            last_ho_time = t_start
            is_first_handover = True

            user_path = []
            data_rate_records = []
            user_reward = 0

            # ==========================================================
            # 1) t_begin（第一次選擇）：依 greedy.py 的做法先選一個 (sat,ch)
            #    不強制要求未來 W-slot 都可視；只選當下最好的
            # ==========================================================
            best_sat, best_ch, best_score, best_data_rate = None, None, float("-inf"), 0

            if t_start in range(t_start, t_end + 1):
                visible = self.access_matrix[t_start]["visible_sats"]
                for sat in visible:
                    for ch in tmp_sat_dict[sat]:
                        if tmp_sat_dict[sat][ch] > 0:
                            continue  # 已被佔用
                        SINR, dr = compute_sinr_and_rate(self.params, self.path_loss, sat, t_start, tmp_sat_dict, ch)
                        if dr is None or dr <= 0:
                            continue
                        m_s_t = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())
                        score = compute_score(self.params, m_s_t, dr, sat)
                        if score > best_score:
                            best_score = score
                            best_sat, best_ch = sat, ch
                            best_data_rate = dr

            # 若起始就找不到連線，這個 user 無路徑
            if best_sat is None:
                self.position[user_id] = []
                continue

            # 固定第一次的 (sat,ch)，但只放入 t_start 這一格
            current_sat, current_ch = best_sat, best_ch
            user_path.append((current_sat, current_ch, t_start))
            data_rate_records.append((user_id, t_start, current_sat, current_ch, best_data_rate))
            # t_start 這格計分（只在 dr>0 時）
            if best_data_rate > 0:
                m_s_t0 = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())
                user_reward += compute_score(self.params, m_s_t0, best_data_rate, current_sat)

            # ==========================================================
            # 2) 從下一個 slot 開始：可換手時才檢查未來 W-slot 可視並批次前進 W；
            #    否則不換手，每次只前進 1 slot（只需確保當下可視）
            # ==========================================================
            t = t_start + 1
            while t <= t_end:
                can_handover = is_first_handover or (t - last_ho_time >= self.W)
                did_handover = False

                # 預設保持原 (sat,ch)
                best_sat, best_ch, best_score = current_sat, current_ch, float("-inf")

                if can_handover:
                    vsats = self.access_matrix[t]["visible_sats"]
                    for sat in vsats:
                        for ch in tmp_sat_dict[sat]:
                            if tmp_sat_dict[sat][ch] > 0:
                                continue
                            # 換手才檢查未來 W-slot 可視
                            if not check_visibility(self.df_access, sat, t, min(t_end, t + self.W - 1)):
                                continue
                            _, dr0 = compute_sinr_and_rate(self.params, self.path_loss, sat, t, tmp_sat_dict, ch)
                            if dr0 is None or dr0 <= 0:
                                continue
                            m_s_t = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())
                            score0 = compute_score(self.params, m_s_t, dr0, sat)
                            if score0 > best_score:
                                best_score = score0
                                best_sat, best_ch = sat, ch

                    # 如果挑到更好的 (sat,ch)，就換手
                    if (best_sat is not None) and (best_sat != current_sat or best_ch != current_ch):
                        current_sat, current_ch = best_sat, best_ch
                        last_ho_time = t
                        is_first_handover = False
                        did_handover = True

                # 步長決定：有換手→批次 W；沒換手→只前進 1
                step = self.W if did_handover else 1

                for w in range(step):
                    tt = t + w
                    if tt > t_end:
                        break

                    # 不換手時，只需確認這一格可視（換手已保證整段 W 可視）
                    if not did_handover:
                        if current_sat not in self.access_matrix[tt]["visible_sats"]:
                            break

                    # 逐格計算（避免把第一格的結果複製整段）
                    _, dr = compute_sinr_and_rate(self.params, self.path_loss, current_sat, tt, tmp_sat_dict, current_ch)

                    user_path.append((current_sat, current_ch, tt))
                    data_rate_records.append((user_id, tt, current_sat, current_ch, dr if dr else 0))

                    if dr and dr > 0:
                        m_s_t = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())
                        user_reward += compute_score(self.params, m_s_t, dr, current_sat)

                t += step  # 下一輪

            # === 收尾：寫回個人結果 & 佔用計數 ===
            if user_path:
                self.position[user_id] = user_path
                # 將本 user 使用過的 (sat,ch) 標成佔用（一次即可）
                for s, c in set((s, c) for s, c, _ in user_path):
                    tmp_sat_dict[s][c] += 1
                self.data_rates.extend(data_rate_records)
                total_reward += user_reward
            else:
                self.position[user_id] = []

        self.reward = total_reward

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
    user_df = user_df.sort_values(by="t_start").reset_index(drop=True) #確保user依照開始時間排序，誰先開始request 就開始幫他找路徑
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
                    unique_sat_ch = set((s, c) for s, c, _ in path)
                    for sat, ch in unique_sat_ch:
                        sat_load_dict[sat][ch] = max(0, sat_load_dict[sat][ch] - 1)
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
            params=params       
        )

        # 更新負載 
        if path:
            # 只對 unique (sat, c) 更新一次負載
            unique_sat_ch = set((s, c) for s, c, _ in path)
            for sat, ch in unique_sat_ch:
                sat_load_dict[sat][ch] += 1

            # 但 load_by_time 還是要逐 time slot 累加
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