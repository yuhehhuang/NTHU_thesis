import random
import copy
import pandas as pd
from collections import defaultdict
from src.utils import compute_sinr_and_rate, compute_score, update_m_s_t_from_channels, check_visibility
from src.dp import run_dp_path_for_user
class Individual:
    """
    一個個體（染色體），表示所有使用者的 path 分配方案。
    包含分配路徑與 reward。
    """
    def __init__(self, user_df, access_matrix, W, path_loss, sat_channel_dict, params):
        # 初始化使用者資料與系統參數
        self.user_df = user_df
        self.access_matrix = access_matrix
        self.W = W
        self.path_loss = path_loss
        self.sat_channel_dict = sat_channel_dict
        self.params = params

        self.position = {}  # 儲存每個 user 的 path
        self.data_rates = []  # 儲存所有 user 的 data rate 紀錄
        self.reward = 0  # 總 reward

        self.generate_fast_path()  # 使用快速初始化方法建立初始解

    def generate_fast_path(self):
        # 初始化結果儲存容器
        self.position = {}
        self.data_rates = []
        self.reward = 0

        tmp_sat_dict = copy.deepcopy(self.sat_channel_dict)  # 複製 channel 使用狀態
        total_reward = 0  # 全部使用者的總 reward

        # 遍歷每個使用者
        for _, user in self.user_df.iterrows():
            user_id = int(user["user_id"])
            t_start = int(user["t_start"])
            t_end = int(user["t_end"])
            t = t_start
            current_sat, current_ch = None, None  # 當前使用的衛星與頻道
            last_ho_time = t_start  # 上次 handover 的時間
            is_first = True  # 是否為第一次分配
            user_path = []  # 該 user 的 path
            data_rate_records = []  # 該 user 的 data rate 記錄
            user_reward = 0  # 該 user 累積 reward

            # 開始依照 time slot 分配 path
            while t <= t_end:
                can_handover = is_first or (t - last_ho_time >= self.W)  # 是否可以換手
                best_score, best_sat, best_ch, best_data_rate = -1e9, None, None, 0  # 初始化最佳選擇

                # === 不換手情況 ===
                if current_sat is not None and current_ch is not None:
                    m_s_t = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())  # 更新負載
                    SINR, data_rate = compute_sinr_and_rate(self.params, self.path_loss, current_sat, t, tmp_sat_dict, current_ch)
                    if data_rate is not None:
                        score = compute_score(self.params, m_s_t, data_rate, current_sat)
                        if score > best_score:
                            best_score = score
                            best_sat = current_sat
                            best_ch = current_ch
                            best_data_rate = data_rate

                # === 換手情況 ===
                if can_handover:
                    for sat in self.access_matrix[t]["visible_sats"]:  # 所有可見衛星
                        for ch in tmp_sat_dict[sat]:  # 該衛星下所有 channel
                            if tmp_sat_dict[sat][ch] == 1:
                                continue  # 該 channel 已被佔用
                            # 檢查是否未來 W slot 都可見
                            if not check_visibility(pd.DataFrame(self.access_matrix), sat, t, min(t_end, t + self.W - 1)):
                                continue
                            SINR, data_rate = compute_sinr_and_rate(self.params, self.path_loss, sat, t, tmp_sat_dict, ch)
                            if data_rate is None:
                                continue
                            m_s_t = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())
                            score = compute_score(self.params, m_s_t, data_rate, sat)
                            if score > best_score:
                                best_score = score
                                best_sat = sat
                                best_ch = ch
                                best_data_rate = data_rate

                # === 若找不到合法選擇則中止 ===
                if best_sat is None:
                    break

                # 更新換手狀態與目前選擇
                if (best_sat != current_sat or best_ch != current_ch):
                    last_ho_time = t
                    is_first = False
                current_sat, current_ch = best_sat, best_ch

                # 將這次決定的 (sat, ch) 套用到接下來 W 個 slot
                for w in range(self.W):
                    if t + w > t_end:
                        break
                    user_path.append((current_sat, current_ch, t + w))
                    data_rate_records.append((user_id, t + w, current_sat, current_ch, best_data_rate))
                    user_reward += best_score  # 每個 slot 累積 reward

                t += self.W

            # 如果 user 有合法路徑，就記錄下來
            if user_path:
                self.position[user_id] = user_path
                for s, c in set((s, c) for s, c, _ in user_path):  # ✅ 修正 unpack 錯誤
                    tmp_sat_dict[s][c] += 1  # 更新負載
                self.data_rates.extend(data_rate_records)
                total_reward += user_reward  # ✅ 改用累積 reward
            else:
                self.position[user_id] = []  # 無路徑則設為空

        self.reward = total_reward  # 記錄總 reward


# 其餘 GA 相關邏輯略，建議類似補註解

class GeneticAlgorithm:
    def __init__(self, population_size, user_df, access_matrix, W, path_loss, sat_channel_dict, params):
        # 初始化族群，產生多個個體（染色體）作為初始解
        self.population = [
            Individual(user_df, access_matrix, W, path_loss, sat_channel_dict, params)
            for _ in range(population_size)
        ]
        # 根據 reward 對初始族群排序，保留最好的個體作為 best_individual
        self.population.sort(key=lambda ind: ind.reward, reverse=True)
        self.best_individual = copy.deepcopy(self.population[0])

    def evolve(self, generations, elite_size=2, mutation_rate=0.2):
        # 執行多次演化迭代
        for _ in range(generations):
            next_gen = self.population[:elite_size]  # 精英策略：保留前 elite_size 名不變

            while len(next_gen) < len(self.population):  # 補足族群大小
                parent1 = self.tournament_selection()  # 使用錦標賽選擇法挑選親代
                parent2 = self.tournament_selection()
                child = self.crossover(parent1, parent2)  # 交配產生新個體
                self.mutate(child, mutation_rate)  # 隨機突變
                next_gen.append(child)

            # 重新排序並更新當前最好的個體
            self.population = sorted(next_gen, key=lambda ind: ind.reward, reverse=True)
            if self.population[0].reward > self.best_individual.reward:
                self.best_individual = copy.deepcopy(self.population[0])

    def tournament_selection(self, k=3):
        # 從族群中隨機抽取 k 個個體，選出 reward 最佳者
        candidates = random.sample(self.population, k)
        return max(candidates, key=lambda ind: ind.reward)

    def crossover(self, parent1, parent2):
        # 對每個 user 決定從 parent1 或 parent2 複製路徑
        child = copy.deepcopy(parent1)
        for user_id in child.position:
            if random.random() < 0.5:
                child.position[user_id] = copy.deepcopy(parent2.position[user_id])
        return child

    def mutate(self, individual, mutation_rate):
        # 對每個 user 根據機率進行突變（重新生成路徑）
        for user in individual.user_df.itertuples():
            if random.random() < mutation_rate:
                user_id = int(user.user_id)
                t_start = int(user.t_start)
                t_end = int(user.t_end)

                # 直接用 DP 重新計算該 user 的最佳路徑
                path, reward, success, data_rate_records = run_dp_path_for_user(
                    user_id, t_start, t_end, individual.W,
                    individual.access_matrix, individual.path_loss,
                    copy.deepcopy(individual.sat_channel_dict),  # 用快照避免污染其他 user
                    individual.params
                )

                # 若成功產生路徑則更新個體
                if success:
                    individual.position[user_id] = path
    def export_best_result(self):
        # 將最佳個體的結果轉換為與主程式一致的格式（用於儲存與分析）
        best = self.best_individual
        all_user_paths = []
        load_by_time = defaultdict(lambda: defaultdict(int))
        all_user_data_rates = []
        results = []

        for user_id, path in best.position.items():
            if not path:
                results.append({"user_id": user_id, "reward": None, "success": False})
                continue

            reward = 0  # reward 可在後處理補算
            t_begin = min(t for _, _, t in path)
            t_end = max(t for _, _, t in path)
            for s, c, t in path:
                load_by_time[t][s] += 1

            data_rate_records = []
            for s, c, t in path:
                data_rate_records.append((user_id, t, s, c, 0))  # data_rate 可補算

            all_user_paths.append({
                "user_id": user_id,
                "path": path,
                "t_begin": t_begin,
                "t_end": t_end,
                "success": True,
                "reward": reward
            })
            results.append({"user_id": user_id, "reward": reward, "success": True})
            all_user_data_rates.extend(data_rate_records)

        df_data_rates = pd.DataFrame(all_user_data_rates, columns=["user_id", "time", "sat", "channel", "data_rate"])
        return pd.DataFrame(results), all_user_paths, load_by_time, df_data_rates
