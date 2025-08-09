import random
import copy
import pandas as pd
from collections import defaultdict
from src.utils import compute_sinr_and_rate, compute_score, update_m_s_t_from_channels, check_visibility
from src.dp import run_dp_path_for_user
import random
class Individual:
    def __init__(self, user_df, access_matrix, W, path_loss, sat_channel_dict, params, seed=None):
        self.user_df = user_df
        self.access_matrix = access_matrix
        self.df_access = pd.DataFrame(access_matrix)
        self.W = W
        self.path_loss = path_loss
        self.sat_channel_dict = sat_channel_dict
        self.params = params

        # 初始化 RNG（可重現）
        self.rng = random.Random(seed)

        self.position = {}
        self.data_rates = []
        self.reward = 0

        self.generate_fast_path()
    def generate_fast_path(self):
            self.position = {}
            self.data_rates = []
            self.reward = 0

            tmp_sat_dict = copy.deepcopy(self.sat_channel_dict)
            total_reward = 0

            # ✅ 依 t_start 遞增排序，再 groupby t_start；每個批次內打亂
            for t_val, group_df in self.user_df.sort_values("t_start").groupby("t_start"):
                users = list(group_df.itertuples(index=False))
                self.rng.shuffle(users)   # 只打亂同一個 t_start 批次內的順序

                for user in users:
                    user_id = int(user.user_id)
                    t_start = int(user.t_start)
                    t_end = int(user.t_end)

                    t = t_start
                    current_sat, current_ch = None, None
                    last_ho_time = t_start
                    is_first_handover = True

                    user_path = []
                    data_rate_records = []
                    user_reward = 0

                    # ==========================
                    # 1) 第一次選擇（在 t_start）
                    # ==========================
                    best_sat, best_ch, best_score, best_data_rate = None, None, float("-inf"), 0.0

                    for sat in self.access_matrix[t_start]["visible_sats"]:
                        for ch in tmp_sat_dict[sat]:
                            if tmp_sat_dict[sat][ch] > 0:
                                continue
                            _, data_rate = compute_sinr_and_rate(self.params, self.path_loss, sat, t_start, tmp_sat_dict, ch)
                            if data_rate is None or data_rate <= 0:
                                continue
                            m_s_t = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())
                            score = compute_score(self.params, m_s_t, data_rate, sat)
                            # 平手時加一點隨機抖動，避免每個體完全一致（很小，不影響趨勢）
                            score += 1e-9 * self.rng.random()
                            if score > best_score:
                                best_score = score
                                best_sat, best_ch = sat, ch
                                best_data_rate = data_rate

                    if best_sat is None:
                        self.position[user_id] = []
                        continue

                    current_sat, current_ch = best_sat, best_ch
                    user_path.append((current_sat, current_ch, t_start))
                    data_rate_records.append((user_id, t_start, current_sat, current_ch, best_data_rate))
                    m_s_t0 = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())
                    user_reward += compute_score(self.params, m_s_t0, best_data_rate, current_sat)

                    # ==========================
                    # 2) t_begin 之後
                    # ==========================
                    t = t_start + 1
                    while t <= t_end:
                        can_handover = is_first_handover or (t - last_ho_time >= self.W)
                        did_handover = False

                        best_sat, best_ch, best_score = current_sat, current_ch, float("-inf")

                        if can_handover:
                            vsats = self.access_matrix[t]["visible_sats"]
                            # 小隨機性：遍歷順序也洗一下（不影響正確性，增加多樣性）
                            vsats = list(vsats)
                            self.rng.shuffle(vsats)

                            for sat in vsats:
                                ch_list = list(tmp_sat_dict[sat].keys())
                                self.rng.shuffle(ch_list)
                                for ch in ch_list:
                                    if tmp_sat_dict[sat][ch] > 0:
                                        continue
                                    if not check_visibility(self.df_access, sat, t, min(t_end, t + self.W - 1)):
                                        continue
                                    _, dr0 = compute_sinr_and_rate(self.params, self.path_loss, sat, t, tmp_sat_dict, ch)
                                    if dr0 is None or dr0 <= 0:
                                        continue
                                    m_s_t = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())
                                    score0 = compute_score(self.params, m_s_t, dr0, sat) + 1e-9 * self.rng.random()
                                    if score0 > best_score:
                                        best_score = score0
                                        best_sat, best_ch = sat, ch

                            if (best_sat is not None) and (best_sat != current_sat or best_ch != current_ch):
                                current_sat, current_ch = best_sat, best_ch
                                last_ho_time = t
                                is_first_handover = False
                                did_handover = True

                        step = self.W if did_handover else 1

                        for w in range(step):
                            tt = t + w
                            if tt > t_end:
                                break
                            if not did_handover:
                                if current_sat not in self.access_matrix[tt]["visible_sats"]:
                                    break

                            _, dr = compute_sinr_and_rate(self.params, self.path_loss, current_sat, tt, tmp_sat_dict, current_ch)

                            user_path.append((current_sat, current_ch, tt))
                            data_rate_records.append((user_id, tt, current_sat, current_ch, dr if dr else 0.0))

                            if dr and dr > 0:
                                m_s_t = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())
                                user_reward += compute_score(self.params, m_s_t, dr, current_sat)

                        t += step

                    if user_path:
                        self.position[user_id] = user_path
                        for s, c in set((s, c) for s, c, _ in user_path):
                            tmp_sat_dict[s][c] += 1
                        self.data_rates.extend(data_rate_records)
                        total_reward += user_reward
                    else:
                        self.position[user_id] = []

            self.reward = total_reward
    def rebuild_from_position(self):
        """根據目前 self.position（各 user 的 path）重建 data_rates / reward，
        並用 user-based 規則更新佔用：每個 user 用過的 (sat,ch) 於該 user 結束時 +1。"""
        self.data_rates = []
        self.reward = 0.0

        # 佔用快照（從初始字典重建，不污染 self.sat_channel_dict）
        tmp_sat_dict = copy.deepcopy(self.sat_channel_dict)
        total_reward = 0.0

        # 依 t_start 批次處理使用者（維持時間順序）
        df_ts = self.user_df[["user_id", "t_start"]].copy()
        df_ts = df_ts.sort_values("t_start")

        for _, row in df_ts.iterrows():
            user_id = int(row["user_id"])
            path = self.position.get(user_id, [])
            if not path:
                continue

            # 時間排序（保險）
            path = sorted(path, key=lambda x: x[2])

            user_reward = 0.0
            used_pairs = set()  # 紀錄該 user 實際有使用過的 (sat,ch)

            for sat, ch, t in path:
                # 可視性檢查（保守：不可視就跳過此 slot）
                if sat not in self.access_matrix[t]["visible_sats"]:
                    # 直接丟棄這個 slot
                    continue

                # 計算該 slot 的速率
                _, dr = compute_sinr_and_rate(self.params, self.path_loss, sat, t, tmp_sat_dict, ch)
                self.data_rates.append((user_id, t, sat, ch, dr if dr else 0.0))

                # 速率 > 0 才納入 reward
                if dr and dr > 0:
                    m_s_t = update_m_s_t_from_channels(tmp_sat_dict, tmp_sat_dict.keys())
                    user_reward += compute_score(self.params, m_s_t, dr, sat)
                    used_pairs.add((sat, ch))  # 只有真的有 throughput 的才算「用過」

            # ✅ user-based 佔用：該 user 結束後，將他實際用過的 (sat,ch) 做 +1
            for s, c in used_pairs:
                # 注意：你的原邏輯是「用過就 +1」，不看時間長度
                tmp_sat_dict[s][c] = tmp_sat_dict[s].get(c, 0) + 1

            # 更新 reward
            total_reward += user_reward

            # 將 path（保留你先前的 path，不在這裡改動；若你要「丟掉不可視/無速率的 slot」，
            # 可把上面有效 slot 收集成 valid_path，再覆蓋 self.position[user_id] = valid_path）

        self.reward = total_reward
# 其餘 GA 相關邏輯略，建議類似補註解

class GeneticAlgorithm:
    def __init__(self, population_size, user_df, access_matrix, W, path_loss, sat_channel_dict, params, seed=None):
        self.population_size = population_size
        self.user_df = user_df
        self.access_matrix = access_matrix
        self.W = W
        self.path_loss = path_loss
        self.params = params

        # 基底種子（可重現）。若未提供，隨機取一個
        self.seed_base = seed if seed is not None else random.randrange(1 << 30)

        # 建立多樣化初始族群：每個 Individual 給不同 seed
        self.population = [
            Individual(
                user_df=self.user_df,
                access_matrix=self.access_matrix,
                W=self.W,
                path_loss=self.path_loss,
                sat_channel_dict=copy.deepcopy(sat_channel_dict),  # 每個體自己的副本
                params=self.params,
                seed=self.seed_base + i * 7919  # 不同種子（質數間距避免規律）
            )
            for i in range(self.population_size)
        ]

        # 依 reward 排序並保存目前最佳
        self.population.sort(key=lambda ind: ind.reward, reverse=True)
        self.best_individual = copy.deepcopy(self.population[0])

    def evolve(self, generations, elite_size=2, mutation_rate=0.2):
        # 開始前保險：確保每個個體的 reward 是「由當前 position 計的」
        for ind in self.population:
            ind.rebuild_from_position()

        for _ in range(generations):
            next_gen = self.population[:elite_size]  # 精英保留（可選：deepcopy）

            while len(next_gen) < len(self.population):
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                child = self.crossover(parent1, parent2)
                self.mutate(child, mutation_rate)
                next_gen.append(child)

            # 重算（以防萬一），再排序
            for ind in next_gen:
                ind.rebuild_from_position()

            self.population = sorted(next_gen, key=lambda ind: ind.reward, reverse=True)
            if self.population[0].reward > self.best_individual.reward:
                self.best_individual = copy.deepcopy(self.population[0])
    def tournament_selection(self, k=3):
        # 從族群中隨機抽取 k 個個體，選出 reward 最佳者
        candidates = random.sample(self.population, k)
        return max(candidates, key=lambda ind: ind.reward)

    def crossover(self, parent1, parent2):
        child = copy.deepcopy(parent1)
        for user_id in child.position:
            if random.random() < 0.5:
                child.position[user_id] = copy.deepcopy(parent2.position.get(user_id, []))
        # ✅ 重算 fitness（user-based 佔用）
        child.rebuild_from_position()
        return child

    def mutate(self, individual, mutation_rate):
        mutated = False
        for user in individual.user_df.itertuples():
            if random.random() < mutation_rate:
                user_id = int(user.user_id)
                t_start = int(user.t_start)
                t_end = int(user.t_end)

                path, reward, success, data_rate_records = run_dp_path_for_user(
                    user_id, t_start, t_end, individual.W,
                    individual.access_matrix, individual.path_loss,
                    copy.deepcopy(individual.sat_channel_dict),
                    individual.params
                )
                if success:
                    individual.position[user_id] = path
                    mutated = True

        if mutated:
            # ✅ 重算 fitness（user-based 佔用）
            individual.rebuild_from_position()

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
