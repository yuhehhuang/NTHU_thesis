import numpy as np

def update_m_s_t_from_channels(sat_channel_dict, all_sats):
    """根據 channel 使用狀態計算每顆衛星的總負載"""
    return {sat: sum(sat_channel_dict[sat].values()) for sat in all_sats}


def check_visibility(df_access, sat, t_start, t_end):
    """檢查衛星在 t_start~t_end 是否連續可見"""
    for t in range(t_start, t_end + 1):
        row = df_access[df_access["time_slot"] == t]
        if row.empty or sat not in row["visible_sats"].values[0]:
            return False
    return True

def compute_sinr_and_rate(params, path_loss, sat, t, sat_channel_dict, chosen_channel):
    PL = path_loss.get((sat, t))
    if PL is None:
        return None, None

    P_rx = params["eirp_linear"] * params["grx_linear"] / PL

    interference = 0
    for other_sat, channels in sat_channel_dict.items():
        if other_sat == sat:
            continue
        if channels.get(chosen_channel, 0) == 1:
            PL_other = path_loss.get((other_sat, t))
            if PL_other:
                interference += params["eirp_linear"] * params["grx_linear"] / PL_other

    SINR = P_rx / (params["noise_watt"] + interference)
    data_rate = params["channel_bandwidth_hz"] * np.log2(1 + SINR) /1e6  # Mbps
    return SINR, data_rate
    
def compute_score(params, m_s_t, data_rate, sat):
    L = m_s_t[sat] / params["num_channels"]
    return (1 - params["alpha"] * L) * data_rate