import pandas as pd
import pickle
import json

def load_system_data():
    # === 讀取 User 資料 ===
    df_users = pd.read_csv("data/user_info.csv")

    # === 讀取 Access Matrix ===
    df_access = pd.read_csv("data/access_matrix.csv")
    df_access["visible_sats"] = df_access["visible_sats"].apply(eval)  # 轉成 list

    # === 讀取衛星座標 ===
    with open("data/satellite_positions.pkl", "rb") as f:
        sat_positions = pickle.load(f)

    # === 讀取 Path Loss ===
    with open("data/path_loss.pkl", "rb") as f:
        path_loss = pickle.load(f)

    # === 讀取系統參數 ===
    with open("data/system_params.json", "r") as f:
        params = json.load(f)

    # 將 EIRP 與 G_rx 轉換為線性值
    params["eirp_linear"] = 10 ** (params["eirp_dbw"] / 10)
    params["grx_linear"] = 10 ** (params["grx_dbi"] / 10)

    system = {
        "users": df_users,
        "access_matrix": df_access,
        "sat_positions": sat_positions,
        "path_loss": path_loss,
        "params": params
    }

    print("✅ System Data Loaded")
    print(f"Users: {len(df_users)}")
    print(f"Time Slots: {len(df_access)}")
    print(f"Sat Positions: {len(sat_positions)} entries")
    print(f"Path Loss entries: {len(path_loss)}")

    return system

if __name__ == "__main__":
    system = load_system_data()
