import pandas as pd
import pickle
import numpy as np
from math import radians, cos, sin, atan2, degrees

FC = 14.5e9
C_LIGHT = 3e8
EARTH_RADIUS = 6371.0

# 3GPP TR 38.811 Table 6.6.2-1 (Ka-band LOS)
KA_BAND_LOS_SIGMA = {10: 2.9, 20: 2.4, 30: 2.7, 40: 2.4, 50: 2.4,
                     60: 2.7, 70: 2.6, 80: 2.8, 90: 0.6}

def get_sigma_sf(elev):
    return KA_BAND_LOS_SIGMA[min(KA_BAND_LOS_SIGMA.keys(), key=lambda x: abs(x - elev))]

def fspl(distance_km, fc):
    d_m = distance_km * 1000
    return (4 * np.pi * d_m * fc / C_LIGHT) ** 2

def geodetic_to_ecef(lat, lon, alt_km=0):
    lat_r, lon_r = radians(lat), radians(lon)
    x = (EARTH_RADIUS + alt_km) * cos(lat_r) * cos(lon_r)
    y = (EARTH_RADIUS + alt_km) * cos(lat_r) * sin(lon_r)
    z = (EARTH_RADIUS + alt_km) * sin(lat_r)
    return np.array([x, y, z])

def compute_elevation(user_ecef, sat_ecef):
    user_norm = user_ecef / np.linalg.norm(user_ecef)
    sat_vec = sat_ecef - user_ecef
    sat_norm = sat_vec / np.linalg.norm(sat_vec)
    angle = degrees(atan2(np.linalg.norm(np.cross(user_norm, sat_norm)), np.dot(user_norm, sat_norm)))
    return 90 - angle

# === 讀取資料 ===
df_users = pd.read_csv("data/user_info.csv")
with open("data/satellite_positions.pkl", "rb") as f:
    sat_positions = pickle.load(f)
df_access = pd.read_csv("data/access_matrix.csv")
df_access["visible_sats"] = df_access["visible_sats"].apply(eval)

# 用 target area 中心點作為代表
center_lat = df_users["lat"].mean()
center_lon = df_users["lon"].mean()
center_ecef = geodetic_to_ecef(center_lat, center_lon)

PL_dict = {}
count = 0

for t in df_access["time_slot"].unique():
    row = df_access[df_access["time_slot"] == t]
    if row.empty:
        continue

    visible_sats = row["visible_sats"].values[0]

    for sat_name in visible_sats:
        sat_ecef = np.array(sat_positions.get((sat_name, t)))
        if sat_ecef is None:
            continue

        # 計算距離與仰角
        d_km = np.linalg.norm(sat_ecef - center_ecef)
        elevation = compute_elevation(center_ecef, sat_ecef)

        # FSPL
        FSPL_linear = fspl(d_km, FC)

        # Shadow Fading (依仰角決定 σSF)
        sigma_sf = get_sigma_sf(min(90, max(10, elevation)))
        sf_db = np.random.normal(0, sigma_sf)
        sf_linear = 10 ** (sf_db / 10)

        PL_dict[(sat_name, t)] = FSPL_linear * sf_linear
        count += 1

print(f"✅ 計算完成，共 {count} 筆資料")

with open("data/path_loss.pkl", "wb") as f:
    pickle.dump(PL_dict, f)
