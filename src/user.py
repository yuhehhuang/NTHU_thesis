import pandas as pd
import random
from geopy.distance import distance
from geopy.point import Point

T = 96 * 60 // 15  # 384 個 slot
NUM_USERS = 1000
NUM_GROUPS = 20
USERS_PER_GROUP = NUM_USERS // NUM_GROUPS
CENTER_LAT = 40.0386
CENTER_LON = -75.5966

center = Point(CENTER_LAT, CENTER_LON)
users = []

for group_id in range(NUM_GROUPS):
    group_t_start = random.randint(0, T - 120)  # 為這一組決定一個起始時間

    for i in range(USERS_PER_GROUP):
        user_id = group_id * USERS_PER_GROUP + i

        duration = random.randint(20, 100)  # 服務時長
        t_end = min(group_t_start + duration, T - 1)  # 確保不超出 T

        # 產生隨機經緯度（半徑 0.8 km 內）
        angle = random.uniform(0, 360)
        radius = random.uniform(0, 0.8)
        point = distance(kilometers=radius).destination(center, bearing=angle)

        users.append({
            "user_id": user_id,
            "t_start": group_t_start,
            "t_end": t_end,
            "lat": point.latitude,
            "lon": point.longitude
        })

df = pd.DataFrame(users)
df.to_csv("data/user_info.csv", index=False)
print("✅ 產生 user_info.csv")