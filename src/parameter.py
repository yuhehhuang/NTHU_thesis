import json

params = {
    "total_bandwidth_hz": 2e9,
    "num_channels": 25,
    "channel_bandwidth_hz": 2e9 / 25,
    "eirp_dbw": 22.5,
    "grx_dbi": 0,
    "noise_temperature_k": 290,
    "boltzmann": 1.38e-23,
    "alpha": 0.5
}

with open("data/system_params.json", "w") as f:
    json.dump(params, f, indent=4)

print("✅ 已建立 data/system_params.json")