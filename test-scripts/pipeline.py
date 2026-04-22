import time
import requests
import pandas as pd

start_date = "2025-01-01"
end_date = "2025-12-31"

city_out = "city_features.csv"
region_out = "region_features.csv"
state_out = "ca_features.csv"

# cities + basic info
cities = [
    # LA basin
    {"city_name": "Los_Angeles", "region_name": "LA_Basin", "latitude": 34.0522, "longitude": -118.2437, "population": 3900000},
    {"city_name": "Long_Beach", "region_name": "LA_Basin", "latitude": 33.7701, "longitude": -118.1937, "population": 450000},
    {"city_name": "Anaheim", "region_name": "LA_Basin", "latitude": 33.8366, "longitude": -117.9143, "population": 350000},

    # Bay Area
    {"city_name": "San_Francisco", "region_name": "Bay_Area", "latitude": 37.7749, "longitude": -122.4194, "population": 815000},
    {"city_name": "San_Jose", "region_name": "Bay_Area", "latitude": 37.3382, "longitude": -121.8863, "population": 1000000},
    {"city_name": "Oakland", "region_name": "Bay_Area", "latitude": 37.8044, "longitude": -122.2711, "population": 440000},

    # Central Valley
    {"city_name": "Sacramento", "region_name": "Central_Valley", "latitude": 38.5816, "longitude": -121.4944, "population": 525000},
    {"city_name": "Fresno", "region_name": "Central_Valley", "latitude": 36.7378, "longitude": -119.7871, "population": 545000},
    {"city_name": "Bakersfield", "region_name": "Central_Valley", "latitude": 35.3733, "longitude": -119.0187, "population": 410000},

    # Inland Empire
    {"city_name": "Riverside", "region_name": "Inland_Empire", "latitude": 33.9806, "longitude": -117.3755, "population": 330000},
    {"city_name": "San_Bernardino", "region_name": "Inland_Empire", "latitude": 34.1083, "longitude": -117.2898, "population": 220000},

    # San Diego
    {"city_name": "San_Diego", "region_name": "San_Diego", "latitude": 32.7157, "longitude": -117.1611, "population": 1400000},
]

api_url = "https://archive-api.open-meteo.com/v1/archive"

# store hourly series for each city
temp_series = []
rh_series = []
cloud_series = []
rad_series = []

for i, c in enumerate(cities, start=1):
    name = c["city_name"]
    print(f"[{i}/{len(cities)}] pulling weather for {name}")

    params = {
        "latitude": c["latitude"],
        "longitude": c["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,cloud_cover,shortwave_radiation",
        "timezone": "UTC",
    }

    data = None
    for attempt in range(5):
        try:
            r = requests.get(api_url, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)

    hourly = data.get("hourly", {})
    if "time" not in hourly:
        raise RuntimeError(f"no hourly data for {name}")

    t = pd.to_datetime(hourly["time"], utc=True)

    temp_series.append(pd.Series(hourly.get("temperature_2m"), index=t, name=name))
    rh_series.append(pd.Series(hourly.get("relative_humidity_2m"), index=t, name=name))
    cloud_series.append(pd.Series(hourly.get("cloud_cover"), index=t, name=name))
    rad_series.append(pd.Series(hourly.get("shortwave_radiation"), index=t, name=name))

    time.sleep(0.2) # don't spam API

# align everything
temp_df = pd.concat(temp_series, axis=1).sort_index()
rh_df = pd.concat(rh_series, axis=1).sort_index()
cloud_df = pd.concat(cloud_series, axis=1).sort_index()
rad_df = pd.concat(rad_series, axis=1).sort_index()

common_times = temp_df.index.intersection(rh_df.index).intersection(cloud_df.index).intersection(rad_df.index)

temp_df = temp_df.loc[common_times]
rh_df = rh_df.loc[common_times]
cloud_df = cloud_df.loc[common_times]
rad_df = rad_df.loc[common_times]

# population weights
pop_by_city = pd.Series({c["city_name"]: c["population"] for c in cities}, dtype="float64")
state_weights = pop_by_city / pop_by_city.sum()

meta = pd.DataFrame(cities)

region_weights = {}
for region, group in meta.groupby("region_name"):
    p = group.set_index("city_name")["population"].astype("float64")
    region_weights[region] = p / p.sum()

def pop_weighted_avg(df, weights):
    cols = [c for c in df.columns if c in weights.index]
    w = weights[cols]
    return (df[cols].astype("float64") * w).sum(axis=1)

# region features
region_cols = {}
for region, w in region_weights.items():
    region_cols[f"{region}__temperature_c"] = pop_weighted_avg(temp_df, w)
    region_cols[f"{region}__relative_humidity_pct"] = pop_weighted_avg(rh_df, w)
    region_cols[f"{region}__cloud_cover_pct"] = pop_weighted_avg(cloud_df, w)
    region_cols[f"{region}__ssrd_proxy_wm2"] = pop_weighted_avg(rad_df, w)

region_features = pd.DataFrame(region_cols, index=common_times)

# statewide (CA)
state_features = pd.DataFrame({
    "CA__temperature_c": pop_weighted_avg(temp_df, state_weights),
    "CA__relative_humidity_pct": pop_weighted_avg(rh_df, state_weights),
    "CA__cloud_cover_pct": pop_weighted_avg(cloud_df, state_weights),
    "CA__ssrd_proxy_wm2": pop_weighted_avg(rad_df, state_weights),
}, index=common_times)

# city-level flat features
city_features = pd.concat({
    "temperature_c": temp_df,
    "relative_humidity_pct": rh_df,
    "cloud_cover_pct": cloud_df,
    "ssrd_proxy_wm2": rad_df,
}, axis=1)

city_features.columns = [f"{feat}__{city}" for feat, city in city_features.columns]

# save everything
city_features.to_csv(city_out)
region_features.to_csv(region_out)
state_features.to_csv(state_out)

print("done. wrote city, region, and CA feature csvs")
print("note: ssrd_proxy_wm2 = open-meteo shortwave radiation")