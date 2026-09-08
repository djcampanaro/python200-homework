# --- Video ---

# 'https://youtu.be/FB4nXVXH7MA'

import os
import requests

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

LATITUDE  = 41.55
LONGITUDE = -73.59

# --- Step 1: Extract ---

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude":   LATITUDE,
    "longitude":  LONGITUDE,
    "start_date": "2023-01-01",
    "end_date":   "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}

response = requests.get(url, params=params)
response.raise_for_status()
daily = response.json()["daily"]
print(f"Fetched {len(daily['time'])} daily records from Open-Meteo")

# --- Step 2: Transform ---

records = [
    {
        "date":               daily["time"][i],
        "temperature_2m_max": daily["temperature_2m_max"][i],
        "temperature_2m_min": daily["temperature_2m_min"][i],
        "precipitation_sum":  daily["precipitation_sum"][i],
        "wind_speed_10m_max": daily["wind_speed_10m_max"][i],
    }
    for i in range(len(daily["time"]))
]

print(f"Prepared {len(records)} records")
print("First record:", records[0])
print("Last record:", records[-1])

# I expected 365 days and did get all 365 days. If the numbers were to differ it may be because weather stations occasionally fail to record the readings on a given day.

# --- Step 3: Load ---

response = (
    supabase.table("weather_raw")
    .upsert(records, on_conflict="date")
    .execute()
)

print(f"Upserted {len(response.data)} rows into weather_raw")

response = (
    supabase.table("weather_raw")
    .select("*")
    .execute()
)

print(len(response.data))

# This tells me that idempotency is in effect as the number of rows stayed the same. Running the code a second time and having 
# the table update in place aligns with the properties of idempotency.

# --- Step 4: Verify ---

count_response = supabase.table("weather_raw").select("date", count="exact").execute()
print(f"Rows in weather_raw: {count_response.count}")

first = supabase.table("weather_raw").select("*").eq("date", "2023-01-01").execute()
last  = supabase.table("weather_raw").select("*").eq("date", "2023-12-31").execute()
print("First record:", first.data)
print("Last record: ", last.data)

date = supabase.table("weather_raw").select("*").eq("date", "2023-07-04").execute()
print("2023-07-04", date.data)
