# --- Supabase Connection ---

# Connection Q1

# The two pieces of information supabase-py needs to connect to my project are the project 
# url and API key (anon key). They are found on the Supabase dashboard under Project Settings, 
# then API. They should never be hardcoded in Python script because storing these in a public 
# repository opens them up to being scrapped by automatic bots within minutes.

# Connection Q2

import os

from dotenv import load_dotenv
from supabase import create_client

def get_client():
    load_dotenv()

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    if SUPABASE_URL and SUPABASE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        return supabase
    else:
        print('You are missing the Supabase URL or Key. Check your .env file to make sure they are both there.')

supabase = get_client()

# Connection Q3

# Row Level Security allows the developer to create fine-grain access policies to the Supabase 
# tables. We disable it for this course because it creates a complexity in development that is 
# not needed for this course. In real life practical use, it would be best applied to a 
# database where a user is only allowed to access their own information and not another user's 
# or other company information. This is important for sensitive accounts such as financial 
# institutions or health organizations that deal with lots of personal data.

# --- supabase-py CRUD ---

# CRUD Q1

def insert_test_record(supabase):
    record = {
        "date":               "2026-09-07",
        "temperature_2m_max": 81.0,
        "temperature_2m_min": 65.0,
        "precipitation_sum":  0.0,
        "wind_speed_10m_max": 8.5,
    }

    response = supabase.table("weather_raw").insert(record).execute()
    print(response.data)

# insert_test_record(supabase)

# Running this function again with the same data will cause an error due to the primary key 
# already existing. In order to make the call safe to run multiple times, insert should be 
# changed to upsert which will only add a row when a primary key doesn't exist, and will 
# update rows in place for primary keys that do exist.

# CRUD Q2

def get_records_by_date_range(supabase, start, end):
    response = (
        supabase.table("weather_raw")
        .select("*")
        .gte("date", start)
        .lte("date", end)
        .execute()
    )
    return response.data

date_range = get_records_by_date_range(supabase, '2026-01-01', '2026-09-30')
print(date_range)

# CRUD Q3

# Insert will add new rows to the table with unique primary keys. Upsert edits the table in 
# place based on existing or new primary keys. Insert is good to use in situations where you 
# want to prevent overwriting a record due to a duplicate primary key. An example would be a 
# username for an app as a primary key. Upsert is good for situations where you rerun a 
# pipeline with data that may be new or changed. An example would be an app that regularly 
# pulls legislation documents for a state over a full session.

def safe_upsert(supabase, records):
    response = (
        supabase.table("weather_raw")
        .upsert(records, on_conflict="date")
        .execute()
    )

    print(f"Upserted {len(response.data)} rows into weather_raw")

# --- Idempotency ---

# Idempotency Q1

# Idempotency matters for a pipeline because it prevents an error from being raised during 
# an update/insert of the table. It allows large amounts of data to be updated in place by 
# its primary keys allowing the table to remain consistent over time. If a non-idempotent 
# pipeline were to encounter an error halfway through and be restarted, it would immediately 
# encounter an error as the first record would have already been added to the table creating 
# a primary key conflict with the data being uploaded.
