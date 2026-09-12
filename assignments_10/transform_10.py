# --- Video ---

#

# --- Part 2: Project — The Double-Transform Pipeline ---

import joblib
import json
import os
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client

# Step 1: Incremental Read

with open("models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

response = supabase.table("weather_raw").select("*").execute()
raw_rows = response.data
print(f"Fetched {len(raw_rows)} rows from weather_raw")

enriched_response = supabase.table("weather_enriched").select("date").execute()
already_done = {row["date"] for row in enriched_response.data}

to_classify = [row for row in raw_rows if row["date"] not in already_done]
print(f"Records to classify: {len(to_classify)} (skipping {len(already_done)} already enriched)")

print(f"{len(raw_rows)} raw records exist. {len(already_done)} are already enriched. {len(to_classify)} records will be processed on this run.")

# Step 2: ML Transform

clf = joblib.load("models/weather_classifier.pkl")

FEATURES = metadata["features"]

df = pd.DataFrame(to_classify)
X = df[FEATURES]

predictions = clf.predict(X)
probabilities = clf.predict_proba(X)[:,1]

enrichment_records = []
for i, row in enumerate(to_classify):
    enrichment_records.append({
        "date":             row["date"],
        "good_for_running": bool(predictions[i]),
        "confidence":       round(float(probabilities[i]), 4),
    })

print(f"Good days predicted: {predictions.sum()} / {len(predictions)}")
print(f"Confidence range: {probabilities.min():.2f} - {probabilities.max():.2f}")

# Step 3: LLM Transform

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM_PROMPT = (
    "Write a one-sentence recommendation for a daily weather app. "
    "Based on the machine learning prediction you receive about which days are good for running, "
    "write exactly one sentence that is direct, practical, and specific to the conditions given. "
    "Do not use bullet points, headers, or phrases like 'Based on the data'."
)

def make_user_message(row, good_for_running, confidence):
    prediction_text = "good for running" if good_for_running else "not a good day for running"
    return (
        f"Date: {row['date']}\n"
        f"High: {row['temperature_2m_max']}°C, Low: {row['temperature_2m_min']}°C\n"
        f"Precipitation: {row['precipitation_sum']} mm\n"
        f"Max wind speed: {row['wind_speed_10m_max']} km/h\n"
        f"Model prediction: {prediction_text} (confidence: {confidence:.0%})"
    )

def validate_summary(text):
    text = text.strip()
    if not text:
        return None
    # Reject if more than two sentences (simple heuristic)
    sentences = [s for s in text.split(".") if s.strip()]
    if len(sentences) > 2:
        return None
    return text

for i, record in enumerate(enrichment_records):
    raw_row = next(r for r in to_classify if r['date'] == record['date'])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": make_user_message(
                    raw_row,
                    record["good_for_running"],
                    record["confidence"],
                ),
            },
        ],
        max_tokens=100,
    )

    raw_summary = response.choices[0].message.content.strip()
    summary = validate_summary(raw_summary) or "Recommendation unavailable."
    record["llm_summary"] = summary

    if (i + 1) % 50 == 0:
        print(f"  Enriched {i + 1} / {len(enrichment_records)} records...")

# Step 4: Load

response = (
    supabase.table("weather_enriched")
    .upsert(enrichment_records, on_conflict="date")
    .execute()
)
print(f"Upserted {len(response.data)} rows into weather_enriched")

# Step 5: Verify

total_rows = supabase.table("weather_enriched").select("date", count="exact").execute()
print(f"total number of rows: {total_rows}")

check = supabase.table("weather_enriched").select("*").limit(5).execute()
for row in check.data:
    print(f"{row['date']} | good={row['good_for_running']} | conf={row['confidence']:.2f}")
    print(f"  {row['llm_summary']}")
    print()

# Count how many were classified as good
good_count = (
    supabase.table("weather_enriched")
    .select("date", count="exact")
    .eq("good_for_running", True)
    .execute()
)
print(f"Good-for-running days in weather_enriched: {good_count.count}")

