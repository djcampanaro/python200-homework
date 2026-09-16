import joblib
import json
import os
import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI
from prefect import task, flow
from prefect.logging import get_run_logger
from supabase import create_client

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

with open("models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)
FEATURES = metadata["features"]

SYSTEM_PROMPT = (
    "You are writing a one-sentence running recommendation for a daily weather summary app. "
    "You will receive weather conditions for a single day along with a machine learning prediction "
    "about whether the day is good for running and its confidence in its prediction. "
    "Write exactly one sentence and be direct, practical, and specific to the conditions. "
    "Do not use bullet points, headers, or phrases like 'Based on the data'."
    "Do not write a run-on sentence."
)

LATITUDE  = 41.55
LONGITUDE = -73.59

response = (supabase.table("weather_raw").delete().neq("date", "2024-01-01").execute())

@task(retries=2, retry_delay_seconds=10)
def extract() -> list:
    logger = get_run_logger()
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "daily": FEATURES,
            "timezone": "America/New_York",
    }
    response = requests.get(url, params=params)
    response.raise_for_status()

    daily = response.json()['daily']

    records = [
            {
                    "date":                 daily["time"][i],
                    "temperature_2m_max":   daily["temperature_2m_max"][i],
                    "temperature_2m_min":   daily["temperature_2m_min"][i],
                    "precipitation_sum":    daily["precipitation_sum"][i],
                    "wind_speed_10m_max":   daily["wind_speed_10m_max"][i],
            }
            for i in range(len(daily["time"]))
    ]

    logger.info(f"Extracted {len(records)} daily records from Open-Meteo")
    return records

@task(retries=2, retry_delay_seconds=5)
def load_raw(records: list) -> None:
    logger = get_run_logger()
    response = (
        supabase.table('weather_raw')
        .upsert(records, on_conflict="date")
        .execute()
    )
    logger.info(f'Upserted {len(response.data)} rows of raw data to weather_raw')

@task
def transform(records: list) -> list:
    logger = get_run_logger()
    presently_enriched = {
        r['date'] for r in supabase.table('weather_enriched').select('date').execute().data
    }
    to_enrich = [r for r in records if r['date'] not in presently_enriched]

    if not to_enrich:
        print(f'All {len(records)} records have been processed and added to weather_enriched')
        return[]

    clf = joblib.load('models/weather_classifier.pkl')
    df = pd.DataFrame(to_enrich)
    X = df[FEATURES]

    predictions = clf.predict(X)
    probabilities = clf.predict_proba(X)[:,1]

    logger.info(f'ML classification process completed. There are {int(predictions.sum())} good running days out of {len(predictions)}')

    enrich_records = [{
        'date': to_enrich[i]['date'],
        'good_for_running': bool(predictions[i]),
        'confidence': round(float(probabilities[i]), 3),
        'llm_summary': None,
    } for i in range(len(predictions))]

    for i, record in enumerate(enrich_records):
        raw_data = to_enrich[i]
        user_message = (
            "The weather info for the day along with the prediction and confidence of the model:\n"
            f"Date: {raw_data['date']}\n"
            f"High temp: {raw_data['temperature_2m_max']}\n"
            f"Low temp: {raw_data['temperature_2m_min']}\n"
            f"Precipitation amount {raw_data['precipitation_sum']}\n"
            f"Wind speed: {raw_data['wind_speed_10m_max']}"
            f"Model prediction: {'good for running' if record['good_for_running'] else 'not good for running'}"
            f"Model confidence level: {record['confidence']:.0%}"
        )
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=100,
            )
            record['llm_summary'] = response.choices[0].message.content.strip() or 'Recommendation unavailable.'
        except Exception as e:
            logger.exception(f"LLM unable to process {record['date']}.\nMessage: {e}")
            record['llm_summary'] = 'Recommendation unavailable.'    

        if (i + 1) % 50 == 0:
            logger.info(f'{i + 1} / {len(enrich_records)} records completed...')     
    logger.info(f'Transform completed with {len(enrich_records)} records enriched.')
    return enrich_records

@task(retries=2, retry_delay_seconds=5)
def load_enriched(enriched_records: list) -> None:
    logger = get_run_logger()
    if not enriched_records:
        logger.info(f'No records to upload.')
        return

    response = (
        supabase.table('weather_enriched')
        .upsert(enriched_records, on_conflict="date")
        .execute()
    )
    logger.info(f'Upserted {len(response.data)} rows of enriched data to weather_enriched')

@flow(log_prints=True)
def etl_pipeline():
    logger = get_run_logger()
    raw_data = extract()
    load_raw(raw_data)
    enriched_data = transform(raw_data)
    load_enriched(enriched_data)

    logger.info('Pipeline complete. All raw records processed and uploaded to enriched table.')

if __name__ == '__main__':
    etl_pipeline()
