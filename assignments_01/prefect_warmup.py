import numpy as np
import pandas as pd

from prefect import task, flow


@task
def create_series(array: np.ndarray) -> pd.Series:
    return pd.Series(array, name='values')


@task
def clean_data(data_series: pd.Series) -> pd.Series:
    return data_series.dropna()


@task
def summarize_data(clean_series: pd.Series) -> dict:
    clean_dict = {}
    clean_dict['mean'] = clean_series.mean()
    clean_dict['median'] = clean_series.median()
    clean_dict['std'] = clean_series.std()
    clean_dict['mode'] = clean_series.mode()[0]
    return clean_dict


@flow
def data_pipeline(arr: np.ndarray) -> dict:
    series = create_series(arr)
    series = clean_data(series)
    series_dict = summarize_data(series)
    return series_dict


def pipeline_flow():

    arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

    data_dict = data_pipeline(arr)

    for key, value in data_dict.items():
        print(f'{key}:{value}')

if __name__ == "__main__":
    pipeline_flow()


"""
1. Prefect is more overhead than is necessary in this example because we are working
with a simple dataset that does not require network calls or caching to run. Using
Prefect creates many steps that go beyond what is necessary to process this data.

2. Realistic scenarios that benefit from Prefect are situations that require larger
datasets with network calls. If the situation benefits from caching slower processes
and retries for failed network calls, Prefect would be of great benefit.
"""
