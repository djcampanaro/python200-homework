import requests
import pandas as pd
import joblib
import json
import sklearn
import sys
import os
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, 
    classification_report,
    RocCurveDisplay,
    roc_curve
)

# Step 1: Fetch the Data

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 41.55,
    "longitude": -73.59,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
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
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)
print(df.describe())

# Step 2: Engineer Labels

def label_running_day(row):
    return int(
        7 <= row["temperature_2m_max"] <= 26
        and row["temperature_2m_min"] >= 4      # adjusted to slightly warmer as a personal preference
        and row["precipitation_sum"] < 6.0      # plenty of rainy days and a little bit more is fine
        and row["wind_speed_10m_max"] < 30
    )

df["good_for_running"] = df.apply(label_running_day, axis=1)

print(df["good_for_running"].value_counts())
print(f"\nFraction of good days: {df['good_for_running'].mean():.2f}")

# 115/365 days are labeled as good for running. Yes, this does seem reasonable for this location.

# Step 3: Train and Tune

FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

X = df[FEATURES]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42)),
])

param_grid = {
    "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc'
)

grid_search.fit(X_train, y_train)

best_lr_pipe = grid_search.best_estimator_

y_pred  = best_lr_pipe.predict(X_test)
y_probs = best_lr_pipe.predict_proba(X_test)[:, 1]

test_auc = roc_auc_score(y_test, y_probs)

print(f'Best C:      {grid_search.best_params_["clf__C"]}')
print(f'Best CV AUC: {grid_search.best_score_:.3f}')
print(classification_report(y_test, y_pred))
print(f'Test AUC:    {test_auc:.3f}')
print('\n')

fpr, tpr, thresholds = roc_curve(y_test, y_probs)

fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax, name=f"Logistic Regression (AUC={test_auc:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label='Random')
ax.set_title('Weather ROC')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/weather_roc.png')
plt.close()

# Step 4: Reflect on Evaluation

# The AUC tells of a high quality model, ranking above the 90th percentile. This is surprisingly 
# better than I'd expected.
# The precision and recall show that a false negative is more common. This means that the model more 
# accurate with catching negative days at the cost of mislabeling days that would be good for running. 
# I would prefer to have more false positives than false negatives as one can always turn back in the 
# weather doesn't suit them. However, missing good days for running would be more disappointing.
# I would set the threshold lower than 0.5 as I would rather catch as many good days for running even 
# if that means more false positives.

# Step 5: Save the Model

joblib.dump(best_lr_pipe, 'models/weather_classifier.pkl')

metadata = {
    "python_version":   sys.version,
    "sklearn_version":  sklearn.__version__,
    "features":         FEATURES,
    "best_params":      grid_search.best_params_,
    "test_auc":         round(test_auc, 4),
    "city":             "Kingston, NY (lat 41.55, lon -73.59)",
    "thresholds":       {
                            "max-temp":          7-26, #Celcius
                            "min-temp":          4,
                            "precipitation-max": 6,    #mm
                            "wind-max":          30    #km pr hr 
                        }
}

with open('models/weather_classifier_metadata.json', 'w') as file:
    json.dump(metadata, file, indent=2)

print("Model and metadata saved to models/")
