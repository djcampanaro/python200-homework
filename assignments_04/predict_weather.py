import joblib
import json
import pandas as pd

# Task 1: Load and Verify

clf = joblib.load('models/weather_classifiesr.pkl')

with open('models/weather_classifier_metadata.json', 'r') as file:
    metadata = json.load(file)

print(f"City: {metadata['city']}")
print(f"Features: {metadata['features']}")
print(f"Test AUC: {metadata['test_auc']}")
print("\n")

# Task 2: Predict on New Data

new_data = pd.DataFrame({
    'temperature_2m_max': [23, 15, 7, 10, 5, 8, 0], 
    'temperature_2m_min': [15, 8, 4, 3, -4, -1, -5], 
    'precipitation_sum': [0, 4, 6, 5, 9, 5, 4], 
    'wind_speed_10m_max': [10, 20, 30, 31, 40, 35, 45]
})

preds = clf.predict(new_data)
probs = clf.predict_proba(new_data)[:, 1]

for i, (pred, prob) in enumerate(zip(preds, probs)):
    label = "good for running" if pred == 1 else "skip"
    print(f"Input feature values: \n{new_data.loc[i, :]}")
    print(f"Predicted label: {label}")
    print(f"Model confidence: {prob:.2f})")
    print("\n")

# Task 3: Reflect

# The borderline case I picked had only a 0.26 confidence rate that it was a running 
# day. So, it ended up as a false negative. If the model showed 0.52, I would consider 
# that a good running day despite the tossup for the model.
# If predict_weather was run before running train_weather_classifier, it would throw a 
# FileNotFoundError. A helpful message would ask the user to esure the files were in the 
# correct directory or run train_weather_classifier in order to create them.
# In order to accommodate a daily weather check to classify the following day's weather, 
# predict_weather would need a call into the forecast site's API or a scraper file. It 
# would need to extract the same features as exist in the model in order to create the 
# prediction for the next day.
