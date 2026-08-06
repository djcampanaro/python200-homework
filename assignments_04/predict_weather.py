import joblib
import json
import pandas as pd

# Task 1: Load and Verify

clf = joblib.load('models/weather_classifier.pkl')

with open('models/weather_classifier_metadata.json', 'r') as file:
    metadata = json.load(file)

print(f"City: {metadata['city']}")
print(f"Features: {metadata['features']}")
print(f"Test AUC: {metadata['test_auc']}")
print("\n")

# Task 2: Predict on New Data

new_data = [
    [23, 15, 0, 10],
    [15, 8, 4, 20],
    [7, 4, 6, 30], #borderline case
    [10, 3, 5, 31],
    [5, -4, 9, 40],
    [8, -1, 5, 35],
    [0, -5, 4, 45],
]

new_df = pd.DataFrame(new_data, columns=metadata['features'])

preds = clf.predict(new_df)
probs = clf.predict_proba(new_df)[:, 1]

for i, (pred, prob) in enumerate(zip(preds, probs)):
    label = "good for running" if pred == 1 else "skip"
    print(f"Input feature values: \n{new_df.loc[i, :]}")
    print(f"Predicted label: {label}")
    print(f"Model confidence: {prob:.2f}")
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
