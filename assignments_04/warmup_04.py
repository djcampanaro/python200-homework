import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
    f1_score,
)
import joblib

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Synthetic dataset — binary classification, two informative features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# --- ROC and AUC ---

# Q1

log = LogisticRegression(max_iter=1000, random_state=42)
log.fit(X_train, y_train)
y_preds_lr = log.predict(X_test)
y_probs_lr = log.predict_proba(X_test)[:, 1]
log_auc = roc_auc_score(y_test, y_probs_lr)
print(f"Logistic Regression AUC: {log_auc:.3f}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_preds_knn = knn.predict(X_test_scaled)
y_probs_knn = knn.predict_proba(X_test_scaled)[:, 1]
knn_auc = roc_auc_score(y_test, y_probs_knn)
print(f"KNN AUC: {knn_auc:.3f}")
print('\n')

# The scaled KNN model has the higher AUC. This model better separates the two classes by 
# creating a good balance between TPR and FPR.

# Q2

fpr, tpr, thresholds = roc_curve(y_test, y_probs_lr)
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, y_probs_knn)

fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax, name=f"Logistic Regression (AUC={log_auc:.2f})")
RocCurveDisplay(fpr=fpr_knn, tpr=tpr_knn).plot(ax=ax, name=f"KNN k=5 (AUC={knn_auc:.2f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label='Random')
ax.set_title('ROC Comparison - Logistic Regression vs. KNN')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/roc_comparison.png')
plt.close()

print('ROC comparison chart created.')
print('\n')

# The KNN model has the lower FPR at TPR = 0.80. This model is likely to produce fewer false 
# positives while catching 80% of true positives.

# Q3

fpr, tpr, thresholds = roc_curve(y_test, y_probs_lr)

f1_scores = []
for threshold in thresholds:
    y_pred = (y_probs_lr >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)

f1_optimum_index = np.argmax(f1_scores)
print('Scores at Optimum')
print('Threshold: ', round(thresholds[f1_optimum_index], 3))
print('TPR: ', tpr[f1_optimum_index])
print('FPR: ', fpr[f1_optimum_index])
print('F1: ', round(f1_scores[f1_optimum_index], 3))
print('\n')

# This threshold is quite a bit lower than 0.5, sitting at 0.276. In a real application, one might 
# choose a threshold lower than 0.5 when willing to accept a larger numver of false positives in 
# order to catch a larger number of True Positives.

# --- GridSearchCV ---

# Q1

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000)),
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

# y_pred  = best_lr_pipe.predict(X_test)
y_probs = best_lr_pipe.predict_proba(X_test)[:, 1]

print(f'Best C:      {grid_search.best_params_["clf__C"]}')
print(f'Best CV AUC: {grid_search.best_score_:.3f}')
print(f'Test AUC:    {roc_auc_score(y_test, y_probs):.3f}')
print('\n')

# I was not sure which C to expect, but a C of 100 matches the data from the lesson. The AUC does not appear to have 
# changed at all compared to the default C=1.0.

# Q2

param_grid_2 = {
    "max_depth": [2, 3, 5, 8, None]
}

grid_search_2 = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid_2,
    cv=5,
    scoring='roc_auc'
)

grid_search_2.fit(X_train, y_train)

best_pipe_2 = grid_search_2.best_estimator_

# y_pred  = best_pipe.predict(X_test)
y_probs_2 = best_pipe_2.predict_proba(X_test)[:, 1]

print(f'Best Max Depth:      {grid_search_2.best_params_["max_depth"]}')
print(f'Best CV AUC: {grid_search_2.best_score_:.3f}')
print(f'Test AUC:    {roc_auc_score(y_test, y_probs_2):.3f}')
print('\n')

# I would bring the decision tree model into further development. AUC is not the only thing I would consider, 
# but the there is a striking differenc between the scores of these two models. Enough of a difference to warrant 
# further exploration of the decision tree model.

# Q3

results = pd.DataFrame(grid_search.cv_results_)
print(
    results[["param_clf__C", "mean_test_score", "std_test_score"]]
    .sort_values("mean_test_score", ascending=False)
    .to_string(index=False)
)
print('\n')

# The top two parameters score very close to each other on mean_test_score. However, C=100 has a slightly lower 
# std_test_score. That is the parameter I would pick because all things being equal, a lower std would improve the model.

# --- joblib ---

# Q1

joblib.dump(best_lr_pipe, 'models/warmup_model.pkl')

loaded_clf = joblib.load("models/warmup_model.pkl")

original_preds = best_lr_pipe.predict(X_test)
loaded_preds   = loaded_clf.predict(X_test)

assert (original_preds == loaded_preds).all(), "Predictions do not match!"
print("Predictions match. Model saved and loaded successfully.")
print('\n')

# If only the logistic regression model was saved, without the scaler, any unscaled data put 
# into the loaded model would provide wrong outputs. There is no error thrown as the loaded 
# model does not know the data is unscaled, but because it was built on scaled data, unscaled 
# data will provide false/misleading results.

# Q2

# --- Simulated prediction script ---

load_clf = joblib.load("models/warmup_model.pkl")

# Three hand-crafted test cases — raw, unscaled data
new_samples = np.array([
    [2.5,  1.2, -0.3,  0.8,  1.0, -0.5,  0.2,  0.9, -1.1,  0.4],
    [-1.0, 0.5,  0.9, -0.7, -0.2,  1.3, -0.8,  0.1,  0.5, -0.3],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
])

load_preds = load_clf.predict(new_samples)
load_probs = load_clf.predict_proba(new_samples)[:, 1]

print('Predicted class: ', load_preds)
print(f'Predicted probabilities: {load_probs}')

# I had expected the row of zeroes to predict a class of 0. I had 
# thought that the zeroes would sit at or below the threshold and 
# would create a predicted probablity closer to 0.5. In using random 
# numbers, I thought the distribution might be more evenly spaced 
# around 0.
