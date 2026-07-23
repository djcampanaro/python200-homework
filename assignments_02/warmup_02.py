# --- scikit-learn API ---

# Q1
import numpy as np
from sklearn.linear_model import LinearRegression

years = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

new_x = np.array([4, 8]).reshape(-1, 1)
model = LinearRegression()
model.fit(years, salary)
y_predicted = model.predict(new_x)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predictions:", y_predicted)

# Q2

x = np.array([10, 20, 30, 40, 50])
print(x.shape)
x2d = x.reshape(-1, 1)
print(x2d.shape)

# scikit-learn requires the x-value to be 2d, and will throw an error if it is 1d. Even when there is a single feature 
# example, it expects two values, num_samples and num_features.

# Q3

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters)

centroids = kmeans.cluster_centers_

print("Cluster Centers:", centroids)
print("Number of Points:", np.bincount(labels))

plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=800, alpha=1, marker='x')
plt.title('K-Means Clusters')
plt.xlabel('Temperature')
plt.ylabel('Layers of Clothing')
plt.savefig('outputs/kmeans_clusters.png', dpi=300)
plt.close()

# --- Linear Regression ---
import os
from sklearn.model_selection import train_test_split

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# Q1

plt.scatter(age, cost, c=smoker, cmap='coolwarm')
plt.title('Medical Cost vs Age')
plt.xlabel('Age')
plt.ylabel('Medical Cost')
plt.savefig('outputs/cost_vs_age.png', dpi=300)
plt.close()

# There are two distinct groups visible in this scatter chart. At every age, the smokers have higher medical costs than 
# non-smokers. The smoker variable drastically changes the overall costs for each age group, driving up the mean. 
# This shows why insurance companies separate the insured by their smoking habits.

# Q2

X_age = age.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X_age, cost, test_size=0.2, random_state=42)
print("X train shape:", X_train.shape)
print("X test shape:", X_test.shape)
print("y train shape:", y_train.shape)
print("y test shape:", y_test.shape)

# Q3

model_age = LinearRegression()
model_age.fit(X_train, y_train)

print("Slope:", model_age.coef_[0])
print("Intercept:", model_age.intercept_)

y_age_predicted = model_age.predict(X_test)

print("RMSE: ", np.sqrt(np.mean((y_age_predicted - y_test) ** 2)))
print("R²: ", model_age.score(X_test, y_test))

# The slope indicates an increase of around $196 in medical costs per year of age between 20 and 65. All other things 
# being equal, a 65-year-old person would spend around $8,820 more than a 20-year-old. However, considering the RMSE, 
# other factors would need to be considered to get a clearer picture of this difference.

# Q4

X_full = np.column_stack([age, smoker])

X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(X_full, cost, test_size=0.2, random_state=42)

model_full = LinearRegression()
model_full.fit(X_full_train, y_full_train)
y_full_predicted = model_full.predict(X_full_test)
print("R²: ", model_full.score(X_full_test, y_full_test))

print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])

# The smoker coefficient shows how much medical costs increase with age at a constant. In this case, smoking increases 
# medical costs by around $14,500 with age constant.

# Q5

plt.scatter(y_full_predicted, y_full_test, color="blue")
# plt.axline([0, 0], [1, 1], color="red", label="Linear fit")
plt.plot(y_full_predicted, y_full_predicted, color='red')
plt.title("Predicted vs Actual")
plt.xlabel("Predicted Cost")
plt.ylabel("Actual Cost")
plt.savefig('outputs/predicted_vs_actual.png', dpi=300)
plt.close()

# The dots above the diagonal are when the model underpredicted the costs as related to age and smoking. The dots below 
# the diagonal are when the model overpredicted the results.
