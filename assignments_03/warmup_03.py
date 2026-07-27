import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# --- Preprocessing ---

# Q1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42, stratify=y)
print('Shape of X_train: ', X_train.shape)
print('Shape of X_test: ', X_test.shape)
print('Shape of y_train: ', y_train.shape)
print('Shape of y_test: ', y_test.shape)

# Q2

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

for col in X_train_scaled:
    print(col.mean())

# You fit the scaler on X_train only to prevent data leakage by not allowing the mean and standard deviation of each 
# feature to be influenced by the test set.

# --- KNN ---

# Q1

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

preds = knn.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# Q2

knn2 = KNeighborsClassifier(n_neighbors=5)
knn2.fit(X_train_scaled, y_train)

preds = knn2.predict(X_test_scaled)

print("Accuracy: ", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# Scaling hurts the performance. The units of this data all fall within a similar range and are already comprable. Scaling 
# the data can minimize the natural separation.

# Q3

knn = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn, X_train, y_train, cv=5)

print(cv_scores)
print(f'Mean: {cv_scores.mean():.3f}')
print(f'Std: {cv_scores.std():.3f}')

# This result is more trustworthy than a single train/test split. Instead of using a single split, this method creates 
# multiple folds in the training data. One fold is used to evaluate the others which are used for training. This is 
# repeated for each fold and then an average is taken. That average will provide more accuracy than the single test of 
# the train/test split.

# Q4

k_values = [1, 3, 5, 7, 9, 11, 13, 15]

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores =  cross_val_score(knn, X_train, y_train, cv=5)
    print(f'k={k:2d}, mean={scores.mean():.3f}')

# I would use k=5 in this case. 5 and 7 both have the highest mean cv score, so either works well. However, I would rather
# use the lowever of the two as it is the first occurrence to strike the balance between underfitting and overfitting. I 
# prefer to use than than move up to the next one that leans closer to overfitting.

# --- Classifier Evaluation ---

# Q1

cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(colorbar=False)
plt.title('Iris Confusion Matrix')
plt.savefig('outputs/knn_confusion_matrix.png')

# The pair of species the model most often confuses are versicolor and virginica

# --- The sklearn API: Decision Trees ---

# Q1



# --- Logistic Regression and Regularization ---

# Q1



# --- PCA ---

digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting

# Q1



# Q2



# Q3



# Q4


