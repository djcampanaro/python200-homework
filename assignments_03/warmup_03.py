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
from sklearn.multiclass import OneVsRestClassifier

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

preds_2 = knn2.predict(X_test_scaled)

print("Accuracy: ", accuracy_score(y_test, preds_2))
print(classification_report(y_test, preds_2))

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
plt.clf()

# The pair of species the model most often confuses are versicolor and virginica

# --- The sklearn API: Decision Trees ---

# Q1

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
y_dt_preds = dt.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_dt_preds))
print(classification_report(y_test, y_dt_preds))

# The accuracy of the Decision Tree is a little worse than the KNN unscaled. It sits about halfway between the scaled and 
# non-scaled KNN figures.

# Given that Decision Trees don't rely on distance calculations, the scaled vs. unscaled data does not affect the result.

# --- Logistic Regression and Regularization ---

# Q1

log_reg_small = OneVsRestClassifier(LogisticRegression(C=0.01, max_iter=1000, solver="liblinear"))
log_reg_med = OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
log_reg_large = OneVsRestClassifier(LogisticRegression(C=100, max_iter=1000, solver="liblinear"))

log_reg_small.fit(X_train_scaled, y_train)
log_reg_med.fit(X_train_scaled, y_train)
log_reg_large.fit(X_train_scaled, y_train)

def find_sum_coefficients(log):
    sum = 0
    for l in log.estimators_:
        sum += np.abs(l.coef_).sum()
    return sum

print("C = .01 coefficient sum: ", find_sum_coefficients(log_reg_small))
print("C = 1.0 coefficient sum: ", find_sum_coefficients(log_reg_med))
print("C = 100 coefficient sum: ", find_sum_coefficients(log_reg_large))

# As the value of C increases the total coefficient magnitude also increases. The lower C does not even break a total 
# coefficient sum of two, while the others are closer to 12.5 and 38 respectively. This tells me that with the correctly 
# balanced C, regularization is keeping the coefficients in check by not allowing any factor(s) from having a more weighted 
# affect on the model.

# --- PCA ---

digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting

# Q1

print(X_digits.shape)
print(images.shape)

for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(images[i], cmap='gray_r');
    plt.axis('off');
    plt.title(i)
plt.savefig('outputs/sample_digits.png')
plt.clf()

# Q2

pca = PCA()
pca.fit(X_digits)
scores = pca.transform(X_digits)
print(scores.shape)
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)  # c = color array
plt.colorbar(scatter, label='Digit')
plt.savefig('outputs/pca_2d_projection.png')
plt.clf()

# Yes, it does appear that the same-digit images tend to cluster together in this 2D space.

# Q3

variance_vs_numcomp = np.cumsum(pca.explained_variance_ratio_)

plt.plot(variance_vs_numcomp, range(len(variance_vs_numcomp)))
plt.savefig('outputs/pca_variance_explained.png')
plt.clf()

# Approximately 13 components are needed to explain 80% of the variance.

# Q4


def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)


n_components = [2, 5, 15, 40]
fig, axes = plt.subplots(5, 5)

for x in range(5):
    # Original
    ax = axes[0, x]
    ax.imshow(images[x], cmap='gray_r')
    ax.axis('off')
    for n in range(len(n_components)):
        reconstruct = reconstruct_digit(x, scores, pca, n_components[n])
        ax = axes[n+1, x]
        ax.imshow(reconstruct, cmap='gray_r')
        ax.axis('off')

plt.tight_layout()
plt.savefig('outputs/pca_reconstructions.png')

# The digits become clearly visible at n=40. This appears to be close to the point where the variance curve levels off.
