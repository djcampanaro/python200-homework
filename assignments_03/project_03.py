import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.inspection import DecisionBoundaryDisplay

warnings.filterwarnings("ignore", category=RuntimeWarning)

COLUMN_NAMES = [
    "word_freq_make",        # 0   percent of words that are "make"
    "word_freq_address",     # 1
    "word_freq_all",         # 2
    "word_freq_3d",          # 3   almost never appears
    "word_freq_our",         # 4
    "word_freq_over",        # 5
    "word_freq_remove",      # 6   common in "remove me from this list"
    "word_freq_internet",    # 7
    "word_freq_order",       # 8
    "word_freq_mail",        # 9
    "word_freq_receive",     # 10
    "word_freq_will",        # 11
    "word_freq_people",      # 12
    "word_freq_report",      # 13
    "word_freq_addresses",   # 14
    "word_freq_free",        # 15  classic spam word
    "word_freq_business",    # 16
    "word_freq_email",       # 17
    "word_freq_you",         # 18
    "word_freq_credit",      # 19
    "word_freq_your",        # 20  often high in spam
    "word_freq_font",        # 21  HTML emails
    "word_freq_000",         # 22  "win $ x,000" style offers
    "word_freq_money",       # 23  money related
    "word_freq_hp",          # 24  HP specific
    "word_freq_hpl",         # 25
    "word_freq_george",      # 26  specific HP person
    "word_freq_650",         # 27  area code
    "word_freq_lab",         # 28
    "word_freq_labs",        # 29
    "word_freq_telnet",      # 30
    "word_freq_857",         # 31
    "word_freq_data",        # 32
    "word_freq_415",         # 33
    "word_freq_85",          # 34
    "word_freq_technology",  # 35
    "word_freq_1999",        # 36
    "word_freq_parts",       # 37
    "word_freq_pm",          # 38
    "word_freq_direct",      # 39
    "word_freq_cs",          # 40
    "word_freq_meeting",     # 41
    "word_freq_original",    # 42
    "word_freq_project",     # 43
    "word_freq_re",          # 44  reply threads
    "word_freq_edu",         # 45
    "word_freq_table",       # 46
    "word_freq_conference",  # 47
    "char_freq_;",           # 48  frequency of ';'
    "char_freq_(",           # 49  frequency of '('
    "char_freq_[",           # 50  frequency of '['
    "char_freq_!",           # 51  exclamation marks (often big)
    "char_freq_$",           # 52  dollar sign (money related)
    "char_freq_#",           # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",    # 56  total number of capital letters
    "spam_label"                    # 57  1 = spam, 0 = not spam
]



# Task 1: Load and Explore

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

df = pd.read_csv(BytesIO(response.content), header=None)
df.columns = COLUMN_NAMES
# df.info()

# There are 4601 emails in the dataset. The balance of the two classes is 1813, 39%, for spam and 2788, 61% for non-spam. With so many more examples of 
# non-spam, this may cause the raw accuracy to score high even if the model is not very good as there are fewer opportunities to check emails for the 
# selected categories.

spam_v_ham = df[['word_freq_free', 'char_freq_!', 'capital_run_length_total', 'spam_label']]

for col in spam_v_ham.columns:
    if col == 'spam_label':
        continue
    plt.boxplot(spam_v_ham[col][spam_v_ham['spam_label'] == 0], positions=[0])
    plt.boxplot(spam_v_ham[col][spam_v_ham['spam_label'] == 1], positions=[1])
    plt.title(col)
    plt.xticks([0, 1], ['Ham', 'Spam'])
    plt.savefig(f'outputs/{col}_ham_v_spam.png')

plt.clf()

# There is definitely a visual difference between the two classes, but I would not say it is dramatic. The medians appear to be so low that there are 
# many points that fall outside the boxplots. In some of these cases the highest outlier is in the non-spam class. The boxplot for the non-spam class 
# is basically a line in each of the plots, showing most of the examples fall within a very small range.

# The heavy skew toward zero for most of the word-frequency features tells me that the data may not be as varied as would be ideal for creating a robust model. Looking at the word frequency for 'free', 90% of the non-spam emails have zero occurrences. 

free = spam_v_ham['word_freq_free'][spam_v_ham['spam_label'] == 0].apply(lambda x: 0 if x == 0 else np.nan).dropna()
print(free.count())



# Task 2: Prepare Your Data

X = df.drop('spam_label', axis=1)
y = df['spam_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# After loading in the data and observing the feature scales, I am going to use Standardization to scale the data. 

pca = PCA()
pca.fit(X_train_scaled)

cumu_variance = np.cumsum(pca.explained_variance_ratio_)

plt.plot(cumu_variance, range(len(cumu_variance)))
# plt.savefig('outputs/spam_variance_explained.png')
plt.clf()

min, max = 0, (len(cumu_variance) - 1)
half_variance = round(len(cumu_variance)/2)
if cumu_variance[half_variance] > .9:
    max = half_variance
else:
    min = half_variance

n = 0
for i in range(round((max-min) / 2) + 1):
    if cumu_variance[min] >= .9:
        n = min
        break
    else:
        min += 1
    if cumu_variance[max] <.9:
        n = max + 1
        break
    else:
        max -= 1
print('n: ', n)

X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca  = pca.transform(X_test_scaled)[:, :n]



# Task 3: A Classifier Comparison

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

preds = knn.predict(X_test)

print('KNN Unscaled data:')
print("Accuracy: ", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)

preds_scaled = knn_scaled.predict(X_test_scaled)

knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)

preds_pca = knn_pca.predict(X_test_pca)

print('KNN Scaled data:')
print("Accuracy: ", accuracy_score(y_test, preds_scaled))
print(classification_report(y_test, preds_scaled))
print('KNN PCA data:')
print("Accuracy: ", accuracy_score(y_test, preds_pca))
print(classification_report(y_test, preds_pca))

# The scaled and PCA data are nearly identical in their precision, recall, and F1 scores. The macro and weighted averages 
# match and the accuracies are off by just over two hundredths of a percent.


def decision_tree_calc(dt_list: list) -> None:
    for d in dt_list:
        dt = DecisionTreeClassifier(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        dt_train_y = dt.predict(X_train)
        print(f'Max Depth {d} train accuracy: ', accuracy_score(y_train, dt_train_y))
        dt_preds = dt.predict(X_test)
        print(f'Max Depth {d} test accuracy: ', accuracy_score(y_test, dt_preds))


dt_list = [3, 5, 10, None]
decision_tree_calc(dt_list)

# As the depth increases the accuracy improves for both the training and testing 
# data. However, the gap between the training and testing data's accuracy also grows. 
# With no max-depth specified, the training accuracy is nearly 100% while the testing 
# accuracy lags behind by almost 9 percentage points. These are signs of overfitting.

# In production, I would use the depth of 5. My reasoning is that the accuracy for 
# both training and testing is higher than a depth of 3 while still maintaining a 
# small enough gap to not signal overfitting. The next depth of 10 once again 
# improves the accuracy, but the gap widens to nearly 6 percentage points, moving 
# closer to overfitting.

dt5 = DecisionTreeClassifier(max_depth=5, random_state=42)
dt5.fit(X_train, y_train)
y_dt5_preds = dt5.predict(X_test)

print('Decision Tree Max-depth 5:')
print('Accuracy', accuracy_score(y_test, y_dt5_preds))
print(classification_report(y_test, y_dt5_preds))

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_rf_preds = rf.predict(X_test)

print('Random Forest:')
print('Accuracy: ', accuracy_score(y_test, y_rf_preds))
print(classification_report(y_test, y_rf_preds))

log_reg = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
log_reg.fit(X_train_scaled, y_train)
y_scaled_preds = log_reg.predict(X_test_scaled)

log_reg_2 = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
log_reg_2.fit(X_train_pca, y_train)
y_pca_preds = log_reg_2.predict(X_test_pca)

print('Logistic Regression Scaled Data:')
print("Accuracy: ", accuracy_score(y_test, y_scaled_preds))
print(classification_report(y_test, y_scaled_preds))

print('Logistic Regression PCA Data:')
print("Accuracy: ", accuracy_score(y_test, y_pca_preds))
print(classification_report(y_test, y_pca_preds))

# The model that perfomed best was the Random Forest Classifier. It scored highest on all metrics and report categories. 
# For the KNN classifiers, the non-PCA data performed slightly better than the PCA data. For Logistic Regression, The 
# non-PCA data performed better once again albeit with a slightly larger gap than with KNN. I had thought the PCA data 
# would perform better because minimizing the dimensions would create stronger correlation patterns within the categories. 
# For a spam filter, I don't see accuracy as the best metric for optimization. In my opinion, the best course would be to 
# minimize false positives. 

feature_importances = pd.DataFrame({'columns': X.columns, 'decisiontree': dt5.feature_importances_, 'random_forest': rf.feature_importances_})

top_features_dt = feature_importances.sort_values(by='decisiontree', ascending=False).reset_index()
top_features_rf = feature_importances.sort_values(by='random_forest', ascending=False).reset_index()

print('Decision Tree Top Features: ', top_features_dt.loc[:9, 'columns'])
print('Random Forest Top Features: ', top_features_rf.loc[:9, 'columns'])

labels = top_features_rf.loc[:9, 'columns'].str.replace('_', ' ').str.title()
plt.bar(top_features_rf.loc[:9, 'columns'], top_features_rf.loc[:9, 'random_forest'], tick_label=labels)
plt.title('Random Forest Top Ten Features for Spam')
plt.xlabel('Feature')
plt.ylabel('Ranking')
plt.xticks(rotation='vertical')
plt.subplots_adjust(bottom=0.5)
plt.savefig('outputs/feature_importances.png')
plt.clf()

# The two models agree on six of the top ten features for identifying spam emails. For the most part they align with my 
# intuition as to what one would expect to see in spam emails, including the word 'free', exclamation points and dollar 
# signs, as well as 'capital_run_length_total'. 



# Task 4: Cross-Validation


def cross_validation(classifier, X, classifier_name):
    cv_scores = cross_val_score(classifier, X, y_train, cv=5)
    print(f'{classifier_name}:')
    print(f'Mean: {cv_scores.mean():.3f}')
    print(f'Std: {cv_scores.std():.3f}')


cross_validation(knn, X_train, 'KNN Unscaled data')
cross_validation(knn_scaled, X_train_scaled, 'KNN Scaled data')
cross_validation(knn_pca, X_train_pca, 'KNN PCA data')
cross_validation(dt5, X_train, 'Decision Tree Max-depth 5')
cross_validation(rf, X_train, 'Random Forest')
cross_validation(log_reg, X_train_scaled, 'Logistic Regression Scaled Data')
cross_validation(log_reg_2, X_train_pca, 'Logistic Regression PCA Data')

# The most accurate model is still the Random Forest classifier with a mean of .954. The most stable is the Logistic 
# Regression classifier using PCA data. The top classifier does match the single train/test split, but using 
# cross-validation scores has shrunk the gap between it and the other classifiers' scores.



# Task 5: Building a Prediction Pipeline

random_forest_pipeline = Pipeline([
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

log_reg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
])

random_forest_pipeline.fit(X_train, y_train)
rf_preds = random_forest_pipeline.predict(X_test)

print('Random Forest Pipeline:')
print(classification_report(y_test, rf_preds))

log_reg_pipeline.fit(X_train, y_train)
log_preds = log_reg_pipeline.predict(X_test)

print('Logistic Regression Pipeline:')
print(classification_report(y_test, log_preds))

# My pipelines do not have the same structure. The tree pipeline does not require scaling or dimensionality reduction, 
# so the only step in the pipeline is the classifier itself. Logistic Regression is best utilized with a scaler, so a 
# standard scaler was added to this pipeline. I did not add a PCA step because the non-PCA model performed better in the 
# single train/test split. The practical value in creating this pipelines is cutting down on writing out all the 
# instructions for each step one wants to explore when using these classifiers. These pipelines become something akin to 
# a function where you pass in the data and let it take care of all the necessary steps to create the classifier's output. 
# This is very beneficial to a when handing off or deploying it as the user would be able to use your pipelines with 
# their own data without having to rewrite the whole process.
