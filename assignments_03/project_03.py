import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
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

# There is definitely a visual difference between the two classes, but I would not say it is dramatic. The medians appear to be so low that there are 
# many points that fall outside the boxplots. In some of these cases the highest outlier is in the non-spam class. The boxplot for the non-spam class 
# is basically a line in each of the plots, showing most of the examples fall within a very small range.

# The heavy skew toward zero for most of the word-frequency features tells me that the data may not be as varied as would be ideal for creating a robust model. Looking at the word frequency for 'free', 90% of the non-spam emails have zero occurrences. 

free = spam_v_ham['word_freq_free'][spam_v_ham['spam_label'] == 0].apply(lambda x: 0 if x == 0 else np.nan).dropna()
print(free.count())
