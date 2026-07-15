# --- Pandas ---
import pandas as pd

print('PANDAS\n')

# Pandas Q1

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

print(f'First Three Rows:\n{df.head(3)}')
print(f'Shape of df:\n {df.shape}')
print(f'Data Types of Columns:\n{df.dtypes}')

# Pandas Q2

students_passed = df[(df['passed'] == True) & (df['grade'] > 80)]
print(f'Students Who Passed With Grade Above 80:\n{students_passed}')

# Pandas Q3

df['grade_curved'] = df['grade'] + 5
print(f'Grade Curved:\n{df}')

# Pandas Q4

df['name_upper'] = df['name'].str.upper()
print(df.loc[0:, ['name', 'name_upper']])

# Pandas Q5

df_group = df.groupby('city').agg({'grade': 'mean'})
print(df_group)

# Pandas Q6

df = df.replace({'Austin': 'Houston'})
print(df.loc[0:, ['name', 'city']])

# Pandas Q7

df = df.sort_values(by=['grade'], ascending=False)
print(df.head(3))

# --- NumPy ---
import numpy as np

print('\nNumPy\n')

# NumPy Q1

array = np.array([10, 20, 30, 40, 50])
print(f'Shape: {array.shape}, Dtype: {array.dtype}, ndim: {array.ndim}')

# NumPy Q2

arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print(f'Shape: {arr.shape}, Size: {arr.size}')

# NumPy Q3

arr_slice = arr[0:2, 0:2]
print(f'Sliced Array:\n{arr_slice}')

# NumPy Q4

zeroes_array = np.zeros((3, 4))
ones_array = np.ones((2,5))
print(zeroes_array)
print(ones_array)

# NumPy Q5

arr5 = np.arange(0, 50, 5)
print(f'Array: {arr5}, Shape: {arr5.shape}, Mean: {np.mean(arr5)}, Sum: {np.sum(arr5)}, Standard Deviation: {np.std(arr5)}')

# NumPy Q6

arr6 = np.random.normal(0, 1, 200)
print(f'Mean: {np.mean(arr6)}, Standard Deviation: {np.std(arr6)}')

# --- Matplotlib ---
import matplotlib.pyplot as plt

# Matplotlib Q1

x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.plot(x, y)
plt.title('Squares')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Matplotlib Q2

subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]

plt.bar(subjects, scores)
plt.title('Subject Scores')
plt.xlabel('Subjects')
plt.ylabel('Scores')
plt.show()

# Matplotlib Q3

x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.scatter(x1, y1, colorizer='blue')
plt.scatter(x2, y2, colorizer='red')
plt.xlabel('Xs')
plt.ylabel('Ys')
plt.legend(['x1 & y1', 'x2 & y2'])
plt.show()

# Matplotlib Q4

plt.subplot(1, 2, 1)
plt.plot(x, y)
plt.title('Squares')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(1, 2, 2)
plt.bar(subjects, scores)
plt.title('Subject Scores')
plt.xlabel('Subjects')
plt.ylabel('Scores')

plt.tight_layout()
plt.show()

# --- Descriptive Stats ---

# Descriptive Stats Q1

data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]

data_mean = np.mean(data)
data_median = np.median(data)
data_variance = np.var(data)
data_standard_deviation = np.std(data)

print(f'Descriptive Stats Q1:\nMean: {data_mean}, Median: {data_median}, Variance: {data_variance}, Standard Deviation: {data_standard_deviation}\n')

# Descriptive Stats Q2

dsq2_values = np.random.normal(65, 10, 500)

plt.hist(dsq2_values, bins=20)
plt.title('Distribution of Scores')
plt.xlabel("Scores")
plt.ylabel("Frequency")
plt.show()

# Descriptive Stats Q3

group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.boxplot([group_a, group_b], tick_labels=['Group A', 'Group B'])
plt.title('Score Comparison')
plt.show()

# Descriptive Stats Q4

normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

plt.subplot(1, 2, 1)
plt.boxplot(normal_data)
plt.title('Normal')

plt.subplot(1, 2, 2)
plt.boxplot(skewed_data)
plt.title('Exponential')
plt.suptitle('Distribution Comparison')
plt.show()

# The exponential distribution is more skewed. A median statistic would best represent the normal distribution while 
# mean would best represent the exponential example.

# Descriptive Stats Q5
from scipy import stats

data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]
datasets = [data1, data2]

dataset = 1
for d in datasets:
    print(f'data{dataset}: \nmean: {np.mean(d)}, median: {np.median(d)}, mode: {stats.mode(d).mode}\n')
    dataset += 1

# The median and mean are different for data2 because the final number, 150, skews the dataset. While this dataset 
# contains the same number of values as data1, the final number in data2 drives up the mean value.

# --- Hypothesis Testing ---

# Hypothesis Testing Q1
from scipy.stats import ttest_ind

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

ttest = ttest_ind(group_a, group_b)
print(f'Hypotheses Q1\nt-statistic: {ttest.statistic}, p-value: {ttest.pvalue}\n')

# Hypothesis Testing Q2

alpha = 0.05

if ttest.pvalue > alpha:
    print('This is not statistically significant.\n')
elif ttest.pvalue <= alpha:
    print('The result is statistically significant\n')

# Hypothesis Testing Q3

before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

paired_ttest = stats.ttest_rel(before, after)
print(f'Hypotheses Q3\nt-statistic: {paired_ttest.statistic}, p-value: {paired_ttest.pvalue}\n')

# Hypothesis Testing Q4

scores = [72, 68, 75, 70, 69, 74, 71, 73]

one_test = stats.ttest_1samp(scores, popmean=70)
print(f'Hypotheses Q4\nt-statistic: {one_test.statistic}, p-value: {one_test.pvalue}\n')

# Hypothesis Testing Q5

one_tailed = stats.ttest_ind(group_a, group_b, alternative='less')
print(f'Hypotheses Q5\np-value: {one_tailed.pvalue}\n')

# Hypothesis Testing Q6

conclusion = """The result of Q1 is that there is a significant difference between
Group A and Group B. This result rejects the null hypotheses. The difference moves
negative as the mean of Group A is less than the mean of Group B. This is not left
to chance as these numbers are set."""

print(conclusion)

# --- Correlation ---

# Correlation Q1

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

corr_matrix = np.corrcoef(x, y)
print(corr_matrix)
print(corr_matrix[0][1])
print('\n')

# I expected a 1:1 correlation as y = x * 2

# Correlation Q2
from scipy.stats import pearsonr

x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

pr = pearsonr(x, y)
print(f'Correlation Q2\nCorrelation Coefficient: {pr.statistic}, P-value: {pr.pvalue}\n')

# Correlation Q3

people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
df_corr = df.corr()
print(df_corr)

# Correlation Q4

x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

plt.scatter(x, y)
plt.title('Negative Correlation')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Correlation Q4
import seaborn as sns

sns.heatmap(df_corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# --- Pipelines ---

# Pipelines Q1


def create_series(array: list) -> pd.Series:
    return pd.Series(array, name='values')


def clean_data(data_series: pd.Series) -> pd.Series:
    return data_series.dropna()


def summarize_data(clean_series: pd.Series) -> dict:
    clean_dict = {}
    clean_dict['mean'] = clean_series.mean()
    clean_dict['median'] = clean_series.median()
    clean_dict['std'] = clean_series.std()
    clean_dict['mode'] = clean_series.mode()[0]
    return clean_dict


def data_pipeline(arr: list) -> dict:
    series = create_series(arr)
    series = clean_data(series)
    series_dict = summarize_data(series)
    return series_dict


arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

data_dict = data_pipeline(arr)

for key, value in data_dict.items():
    print(f'{key}:{value}')
