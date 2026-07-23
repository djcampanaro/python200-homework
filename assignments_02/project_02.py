import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

pd.set_option('future.no_silent_downcasting', True)

# Task 1

# Another necessary variable to set in read_csv is sep= as this csv uses a semicolon as a separator.

df = pd.read_csv('student_performance_math.csv', sep=';')
print('Shape of DataFrame: ', df.shape)
print('First five rows: ', df.head())
print('Data types: ', df.dtypes)

plt.hist(df['G3'], bins=21)
plt.title('Distribution of Final Math Grades')
plt.xlabel('Final Grade')
plt.ylabel('Students')
plt.savefig('outputs/g3_distribution.png', dpi=300)
plt.close()

# Task 2

print('Shape before transformation: ', df.shape)
df2 = df.copy()
df2['G3'] = df2['G3'].apply(lambda x: x if x != 0 else np.nan)
df2 = df2.dropna()
print('Shape after transformation: ', df2.shape)

# Keeping these rows with a G3 equal to 0 would skew the model. The other variables from the students who have a score of 0 in G3 would affect 
# how the model predicts the outcome of other students with similar variables.

print(df2.head())
df2.loc[:, ['schoolsup', 'internet', 'higher', 'activities', 'sex']] = df2[['schoolsup', 'internet', 'higher', 'activities', 'sex']].replace({'yes': 1, 'no': 0, 'F': 0, 'M': 1})
print(df2.head())

r, p = pearsonr(df['absences'], df['G3'])
r2, p2 = pearsonr(df2['absences'], df2['G3'])

print('Correlation before filter: ', r)
print('Correlation after filter: ', r2)

plt.scatter(df['G3'], df['absences'])
plt.title('Original DF G3 vs Absences')
plt.xlabel('G3')
plt.ylabel('Absences')
plt.show()
plt.clf()

plt.scatter(df2['G3'], df2['absences'])
plt.title('Filtered DF G3 vs Absences')
plt.xlabel('G3')
plt.ylabel('Absences')
plt.show()
plt.clf()

students_no_g3 = df[df['G3'] == 0]
plt.scatter(students_no_g3['G1'], students_no_g3['G2'])
plt.title('G1 and G2 of Students with G3=0')
plt.xlabel('G1')
plt.ylabel('G2')
plt.show()
plt.close()

# The students with no G3 score made absences look like a weak predictor because they all had 0 absences. This would skew 
# the results towards a lower number of absences having a negative correlation with G3. When these students are removed, 
# the negative correlation is returned to the higher number of absences. There are a lot of these students who do not have 
# a G2 score either. The low absence rate and lack of G2 scores suggests many of these students may have left school and 
# not taken the G3 rather than scoring an actual 0.

# Task 3

numeric_columns = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'absences', 'freetime', 'goout', 'Walc']
pearson_comparisons = pd.DataFrame(columns=['column', 'pearson_with_G3'])

for col in numeric_columns:
    r, p = pearsonr(df2[col], df2['G3'])
    pearson_comparisons.loc[len(pearson_comparisons)] = [col, r]

pearson_comparisons = pearson_comparisons.sort_values(by='pearson_with_G3')
print(pearson_comparisons)

# failures has the strongest correlation with G3. I find traveltime and freetime to be surprising as they can be directly 
# related, and I would expect having little free time to affect a student's motivation. I am also surprised that the 
# strongest positive linear relationship is nearly the opposite of only the third strongest negative linear relationship. 
# The negatives far outweigh the positives.

age_study = df2.groupby('age').agg({'studytime': 'mean', 'goout': 'mean', 'failures': 'mean', 'absences': 'mean', 'G3': 'median', 'age': 'sum'})
age_study['age'] = age_study['age'].astype('float')
age_study.loc[:, 'age'] = age_study['age'].apply(lambda x: x / 400)
age_study.loc[:, 'absences'] = age_study['absences'].apply(lambda x: x / 4)
age_study.loc[:, 'G3'] = age_study['G3'].apply(lambda x: x / 5)

num_walc = df2.groupby('Walc').agg({'age': 'sum'})

plt.bar(num_walc.index, num_walc['age'])
plt.title('Num Students per Was Level')
plt.xlabel('Walc')
plt.ylabel('Students')
plt.savefig('outputs/students_per_walc.png', dpi=300)
plt.clf()

plt.bar(df2['Walc'], df2['absences'])
plt.title('Weekend Alcohol Consumption vs. Absence')
plt.xlabel('Weekend Alcohol Consumption')
plt.ylabel('Absences')
plt.savefig('outputs/walc_vs_absence.png', dpi=300)
plt.clf()

# The number of students who are rated at 1 in Wac having such a high absence rate isn't too surprising. There are many 
# more students listed at 1 than the other levels, and most students are probably absent for other reasons than how much 
# they drank of the weekend. 

plt.plot(age_study.index, age_study['G3'], label='Median G3 score', c='green')
plt.plot(age_study.index, age_study['failures'], label='Mean failures', c='red')
plt.plot(age_study.index, age_study['absences'], label='Mean absences', c='yellow')
plt.plot(age_study.index, age_study['studytime'], label='Mean study time', c='blue')
plt.plot(age_study.index, age_study['age'], label='Num students', c='brown')
plt.legend()
plt.title('Scaled Comparison of Student Variables Grouped By Age')
plt.savefig('outputs/scaled_comparison_by_age.png', dpi=300)
plt.clf()

# This graph shows an increasing failure rate, an inconsistent, but generally high, absence rate that spikes, and a G3 
# rate on a general downward slope with two spikes. The first G3 spike happens between 19 to 20 year olds where the 
# absence rate plummets. The failure rate also starts to level off around this area. The student body has decreased 
# dramatically between 20-22, so the absence and failure rates are a bit more dramatic as they affect the average. 

plt.scatter(age_study['goout'], age_study['studytime'])
plt.title('Going Out vs. Study Time')
plt.xlabel('Time Spent Out')
plt.ylabel('Time Spent Studying')
plt.savefig('outputs/goout_vs_studytime.png', dpi=300)
plt.close()

# Save for one dot at the bottom of goout 3, These results align with what I would expect. The students who don't spend as 
# much time out spend more time studing than students who spend more of their time out.

# Task 4

X_failures = df2[['failures']]
X_train, X_test, y_train, y_test = train_test_split(X_failures, df2['G3'], test_size=0.2, random_state=42)

model_failures = LinearRegression()
model_failures.fit(X_train, y_train)
y_failures_predicted = model_failures.predict(X_test)

print("Slope:", model_failures.coef_[0])
print("RMSE: ", np.sqrt(np.mean((y_failures_predicted - y_test) ** 2)))
print("R²: ", model_failures.score(X_test, y_test))

# The Slope shows a 1.43 deficit as failures increase, which is a good chunk to lose. However, the RMSE shows the 
# prediction could be off by nearly 3 points, about twice the Slope value.
# The R² is worse than I had expected. Since failures has the strongest correlation with G3, I thought it would start a 
# little higher than it has.

# Q5

feature_cols = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
                "absences", "freetime", "goout", "Walc", "schoolsup",
                "internet", "higher", "activities", "sex"]
X = df2[feature_cols].values
y = df2["G3"].values
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train_all, y_train_all)
y_predicted = model.predict(X_test_all)

print("R² Train: ", model.score(X_train_all, y_train_all))
print("R²: ", model.score(X_test_all, y_test_all))
print("RMSE: ", np.sqrt(np.mean((y_predicted - y_test_all) ** 2)))

for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# That size of the negative correlation for schoolsup I found the most surprising. I had thought that receiving extra help
# would improve scores. However the students who need the extra help are likely struggling enough to end up with lower 
# scores even with the extra help.

# There is a small gap between the training R² and test R², with test having a slight advantage. This tells me that the 
# model has a decent level of prediction.

# If I was deploying this model, I would drop categories with a coefficient inside -.1 and .1 as they don't seem to have 
# much of an effect on the prediction. These include traveltime, absences, freetime, and activities. I think the important 
# catagories are those that fall outside of -.2 and .2.

# Task 6

plt.scatter(y_predicted, y_test_all, color="blue")
# plt.axline(y_predicted, y_predicted, color="red", label="Linear fit")
plt.plot(y_predicted, y_predicted, color="red")
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted G3")
plt.ylabel("Actual G3")
plt.savefig('outputs/predicted_vs_actual_g3.png', dpi=300)
plt.close()

# It appears as though the model struggles more with the high end as there are more dots on the low end in close proximity 
# with the predicted line than high dots. The dots on the lower side are also spread more evenly above and below the line 
# whereas the dots on the high side are mostly above the line. The middle does show a lot of variety in distance from the 
# predicted line. Values above the diagonal are examples where the model underpredicted the G3 score of the given variables. 
# Dots below the diagonal are when the model overpredicted the G3 scores based on the variables.

# The filtered dataset brought the row count down to 357 while the size of the test set was 20% of that, which is 72. 
# The RMSE and R² of the best model were 2.664 and 0.263 respectively. Adding all the variables from the data did bring 
# down the typical prediction error and improve the predicting power of the model. The typical predition error is the 
# typical range of error in the prediction. In this case, 2.664 is quite a good chunk of the possible 20 points for G3. 
# The largest positive coefficient is internet access at home, while the largest negative coefficient is extra school 
# support. 

# Neglected Feature: The Power of G1

feature_cols = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
                "absences", "freetime", "goout", "Walc", "schoolsup",
                "internet", "higher", "activities", "sex", "G1"]
X = df2[feature_cols].values
y = df2["G3"].values
X_train_G1, X_test_G1, y_train_G1, y_test_G1 = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train_G1, y_train_G1)
y_predicted = model.predict(X_test_G1)

print("R² with G1: ", model.score(X_test_G1, y_test_G1))

for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# No, the higher R² with G1 included does not imply causation. However, there is a greater correlation that vastly 
# improves the predictability of the model. I am not sure that this creates a stronger model for finding struggling 
# students as it places much more strength on test performance than the other factors that affect them. On the other 
# hand, it might provide a better balance to identify those factors. The coefficients are not so widely separated when 
# including G1. Educators may use a constant for G1 to help identify and intervene with struggling students.
