import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('IITB_UG_Student_Dataset.xlsx - Sheet1.csv')

numeric_cols = ['Age', 'StudyHoursPerDay', 'SocialMediaHours', 'MovieTvShowHours', 
                'AttendancePercentage', 'SleepHoursPerNight', 'Exercise_frequency', 
                'mental_health_rating']

categorical_cols = ['Gender' , 'PoR' , 'Diet_Quality', 'parental_education_level' , 'internet_quality'
                    , 'extracurricular_participation']


df[numeric_cols] = df[numeric_cols].interpolate(method = 'linear')
df[numeric_cols] = df[numeric_cols].fillna(method='bfill')

for col in categorical_cols:

    mode_value = df[col].mode()[0]  # [0] because mode() returns a Series
    df[col] = df[col].fillna(mode_value)


df.to_csv('processed_Q1&2.csv', index=False)

target = 'Cumulative_Grade'
features = ['Age', 'StudyHoursPerDay', 'SocialMediaHours', 'MovieTvShowHours', 
                'AttendancePercentage', 'SleepHoursPerNight', 'Exercise_frequency', 
                'mental_health_rating','Gender' , 'PoR' , 'Diet_Quality', 'parental_education_level' , 'internet_quality'
                    , 'extracurricular_participation']

categorical_cols = ['Gender' , 'PoR' , 'Diet_Quality', 'parental_education_level' , 'internet_quality'
                    , 'extracurricular_participation']


means = df[numeric_cols].mean()
medians = df[numeric_cols].median()
stds = df[numeric_cols].std()


#cummulative greade vs study hours
plt.figure(figsize=(8,5))
sns.scatterplot(x='StudyHoursPerDay', y='Cumulative_Grade', data=df, alpha=0.6, color="green")
sns.regplot(x='StudyHoursPerDay', y='Cumulative_Grade', data=df, scatter=False, color='red')
plt.title('Study Hours vs. Cumulative Grade')
plt.xlabel('Study Hours Per Day')
plt.ylabel('Cumulative Grade')
plt.tight_layout()


# Study Hours Per Day
plt.figure(figsize=(7,4))
sns.histplot(df['StudyHoursPerDay'], bins=30, kde=True, color='salmon')
plt.title('Distribution of Study Hours Per Day')
plt.xlabel('Study Hours Per Day')
plt.ylabel('Count')
plt.tight_layout()

# Cumulative Grades
plt.figure(figsize=(7,4))
sns.histplot(df['Cumulative_Grade'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Cumulative Grades')
plt.xlabel('Cumulative Grade')
plt.ylabel('Count')
plt.tight_layout()




#cummulative grade vs gender
sns.stripplot(x='Gender', y='Cumulative_Grade', data=df, jitter=0.25, alpha=0.7, palette='Set2')
plt.title('Cumulative Grade by Gender')
plt.xlabel('Gender')
plt.ylabel('Cumulative Grade')
plt.tight_layout()




#histrogram of sleep hours dist
sns.histplot(df['SleepHoursPerNight'], bins=20, kde=True, color='purple')
plt.title('Distribution of Sleep Hours Per Night')
plt.xlabel('Sleep Hours Per Night')
plt.ylabel('Number of Students')
plt.tight_layout()



#heatmap avg socual hours vs dieat quality
grade_bins = [0, 60, 70, 80, 90, 100]
grade_labels = ['<60', '60-70', '70-80', '80-90', '90-100']
df['Grade_Bin'] = pd.cut(df['Cumulative_Grade'], bins=grade_bins, labels=grade_labels, include_lowest=True)

# Pivot table: rows = Grade Bin, columns = Diet Quality, values = mean Social Media Hours
data = df.pivot_table(
    index='Grade_Bin',
    columns='Diet_Quality',
    values='SocialMediaHours',
    aggfunc='mean'
)

plt.figure(figsize=(8, 5))
sns.heatmap(data, annot=True, cmap='YlOrBr', fmt='.2f')
plt.title('Average Social Media Hours by Grade Bin and Diet Quality')
plt.xlabel('Diet Quality')
plt.ylabel('Cumulative Grade Bin')
plt.tight_layout()



#exercise frequency vs cummulative grade
sns.stripplot(x='Exercise_frequency', y='Cumulative_Grade', data=df, jitter=0.25, alpha=0.7, palette='Set1')
plt.title('Cumulative Grade by Exercise Frequency')
plt.xlabel('Exercise Frequency')
plt.ylabel('Cumulative Grade')
plt.tight_layout()


plt.show()






for col in categorical_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

def print_metrics(y_true, y_pred, name):
    print(f"{name} R²: {r2_score(y_true, y_pred):.2f}")
    print(f"{name} RMSE: {mean_squared_error(y_true, y_pred):.2f}")
    print(f"{name} MAE: {mean_absolute_error(y_true, y_pred):.2f}")

print_metrics(y_test, y_pred_lr, "Linear Regression")
print_metrics(y_test, y_pred_rf, "Random Forest")


importances = rf.feature_importances_
for i in range(14):
    print(f'{features[i]} === {importances[i]}')



plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(y_test, y_pred_lr, alpha=0.5, label='Linear regression')
plt.scatter(y_test, y_pred_rf, alpha=0.5, label='Random Forest', color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('Actual Grade')
plt.ylabel('Predicted Grade')
plt.title('Predicted vs Actual Grades')
plt.legend()

plt.subplot(1,2,2)
sns.histplot(y_test - y_pred_lr, color='blue', label='Linear Regression', kde=True)
sns.histplot(y_test - y_pred_rf, color='orange', label='Random Forest', kde=True)
plt.title('Error Plot')
plt.xlabel('Prediction Error')
plt.legend()
plt.tight_layout()
plt.show()


