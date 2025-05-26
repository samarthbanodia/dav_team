import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
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
# Encode categoricals
# for col in df.select_dtypes(include='object').columns:
#     if col in features:
#         df[col] = df[col].astype(str)
#         le = LabelEncoder()
#         df[col] = le.fit_transform(df[col])

for col in categorical_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Metrics
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


