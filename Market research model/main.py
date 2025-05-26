import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler




training_df = pd.read_csv('Q3, Q4_Consumer_Dataset.csv')
testing_df = pd.read_csv('Q3, Q4_Consumer_Test_Dataset.csv')

numerical_cols = ['Age', 'Family_Size', 'Work_Experience']
categorical_cols = ['Gender', 'Ever_Married', 'Profession', 'Graduated', 'Energy_Consumption', 'Preferred_Renewable']

all_cols = ['Age', 'Family_Size', 'Work_Experience','Gender', 'Ever_Married', 'Profession', 'Graduated', 'Energy_Consumption', 'Preferred_Renewable']

for col in numerical_cols:
    training_df[col].fillna(training_df[col].mean(), inplace=True)
    testing_df[col].fillna(testing_df[col].mean(), inplace=True)

for col in categorical_cols:
    training_df[col].fillna(training_df[col].mode()[0], inplace=True)
    testing_df[col].fillna(testing_df[col].mode()[0], inplace=True)



missing_before = testing_df[categorical_cols].isnull().sum()
print("Missing values before interpolation:")
print(missing_before)


from sklearn.preprocessing import LabelEncoder

le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    training_df[col] = le.fit_transform(training_df[col])
    testing_df[col] = le.transform(testing_df[col])
    le_dict[col] = le

# Encode the target
group_le = LabelEncoder()
training_df['Group'] = group_le.fit_transform(training_df['Group'])

testing_df_copy  = testing_df.copy()



from sklearn.ensemble import RandomForestClassifier

X_train = training_df.drop('Group', axis=1)
y_train = training_df['Group']
X_pred = testing_df

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

testing_df['Predicted_Group'] = group_le.inverse_transform(rf.predict(X_pred))

print(testing_df)

testing_df.to_csv('processed_Q2&3.csv', index=False)


from sklearn.cluster import KMeans

scaler = StandardScaler()
scaled_features = scaler.fit_transform(testing_df_copy[all_cols])

kmeans = KMeans(n_clusters=4, random_state=42)
testing_df_copy['Groups'] = kmeans.fit_predict(scaled_features)

testing_df_copy['Groups'] = testing_df_copy['Groups'].map({0: 'A', 1: 'B', 2: 'C', 3: 'D'})
print(testing_df_copy)

testing_df_copy.to_csv('24b0302_bonus_Q3&4.csv', index=False)

