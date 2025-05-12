import pandas as pd
import numpy as np
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




missing_before = df[categorical_cols].isnull().sum()
print("Missing values before interpolation:")
print(missing_before)



# df['Cumulative_Grade'] = pd.to_numeric(df['Cumulative_Grade'], errors='coerce')

# plt.figure(figsize=(8,5))
# sns.histplot(df['Cumulative_Grade'], bins=30, kde=True, color='skyblue')
# plt.title('Distribution of Cumulative Grade')
# plt.xlabel('Cumulative Grade')
# plt.ylabel('Count')
# plt.tight_layout()
# plt.show()

# df['StudyHoursPerDay'] = pd.to_numeric(df['StudyHoursPerDay'], errors='coerce').interpolate()
# plt.figure(figsize=(8,5))
# sns.histplot(df['StudyHoursPerDay'], bins=30, kde=True, color='salmon')
# plt.title('Distribution of Study Hours Per Day')
# plt.xlabel('Study Hours Per Day')
# plt.ylabel('Count')
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8,5))
# sns.boxplot(x='Gender', y='Cumulative_Grade', data=df)
# plt.title('Cumulative Grade by Gender')
# plt.xlabel('Gender')
# plt.ylabel('Cumulative Grade')
# plt.tight_layout()
# plt.show()

# df['AttendancePercentage'] = pd.to_numeric(df['AttendancePercentage'], errors='coerce').interpolate()
# df['AttendanceBin'] = pd.cut(df['AttendancePercentage'], bins=[0,75,85,95,100], labels=['<75','75-85','85-95','95-100'])

# plt.figure(figsize=(8,5))
# sns.boxplot(x='AttendanceBin', y='Cumulative_Grade', data=df)
# plt.title('Cumulative Grade by Attendance Percentage')
# plt.xlabel('Attendance Percentage Bin')
# plt.ylabel('Cumulative Grade')
# plt.tight_layout()
# plt.show()

# pivot = df.pivot_table(index='Gender', columns='Diet_Quality', values='Cumulative_Grade', aggfunc='mean')
# plt.figure(figsize=(8, 5))
# sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.1f', linewidths=.5, cbar_kws={'label': 'Avg Grade'})
# plt.title('Average Cumulative Grade by Gender and Diet Quality')
# plt.ylabel('Gender')
# plt.xlabel('Diet Quality')
# plt.tight_layout()
# plt.show()
#heat mapper

# plt.figure(figsize=(10,8))
# sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Heatmap')
# plt.tight_layout()
# plt.show()


# social_bins = pd.cut(df['SocialMediaHours'], bins=5, labels=False)
# movie_bins = pd.cut(df['MovieTvShowHours'], bins=5, labels=False)
# sleep_bins = pd.cut(df['SleepHoursPerNight'], bins=5, labels=False)

# # 1. Social Media Hours vs Movie/TV Show Hours
# pivot_social_movie = df.pivot_table(
#     index=social_bins, columns=movie_bins, values='Cumulative_Grade', aggfunc='mean'
# )

# plt.figure(figsize=(8, 6))
# sns.heatmap(pivot_social_movie, annot=True, fmt='.1f', cmap='YlGnBu')
# plt.title('Average Grade: Social Media Hours vs Movie/TV Show Hours')
# plt.xlabel('Movie/TV Show Hours (binned)')
# plt.ylabel('Social Media Hours (binned)')
# plt.tight_layout()
# plt.show()
# Define "high grade" threshold (e.g., 85 and above)


# order = ['High School', 'Bachelor', 'Master']
# palette = {'High School': '#E57373', 'Bachelor': '#64B5F6', 'Master': '#81C784'}

# plt.figure(figsize=(10,6))
# sns.stripplot(
#     x='parental_education_level',
#     y='Cumulative_Grade',
#     data=df,
#     order=order,
#     palette=palette,
#     jitter=0.25,
#     alpha=0.7,
#     size=4
# )
# sns.pointplot(
#     x='parental_education_level',
#     y='Cumulative_Grade',
#     data=df,
#     order=order,
#     palette=palette,
#     join=False,
#     markers='D',
#     scale=1.2,
#     ci='sd',
#     errwidth=1.5,
#     capsize=0.18
# )
# plt.title('Distribution of Grades by Parental Education Level', fontsize=16, weight='bold')
# plt.xlabel('Parental Education Level', fontsize=14)
# plt.ylabel('Cumulative Grade', fontsize=14)
# plt.ylim(0, 110)
# plt.grid(axis='y', linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()


# means = df[numeric_cols].mean()
# medians = df[numeric_cols].median()
# stds = df[numeric_cols].std()

# # Prepare DataFrame for plotting
# stats_df = pd.DataFrame({
#     'Mean': means,
#     'Median': medians,
#     'Std Dev': stds
# }).round(2)


# print(means , ' ')
# print(medians)
# print(stds)

# # Plot
# fig, ax = plt.subplots(figsize=(10, 6))
# y = np.arange(len(numeric_cols))

# # Plot mean with error bars for std deviation
# ax.barh(y, stats_df['Mean'], xerr=stats_df['Std Dev'], color='#64b5f6', alpha=0.8, label='Mean ± Std Dev')
# # Overlay median as a vertical marker
# ax.scatter(stats_df['Median'], y, color='#d84315', label='Median', zorder=5, marker='D', s=60)

# ax.set_yticks(y)
# ax.set_yticklabels(numeric_cols, fontsize=12)
# ax.set_xlabel('Value', fontsize=13)
# ax.set_title('Mean, Median, and Std Dev of Numerical Features', fontsize=15, weight='bold')
# ax.legend(loc='lower right')
# plt.grid(axis='x', linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(8,5))
# sns.scatterplot(x='StudyHoursPerDay', y='Cumulative_Grade', data=df, alpha=0.6, color="green")
# sns.regplot(x='StudyHoursPerDay', y='Cumulative_Grade', data=df, scatter=False, color='red')
# plt.title('Study Hours vs. Cumulative Grade')
# plt.xlabel('Study Hours Per Day')
# plt.ylabel('Cumulative Grade')
# plt.tight_layout()

# # 2. Histogram: Study Hours Per Day
# plt.figure(figsize=(7,4))
# sns.histplot(df['StudyHoursPerDay'], bins=30, kde=True, color='salmon')
# plt.title('Distribution of Study Hours Per Day')
# plt.xlabel('Study Hours Per Day')
# plt.ylabel('Count')
# plt.tight_layout()

# # 1. Histogram: Cumulative Grades
# plt.figure(figsize=(7,4))
# sns.histplot(df['Cumulative_Grade'], bins=30, kde=True, color='skyblue')
# plt.title('Distribution of Cumulative Grades')
# plt.xlabel('Cumulative Grade')
# plt.ylabel('Count')
# plt.tight_layout()


# #grade v gender
# sns.stripplot(x='Gender', y='Cumulative_Grade', data=df, jitter=0.25, alpha=0.7, palette='Set2')
# plt.title('Cumulative Grade by Gender')
# plt.xlabel('Gender')
# plt.ylabel('Cumulative Grade')
# plt.tight_layout()



#sleep hours
# sns.histplot(df['SleepHoursPerNight'], bins=20, kde=True, color='purple')
# plt.title('Distribution of Sleep Hours Per Night')
# plt.xlabel('Sleep Hours Per Night')
# plt.ylabel('Number of Students')
# plt.tight_layout()

# plt.show()


# grade_bins = [0, 60, 70, 80, 90, 100]
# grade_labels = ['<60', '60-70', '70-80', '80-90', '90-100']
# df['Grade_Bin'] = pd.cut(df['Cumulative_Grade'], bins=grade_bins, labels=grade_labels, include_lowest=True)

# # Pivot table: rows = Grade Bin, columns = Diet Quality, values = mean Social Media Hours
# pivot = df.pivot_table(
#     index='Grade_Bin',
#     columns='Diet_Quality',
#     values='SocialMediaHours',
#     aggfunc='mean'
# )

# # Plot heatmap
# plt.figure(figsize=(8, 5))
# sns.heatmap(pivot, annot=True, cmap='YlOrBr', fmt='.2f')
# plt.title('Average Social Media Hours by Grade Bin and Diet Quality')
# plt.xlabel('Diet Quality')
# plt.ylabel('Cumulative Grade Bin')
# plt.tight_layout()
# plt.show()


sns.stripplot(x='Exercise_frequency', y='Cumulative_Grade', data=df, jitter=0.25, alpha=0.7, palette='Set1')
plt.title('Cumulative Grade by Exercise Frequency')
plt.xlabel('Exercise Frequency')
plt.ylabel('Cumulative Grade')
plt.tight_layout()
plt.show()

