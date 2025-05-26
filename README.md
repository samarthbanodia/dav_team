

## 🎓 Project 1: Academic Performance Prediction Using Study and Lifestyle Patterns

**Overview**  
Analyzed how undergraduate students’ behaviors and habits affect academic performance, and built predictive models for Cumulative Grade.

**Dataset**
- 👨‍🎓 1,000 students: Demographics, study habits, engagement, academic outcomes

**Objectives**
- 🔍 EDA to identify key patterns
- 🤖 Train regression models (Linear Regression, Random Forest) for grade prediction
- 📊 Analyze feature importance
- 💻 **Bonus:** Excel-based regression modeling

**Preprocessing**
- 🔢 Numerical: Linear interpolation, forward fill
- 🔤 Categorical: Mode imputation
- ⚠️ Outliers: Retained genuine variations
- 🔄 Data types: Explicitly cast

**Key EDA Insights**
- 📈 More study hours → better grades
- 🚻 No significant gender disparity
- 😴 Healthy sleep (6–7 hrs) is common and beneficial
- 🏃‍♂️ Frequent exercise and 🥗 good diet correlate with higher grades
- 📱 Social media use negatively correlates with grades

**Regression Modeling**

| Metric    | 📈 Linear Regression | 🌳 Random Forest |
|-----------|---------------------|-----------------|
| R² Score  | 0.77                | 0.68            |
| RMSE      | 58.46               | 82.42           |
| MAE       | 5.79                | 7.13            |

- **Linear Regression:** Slightly better fit  
- **Random Forest:** Better with non-linearities and outliers

**Top Predictive Features**
- 📚 StudyHoursPerDay
- 😊 MentalHealthRating
- 🏫 AttendanceRate
- 😴 SleepHoursPerNight
- 🌐 InternetUsageHours

**Bonus: Excel Regression**
- 🟩 Used Data Analysis Toolpak
- 📊 Explained R², Adjusted R², Standard Error, p-values, Coefficients



---

## 🎬 Project 2: Sentiment Analysis of Movie Reviews

**Overview**  
Automated sentiment detection for user-written movie reviews, classifying them as positive or negative.

**Problem Statement**
- 🔍 **Binary classification:** Predict sentiment (0 = Negative, 1 = Positive) for new reviews.

**Dataset**
- 📝 **Text:** Informal, user-written reviews
- 🏷️ **Labels:** Binary sentiment

**Solution Pipeline**
1. 🧹 **Text Preprocessing:**  
   - Lowercasing  
   - Remove punctuation & non-alphanumeric characters  
   - Remove stopwords (NLTK)  
   - Tokenization
2. 🏷️ **Feature Extraction:**  
   - TF-IDF vectorization
3. 🤖 **Model Training:**  
   - **Logistic Regression** (scikit-learn)  
   - 80/20 train/test split  
   - Evaluation: Precision, Recall, F1-score, Accuracy (90%)

**Results**

| 📊 Metric   | 😠 Negative | 😃 Positive |
|-------------|------------|------------|
| Precision   | 0.91       | 0.89       |
| Recall      | 0.89       | 0.91       |
| F1-Score    | 0.90       | 0.90       |
| Support     | 4961       | 5039       |





---

## 🌱 Project 3: Customer Segmentation for Targeted Outreach in Renewable Energy Expansion

**Overview**  
Segmented new customers for The Renewables, a company entering the Indian market, to enable targeted marketing of products P, Q, R, S, and T.

**Problem Statement**  
- 🏷️ **Labeled training data:** 8000 customers with known group segments (A, B, C, D)  
- 🆕 **Unlabeled test data:** 2500 potential customers  
- 🎯 **Objective:** Predict the segment for each test customer using supervised ML.  
- 💡 **Bonus:** Unsupervised clustering if labels are unavailable.

**Dataset Features**
- 🔢 **Numerical:** Age, Work_Experience, Family_Size
- 🔤 **Categorical:** Gender, Ever_Married, Profession, Graduated, Energy_Consumption, Preferred_Renewable

**Methodology**
- 🧹 **Preprocessing:**  
  - Fill missing numerical values with mean  
  - Fill missing categorical values with mode  
  - Label encode categorical columns  
  - Validate data cleanliness with `.isnull().sum()`
- 🤖 **Supervised Model:**  
  - **Random Forest Classifier** (scikit-learn)  
  - Trained on labeled data, predicted on test set  
  - Output saved as CSV  
- 🌳 **Why Random Forest?**  
  - Handles mixed data types  
  - Robust to noise and overfitting  
  - Captures non-linear interactions
- 🌀 **Bonus: Unsupervised Clustering**  
  - **K-Means Clustering** (n_clusters=4)  
  - Data scaled with StandardScaler  
  - Clusters mapped to segments A–D  
  - Output saved as CSV

**Key Learnings**
- ✅ Supervised ML accurately predicts customer segments when labels exist.
- 🔄 K-Means is a viable fallback for unlabeled data.
- 📏 Feature scaling is crucial for clustering.

---

## 🛠️ Tech Stack

- 🐍 **Python**
- 🐼 pandas, numpy (data handling)
- 🤖 scikit-learn (ML models, evaluation)
- 📚 nltk (text preprocessing)
- 📊 matplotlib, seaborn (visualization)
- 🟩 Microsoft Excel (bonus regression)

---

## 📞 Contact

For questions or collaborations, reach out at **24b0392@iitb.ac.in** or visit [samarthbanodia.github.io](https://samarthbanodia.github.io).

---

✨ _Thank you for exploring my projects!_
