import pandas as pd
import re #regex lib
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences as pad #for uniformity
import tensorflow as tf

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
categories = ['Negetive', 'Positive']


def preprocessing_text(text):
    text = text.lower()       #lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  #remove punctuation
    words = text.split()
    words = [w for w in words if w not in stop_words]   #remove stopword
    return ' '.join(words)

df = pd.read_csv("Q5_Dataset.csv")
df['clean_review'] = df['review'].apply(preprocessing_text) #apply method for a function in pandas!!

df = df[df['sentiment'].isin(['positive', 'negative'])]


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['review'])
y = df['sentiment'].map({'positive': 1, 'negative': 0})  # Adjust if your labels differ


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)


# y_pred = model.predict(X_test)
# print(classification_report(y_test, y_pred))

user_input = input("input the movie review::  ")
cleaned_text = preprocessing_text(user_input)
    
    # Transform using the same vectorizer
text_vectorized = vectorizer.transform([cleaned_text])
    
    # Make prediction
prediction = model.predict(text_vectorized)[0]
    
    # Return result
if prediction == 1:
    print("Positive")
else:
    print("Negetive")
