# model/train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from utils.preprocess import clean_text, lemmatize

df = pd.read_csv("data/resume_dataset.csv")
df['processed'] = df['resume_text'].apply(lambda x: lemmatize(clean_text(x)))

X_train, X_test, y_train, y_test = train_test_split(df['processed'], df['career_path'], test_size=0.2)

model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

model.fit(X_train, y_train)
print("Test Accuracy:", model.score(X_test, y_test))
joblib.dump(model, "model/career_model.pkl")
