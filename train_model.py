import os
import pickle
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-blocking backend so plt.show() never freezes the script
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "fake_or_real_news.csv")
MODEL_PATH = os.path.join(BASE_DIR, "final_model.sav")

print("Loading dataset from:", CSV_PATH)
df = pd.read_csv(CSV_PATH)
print("Shape:", df.shape)

# 1. Label distribution chart (saved as PNG instead of blocking popup)
sb.countplot(x="label", data=df, hue="label", palette="hls", legend=False)
plt.title("Fake vs Real — label distribution")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "label_distribution.png"))
plt.close()

# 2. Split into training and test sets
y = df["label"]
X_train, X_test, Y_train, Y_test = train_test_split(
    df["text"], y, test_size=0.33, random_state=53
)
print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")

# 3. Build vectorizers
count_vectorizer = CountVectorizer(stop_words="english")
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

# 4. Model A — Naive Bayes (TF-IDF)
nb_tfidf = Pipeline([("tfidf", tfidf_vectorizer), ("clf", MultinomialNB())])
nb_tfidf.fit(X_train, Y_train)
acc_a = metrics.accuracy_score(Y_test, nb_tfidf.predict(X_test))
print(f"Naive Bayes (TF-IDF) accuracy: {round(acc_a * 100, 2)}%")

# 5. Model B — Naive Bayes (Count)
nb_count = Pipeline([("count", count_vectorizer), ("clf", MultinomialNB())])
nb_count.fit(X_train, Y_train)
acc_b = metrics.accuracy_score(Y_test, nb_count.predict(X_test))
print(f"Naive Bayes (Count) accuracy: {round(acc_b * 100, 2)}%")

# 6. Model C — Passive Aggressive Classifier
pa_model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7)),
    ("clf", PassiveAggressiveClassifier(max_iter=50))
])
pa_model.fit(X_train, Y_train)
acc_c = metrics.accuracy_score(Y_test, pa_model.predict(X_test))
print(f"Passive Aggressive accuracy: {round(acc_c * 100, 2)}%")

# 7. Save final_model.sav directly into D:\fakenews
with open(MODEL_PATH, "wb") as f:
    pickle.dump(pa_model, f)

print("\nSUCCESS! Saved:", MODEL_PATH)