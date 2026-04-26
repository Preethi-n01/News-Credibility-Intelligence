import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

# Ensure NLTK data is ready for TextBlob
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
model_dir = os.path.join(BASE_DIR, 'models')
os.makedirs(model_dir, exist_ok=True)

# ----------------------------------------------
# 1. Fake News models - UPGRADED for Real-World News
# ----------------------------------------------
print("Initializing Real-World Intelligence Training (Veracity Layer)...")
fake = pd.read_csv(os.path.join(BASE_DIR, 'data', 'Fake.csv'))
real = pd.read_csv(os.path.join(BASE_DIR, 'data', 'True.csv'))

fake['label'] = 1
real['label'] = 0

data = pd.concat([fake, real]).reset_index(drop=True)
data = data[['text', 'label']].dropna()
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Intelligence Upgrade: Bigrams (1,2) and 10k features to capture "misinformation phrases"
fake_vectorizer = TfidfVectorizer(
    stop_words='english', 
    max_features=10000, 
    ngram_range=(1, 2), # Capture context like "secret exposed"
    min_df=2
)
X = fake_vectorizer.fit_transform(data['text'])
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression (Optimized C for better generalization)
print("  Training Logistic Regression Engine...")
lr = LogisticRegression(max_iter=500, C=1.5, solver='lbfgs')
lr.fit(X_train, y_train)

# Random Forest (Deeper trees for complexity)
print("  Training Random Forest Engine...")
rf = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# SVM (Stronger regularizer)
print("  Training SVM Engine...")
svm_base = LinearSVC(max_iter=2000, C=0.8, dual=False)
svm = CalibratedClassifierCV(svm_base, cv=5)
svm.fit(X_train, y_train)

# Save fake-news models
pickle.dump(lr,             open(os.path.join(model_dir, 'fake_logistic_regression.pkl'), 'wb'))
pickle.dump(rf,             open(os.path.join(model_dir, 'fake_random_forest.pkl'),       'wb'))
pickle.dump(svm,            open(os.path.join(model_dir, 'fake_svm.pkl'),                 'wb'))
pickle.dump(fake_vectorizer,open(os.path.join(model_dir, 'fake_vectorizer.pkl'),          'wb'))
pickle.dump(lr,             open(os.path.join(model_dir, 'fake_model.pkl'),               'wb'))
print("Veracity models deployed.")

# ----------------------------------------------
# 2. Clickbait model - UPGRADED for Headlines
# ----------------------------------------------
print("\nInitializing Engagement Intelligence Training (Clickbait Layer)...")
clickbait = pd.read_csv(os.path.join(BASE_DIR, 'data', 'clickbait.csv'))

# Bigrams are CRITICAL for clickbait (e.g., "You won't believe")
click_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
X_cb = click_vectorizer.fit_transform(clickbait['headline'])
y_cb = clickbait['label']

click_model = LogisticRegression(max_iter=300, C=1.2)
click_model.fit(X_cb, y_cb)

pickle.dump(click_model,    open(os.path.join(model_dir, 'click_model.pkl'),      'wb'))
pickle.dump(click_vectorizer,open(os.path.join(model_dir, 'click_vectorizer.pkl'),'wb'))
print("Engagement models deployed.")

# ----------------------------------------------
# 3. Sentiment model (Emotional Polarity)
# ----------------------------------------------
print("\nInitializing Tone Intelligence Training (Sentiment Layer)...")
from textblob import TextBlob
def polarity_label(text):
    p = TextBlob(str(text)).sentiment.polarity
    return 'positive' if p > 0.05 else 'negative' if p < -0.05 else 'neutral'

sent_data = data.sample(min(15000, len(data)), random_state=42).copy()
sent_data['sent_label'] = sent_data['text'].apply(polarity_label)

sent_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_sent = sent_vectorizer.fit_transform(sent_data['text'])
y_sent = sent_data['sent_label']

sentiment_model = LogisticRegression(max_iter=300)
sentiment_model.fit(X_sent, y_sent)

pickle.dump(sentiment_model,   open(os.path.join(model_dir, 'sentiment_model.pkl'),   'wb'))
pickle.dump(sent_vectorizer,  open(os.path.join(model_dir, 'sentiment_vectorizer.pkl'),'wb'))
print("Tone models deployed.")

print("\nALL INTELLIGENCE ENGINES TRAINED AND DEPLOYED FOR REAL-WORLD ANALYSIS.")
