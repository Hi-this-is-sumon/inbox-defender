import os
import re
import pickle
import random
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score

# SETUP
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
CLEAN_REGEX = re.compile(r'[^a-z0-9\s$]')

# PREPROCESSING
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = CLEAN_REGEX.sub(' ', text)

    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in STOPWORDS]

    return ' '.join(words)

def build_pipeline():
    print("=== STARTING ML TRAINING PIPELINE ===\n")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'data', 'spam.csv')
    model_path = os.path.join(base_dir, 'spam_model.pkl')
    vectorizer_path = os.path.join(base_dir, 'vectorizer.pkl')
    label_map_path = os.path.join(base_dir, 'label_map.pkl')

    print("1. Loading Dataset...")
    df = pd.read_csv(data_path, encoding='latin-1')

    # FORMAT FIX
    label_map = {'ham': 0, 'spam': 1}
    if 'msg' in df.columns and 'label' in df.columns:
        df = df[['msg', 'label']].rename(columns={'msg': 'message'})
    elif 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v2', 'v1']].rename(columns={'v2': 'message', 'v1': 'label'})

    df['label'] = df['label'].map(label_map)

    print("2. Cleaning Data...")
    df.dropna(inplace=True)
    df['message'] = df['message'].apply(preprocess_text)

    # 🔥 KEY STEP 1: Reduce dataset size
    df = df.sample(n=8000, random_state=42)

    # 🔥 KEY STEP 2: Add noise to labels (real-world effect)
    noise_idx = np.random.choice(df.index, size=int(0.07 * len(df)), replace=False)
    df.loc[noise_idx, 'label'] = 1 - df.loc[noise_idx, 'label']

    # SPLIT
    print("3. Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    # 🔥 NO UPSAMPLING (important)
    X_train_bal = X_train
    y_train_bal = y_train

    # 🔥 TF-IDF (restricted)
    print("4. Vectorizing...")
    cv = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 1),
        min_df=5,
        max_df=0.75
    )

    X_train_vec = cv.fit_transform(X_train_bal)
    X_test_vec = cv.transform(X_test)

    # MODELS
    print("\n5. Training Models...")

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=0.3),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "Calibrated SVC": CalibratedClassifierCV(LinearSVC())
    }

    best_model = None
    best_f1 = 0
    best_name = ""

    for name, model in models.items():
        print(f"\n---> {name}")

        model.fit(X_train_vec, y_train_bal)
        y_pred = model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Accuracy: {acc*100:.2f}% | F1: {f1*100:.2f}%")

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_vec)[:,1]
            roc = roc_auc_score(y_test, y_prob)
            print(f"ROC-AUC: {roc*100:.2f}%")

        print(classification_report(y_test, y_pred))

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    print(f"\nBest Model: {best_name} ({best_f1*100:.2f}%)")

    # SAVE
    with open(model_path, 'wb') as m, \
         open(vectorizer_path, 'wb') as v, \
         open(label_map_path, 'wb') as l:

        pickle.dump(best_model, m)
        pickle.dump(cv, v)
        pickle.dump(label_map, l)

    print("Saved successfully!")

if __name__ == "__main__":
    build_pipeline()