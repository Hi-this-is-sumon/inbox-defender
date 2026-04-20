import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))

CLEAN_REGEX = re.compile(r'[^a-z0-9\s$]')

def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = text[:3000].lower()
    text = CLEAN_REGEX.sub(' ', text)

    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in STOPWORDS]

    return ' '.join(words)

def verify():
    print("Loading model...")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'model', 'spam_model.pkl')
    vectorizer_path = os.path.join(base_dir, 'model', 'vectorizer.pkl')

    if not os.path.exists(model_path):
        print("Model not found!")
        return

    model = pickle.load(open(model_path, 'rb'))
    vectorizer = pickle.load(open(vectorizer_path, 'rb'))

    test_cases = [
        ("WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!", "SPAM"),
        ("Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005", "SPAM"),
        ("Hey, are we still meeting for lunch today?", "HAM"),
        ("Your Amazon order #123-456 has been shipped.", "HAM"),
        ("URGENT: Your account has been suspended. Click here to verify.", "SPAM"),
        ("Dear customer, please review the attached invoice #998877 for payment immediately.", "SPAM"),
        ("Security Alert: We noticed a new login from Russia. Reset your password now.", "SPAM"),
        ("Congratulations! You have been promoted to Senior Developer.", "HAM"),
        ("Here is the free report you requested on Q3 earnings.", "HAM"),
        ("Limited time offer: 50% off on all shoes at our store this weekend only.", "HAM")
    ]

    print(f"\nRunning {len(test_cases)} test cases...")
    passed = 0

    for text, expected in test_cases:
        processed = preprocess_text(text)
        vectorized = vectorizer.transform([processed])

        pred = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]

        
        threshold = 0.5

        if proba[1] >= threshold:
            label = "SPAM"
            confidence = proba[1]
        else:
            label = "HAM"
            confidence = proba[0]

        ui_label = "Spam" if label == "SPAM" else "Not Spam"
        expected_ui = "Spam" if expected == "SPAM" else "Not Spam"

        result = "PASS" if ui_label == expected_ui else "FAIL"

        print(f"Text: {text[:50]}...")
        print(f"Expected: {expected_ui}, Got: {ui_label} (Conf: {confidence:.2f}) - {result}")
        print("-" * 30)

        if result == "PASS":
            passed += 1

    print(f"\nAccuracy: {passed}/{len(test_cases)} ({passed/len(test_cases)*100:.1f}%)")

if __name__ == "__main__":
    verify()