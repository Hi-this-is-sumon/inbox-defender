from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pickle
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import os

# Initialize App & NLTK
app = FastAPI()
nltk.download('wordnet', quiet=True)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model/spam_model.pkl")
vectorizer_path = os.path.join(base_dir, "model/vectorizer.pkl")
label_map_path = os.path.join(base_dir, "model/label_map.pkl") 
whitelist_path = os.path.join(base_dir, "data/whitelist.csv")             
trusted_domains_path = os.path.join(base_dir, "data/trusted_domains.csv") 
landing_page_path = os.path.join(base_dir, "templates", "index.html")
static_dir = os.path.join(base_dir, "static")

if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Globals
model = None
vectorizer = None
label_map = None
personal_whitelist = set()
global_trusted_domains = set()

# Lemmatizer replaces the old Stemmer to match the new ML model exactly
lemmatizer = WordNetLemmatizer()
STOPWORDS = set(ENGLISH_STOP_WORDS)
CLEAN_REGEX = re.compile(r'[^a-z0-9\s$]')

def load_resources():
    global model, vectorizer, label_map, personal_whitelist, global_trusted_domains

    try:
        if os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(label_map_path):
            with open(model_path, "rb") as m_file, open(vectorizer_path, "rb") as v_file, open(label_map_path, "rb") as l_file:
                model = pickle.load(m_file)
                vectorizer = pickle.load(v_file)
                label_map = pickle.load(l_file)
        else:
            print("WARNING: Model files not found. Please train the model first.")
    except Exception as exc:
        print(f"Error loading model resources: {exc}")

    try:
        if os.path.exists(whitelist_path):
            df_white = pd.read_csv(whitelist_path)
            if 'email' in df_white.columns:
                personal_whitelist.update(df_white['email'].dropna().astype(str).str.lower().values)
            if 'domain' in df_white.columns:
                global_trusted_domains.update(df_white['domain'].dropna().astype(str).str.lower().values)
    except Exception as exc:
        pass # Silently pass if no personal whitelist exists

    try:
        if os.path.exists(trusted_domains_path):
            df_trust = pd.read_csv(trusted_domains_path, header=None)
            global_trusted_domains.update(df_trust[0].dropna().astype(str).str.lower().values)
            print(f"Loaded {len(global_trusted_domains)} trusted domains.")
    except Exception as exc:
        print(f"Error loading trusted_domains.csv: {exc}")

load_resources()

# EXACT SAME INPUT CONTRACT FOR YOUR EXTENSION
class EmailRequest(BaseModel):
    sender: str
    subject: str
    body: str

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text[:3000].lower()
    text = CLEAN_REGEX.sub(' ', text)
    text = [lemmatizer.lemmatize(word) for word in text.split() if word not in STOPWORDS]
    return ' '.join(text)

def check_whitelist(sender):
    if not sender: return False
    return sender.lower() in personal_whitelist

def is_trusted_domain(sender):
    if not sender or '@' not in sender: return False
    domain = sender.split('@')[-1].lower()
    
    for trusted in global_trusted_domains:
        if domain == trusted or domain.endswith('.' + trusted):
            return True
    return False

def has_spam_indicators(subject, body):
    text = f"{subject} {body}".lower()
    
    spam_keywords = [
        'bitcoin', 'cryptocurrency', 'crypto wallet', 'guaranteed returns', 'investment scheme',
        'urgent act', 'limited time', 'act now', 'expire soon', 'verify your account immediately',
        'suspended account', 'confirm identity', 'unauthorized login', 'account restricted',
        'update kyc', 'kyc pending', 'bank account blocked',
        'winner', 'lottery', 'prize', 'claim now', 'congratulations!!!', 'free money', 'wire transfer'
    ]
    
    spam_count = sum(1 for keyword in spam_keywords if keyword in text)
    return spam_count >= 2

@app.get("/")
def home():
    if os.path.exists(landing_page_path): return FileResponse(landing_page_path)
    return {"message": "API is running."}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict_spam(email: EmailRequest):
    if model is None or vectorizer is None or label_map is None:
        raise HTTPException(status_code=500, detail="Model resources not fully loaded")

    if check_whitelist(email.sender):
        return {"label": "Not Spam", "confidence": 1.0, "reason": "Sender is in your personal whitelist."}
    
    if is_trusted_domain(email.sender):
        return {"label": "Not Spam", "confidence": 0.95, "reason": "Recognized as a globally trusted domain."}
    
    if has_spam_indicators(email.subject, email.body):
        return {"label": "Spam", "confidence": 0.95, "reason": "Contains multiple severe phishing/spam indicators."}

    full_text = f"{email.subject} {email.body}"
    processed_text = preprocess_text(full_text)
    vectorized_text = vectorizer.transform([processed_text])
    
    spam_idx = label_map.get('spam', 1)
    ham_idx = label_map.get('ham', 0)
    
    proba = model.predict_proba(vectorized_text)[0]
    spam_probability = float(proba[spam_idx])
    
    # EXACT SAME OUTPUT CONTRACT FOR YOUR EXTENSION
    if spam_probability >= 0.5:
        return {
            "label": "Spam",
            "confidence": round(spam_probability, 2),
            "reason": "AI detected suspicious language patterns.",
            "analysis": f"High probability of spam ({spam_probability:.1%}).",
            "model_version": "v2.0 (Pro)"
        }
    else:
        return {
            "label": "Not Spam",
            "confidence": round(float(proba[ham_idx]), 2),
            "reason": "AI analysis looks normal.",
            "analysis": f"Appears legitimate ({float(proba[ham_idx]):.1%}).",
            "model_version": "v2.0 (Pro)"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)