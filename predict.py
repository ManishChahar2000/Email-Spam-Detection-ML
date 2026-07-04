import joblib
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------
# Load saved model & vectorizer
# ---------------------------------------------------------
model = joblib.load("models/spam_model.pkl")
tfidf = joblib.load("models/tfidf.pkl")

# ---------------------------------------------------------
# Same cleaning logic used during training
# ---------------------------------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
NEGATIONS = {"no", "not", "nor", "never", "none"}
stop_words = stop_words - NEGATIONS

RE_URL = re.compile(r"http\S+|www\S+|https\S+")
RE_EMAIL = re.compile(r"\S+@\S+")
RE_HTML = re.compile(r"<.*?>")
RE_HTML_ENTITY = re.compile(r"&[a-z]+;|&#\d+;")
RE_NUMBERS = re.compile(r"\d+")
RE_NON_ASCII = re.compile(r"[^\x00-\x7f]")
RE_MULTI_SPACE = re.compile(r"\s+")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)

def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = RE_URL.sub(" ", text)
    text = RE_EMAIL.sub(" ", text)
    text = RE_HTML.sub(" ", text)
    text = RE_HTML_ENTITY.sub(" ", text)
    text = RE_NON_ASCII.sub(" ", text)
    text = RE_NUMBERS.sub(" ", text)
    text = text.translate(PUNCT_TABLE)
    text = RE_MULTI_SPACE.sub(" ", text).strip()
    words = text.split()
    cleaned_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) >= 2
    ]
    return " ".join(cleaned_words)

# ---------------------------------------------------------
# Prediction function
# ---------------------------------------------------------
def predict_email(email_text: str):
    cleaned = clean_text(email_text)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]
    label = "SPAM" if prediction == 1 else "HAM (Not Spam)"
    return label, cleaned

# ---------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Email Spam Detector — type an email, or 'quit' to exit")
    print("=" * 60)

    while True:
        email = input("\nEnter email text:\n> ")
        if email.strip().lower() == "quit":
            break

        label, cleaned = predict_email(email)
        print(f"\nPrediction: {label}")
        print(f"(Cleaned text used for prediction: {cleaned[:100]}...)")