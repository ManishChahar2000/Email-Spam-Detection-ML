import os
import re
import string
import logging
import multiprocessing as mp

import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
INPUT_PATH = "dataset/combined_data.csv"
OUTPUT_PATH = "dataset/cleaned_dataset.csv"
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
MIN_TOKEN_LENGTH = 2
MIN_CLEAN_TEXT_LENGTH = 3

# ---------------------------------------------------------
# NLTK resource setup (only downloads if missing)
# ---------------------------------------------------------
def ensure_nltk_resources():
    resources = {
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
    }
    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info(f"Downloading NLTK resource: {name}")
            nltk.download(name, quiet=True)

ensure_nltk_resources()

# ---------------------------------------------------------
# NLP tools (module-level so worker processes can use them)
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

# ---------------------------------------------------------
# Cleaning function
# ---------------------------------------------------------
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
        if word not in stop_words and len(word) >= MIN_TOKEN_LENGTH
    ]

    return " ".join(cleaned_words)

def clean_text_batch(texts):
    return [clean_text(t) for t in texts]

def parallel_clean(series, n_jobs=None):
    n_jobs = n_jobs or max(1, mp.cpu_count() - 1)
    chunks = [series.iloc[i::n_jobs] for i in range(n_jobs)]

    with mp.Pool(n_jobs) as pool:
        results = pool.map(clean_text_batch, [chunk.tolist() for chunk in chunks])

    cleaned = [None] * len(series)
    for i, chunk_result in enumerate(results):
        for j, val in enumerate(chunk_result):
            cleaned[i + j * n_jobs] = val
    return cleaned

# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------
def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input dataset not found at: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Expected column '{TEXT_COLUMN}' not found. Columns present: {list(df.columns)}")
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Expected column '{LABEL_COLUMN}' not found. Columns present: {list(df.columns)}")

    logger.info("=" * 60)
    logger.info(f"Original dataset shape: {df.shape}")

    before = len(df)
    df.drop_duplicates(inplace=True)
    logger.info(f"Removed {before - len(df)} duplicate rows")

    before = len(df)
    df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN], inplace=True)
    logger.info(f"Removed {before - len(df)} rows with nulls in text/label")

    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip()

    logger.info(f"After removing duplicates & nulls: {df.shape}")
    logger.info(f"Label distribution:\n{df[LABEL_COLUMN].value_counts()}")

    logger.info(f"Cleaning with multiprocessing ({max(1, mp.cpu_count()-1)} workers)...")
    df["clean_text"] = parallel_clean(df[TEXT_COLUMN])

    df["clean_text"] = df["clean_text"].fillna("")

    before = len(df)
    df = df[df["clean_text"].str.len() >= MIN_CLEAN_TEXT_LENGTH]
    logger.info(f"Removed {before - len(df)} rows that were empty/near-empty after cleaning")

    df.reset_index(drop=True, inplace=True)

    if df.empty:
        raise RuntimeError("Cleaned dataset is empty — check input data or cleaning thresholds")

    if df["clean_text"].isna().any():
        raise RuntimeError("Unexpected NaN values remain in clean_text column")

    logger.info("=" * 60)
    logger.info(f"Final cleaned dataset shape: {df.shape}")
    logger.info(f"Final label distribution:\n{df[LABEL_COLUMN].value_counts()}")
    logger.info("\nSample cleaned emails:\n%s", df[[LABEL_COLUMN, "clean_text"]].head().to_string())

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    logger.info(f"✅ Cleaned dataset saved to '{OUTPUT_PATH}'")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()