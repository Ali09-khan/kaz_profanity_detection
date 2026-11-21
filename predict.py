import argparse
import json
import re
import unicodedata
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).with_name("model.joblib")
CONTACT_DETAILS = re.compile(
    r"\b(?:https?://|www\.)\S+|\b[\w.+-]+@[\w.-]+\.[a-z]{2,}"
    r"|(?<!\w)@[\w.]+|(?<!\w)\+?\d(?:[ ()-]*\d){6,}(?!\w)",
    re.IGNORECASE,
)


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = CONTACT_DETAILS.sub(" ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return " ".join(text.split())


def predict(text):
    pipeline, threshold = joblib.load(MODEL_PATH)
    probability = float(pipeline.predict_proba([text])[0, 1])
    return {"label": int(probability >= threshold), "probability": probability}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect Kazakh profanity in a review.")
    parser.add_argument("text", help="One review or comment.")
    print(json.dumps(predict(parser.parse_args().text)))
