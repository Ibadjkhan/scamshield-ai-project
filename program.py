"""Machine-learning utilities for the ScamShield prototype.

The public functions are deliberately small so they can be explained in a demo:
load_data validates/preprocesses a CSV, train_and_compare evaluates four models,
and predict makes one web-app prediction.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# Normalise a few frequent Roman-Urdu spelling variants before vectorisation.
SPELLING = {"inam": "inaam", "mubarik": "mubarak", "rupay": "rupee",
            "rupaye": "rupee", "rs": "rupee", "foran": "fauran",
            "karen": "karein", "bhejen": "bhejein", "msg": "message",
            "plz": "please", "pls": "please"}


def clean_text(text: object) -> str:
    """Convert one message to consistent tokens without deleting useful scam clues."""
    text = str(text).lower()
    text = re.sub(r"(https?://\S+|www\.\S+|\b[\w-]+\.(?:com|net|org|xyz|top|site|online|link|cc|pk|ly)\b\S*)", " urltoken ", text)
    text = re.sub(r"(?:\+?92[-\s]?|\b0)3\d{2}[-\s]?\d{7}\b|\b0\d{9,10}\b|\b\d{10,12}\b", " phonetoken ", text)
    text = re.sub(r"\d+", " numtoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(SPELLING.get(word, word) for word in text.split())


def load_data(path: str | Path) -> tuple[pd.Series, pd.Series, dict[str, int]]:
    """Read text/label CSV, remove invalid rows/duplicate texts, and encode labels.

    scam becomes 1 (the positive class); legit becomes 0.  The returned audit is
    shown in the console and makes preprocessing transparent in the report.
    """
    df = pd.read_csv(path)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns: text, label")
    audit = {"rows_read": len(df), "missing_rows_removed": 0, "duplicates_removed": 0}
    valid = df.dropna(subset=["text", "label"]).copy()
    valid = valid[valid["text"].astype(str).str.strip().ne("")]
    valid = valid[valid["label"].isin(["scam", "legit"])]
    audit["missing_rows_removed"] = audit["rows_read"] - len(valid)
    before_duplicates = len(valid)
    valid = valid.drop_duplicates(subset="text")
    audit["duplicates_removed"] = before_duplicates - len(valid)
    audit["rows_used"] = len(valid)
    return valid["text"], (valid["label"] == "scam").astype(int), audit


def load_and_split(path: str | Path, test_size: float = 0.20, random_state: int = 42):
    """Load one dataset then make a repeatable, stratified train/test split.

    Stratification keeps the scam/legit proportions similar in both splits.
    The fixed random state makes every group member obtain the same result.
    """
    X, y, audit = load_data(path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    audit["training_rows"] = len(X_train)
    audit["testing_rows"] = len(X_test)
    return X_train, X_test, y_train, y_test, audit


def _tfidf() -> TfidfVectorizer:
    """Build TF-IDF features. L2 normalisation is built in, so extra scaling is unnecessary."""
    return TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), sublinear_tf=True)


def get_models() -> dict[str, object]:
    """Return the four classifiers assessed in this project."""
    return {
        "Naive Bayes": make_pipeline(_tfidf(), MultinomialNB(alpha=0.1)),
        "Logistic Regression": make_pipeline(_tfidf(), LogisticRegression(C=10, max_iter=1000, class_weight="balanced", random_state=42)),
        "Linear SVM": make_pipeline(_tfidf(), CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=42), cv=5)),
        "Random Forest": make_pipeline(_tfidf(), RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)),
    }


def train_and_compare(X_train, y_train, X_test, y_test):
    """Fit all models and select the one with highest scam-class F1 score."""
    rows, trained = [], {}
    for name, model in get_models().items():
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        rows.append({"Model": name, "Accuracy": accuracy_score(y_test, predicted),
                     "Precision": precision_score(y_test, predicted, zero_division=0),
                     "Recall": recall_score(y_test, predicted, zero_division=0),
                     "F1": f1_score(y_test, predicted, zero_division=0)})
        trained[name] = model
    results = pd.DataFrame(rows).set_index("Model").sort_values("F1", ascending=False).round(3)
    best_name = results.index[0]
    return results, best_name, trained[best_name]


def build_signal_profiles(model, X_train, y_train) -> dict[str, float]:
    """Build transparent word/phrase evidence from the training data only.

    A positive score means a TF-IDF feature occurred relatively more in scam
    messages; a negative score means it occurred relatively more in legitimate
    messages. These scores explain input evidence, not a guarantee of intent.
    """
    vectorizer = model.named_steps["tfidfvectorizer"]
    matrix = vectorizer.transform(X_train)
    scores = matrix[y_train.to_numpy() == 1].mean(axis=0).A1 - matrix[y_train.to_numpy() == 0].mean(axis=0).A1
    return dict(zip(vectorizer.get_feature_names_out(), scores.tolist()))


def explain_prediction(model, message: str, signal_profiles: dict[str, float]) -> dict:
    """Return prediction plus the strongest matching scam/legit text signals."""
    label, probability = predict(model, message)
    tokens = set(clean_text(message).split())
    # Include phrase features as well as individual cleaned words.
    words = clean_text(message).split()
    tokens.update(" ".join(words[index:index + 2]) for index in range(len(words) - 1))
    evidence = [(term, score) for term, score in signal_profiles.items() if term in tokens and abs(score) > 0]
    evidence.sort(key=lambda item: abs(item[1]), reverse=True)
    scam_signals = [term for term, score in evidence if score > 0][:5]
    legit_signals = [term for term, score in evidence if score < 0][:5]
    return {"label": label, "probability": probability, "threshold": 50,
            "scam_signals": scam_signals, "legit_signals": legit_signals}


def predict(model, message: str) -> tuple[str, float]:
    """Return the app-friendly decision and calibrated probability percentage."""
    probability = float(model.predict_proba([message])[0][1])
    return ("Scam" if probability >= 0.50 else "Not Scam"), round(probability * 100, 1)
