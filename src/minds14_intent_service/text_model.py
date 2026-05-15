from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


@dataclass
class IntentPrediction:
    """Store a predicted intent label and confidence score."""

    label: str
    score: float


def build_text_classifier(random_state: int) -> Pipeline:
    """Create the TF-IDF plus Logistic Regression intent classifier."""
    features = FeatureUnion(
        transformer_list=[
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=50_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("features", features),
            (
                "clf",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def predict_intent(model: Pipeline, text: str) -> IntentPrediction:
    """Predict intent label and score for a single text input."""
    clean_text = text.strip()
    if not clean_text:
        raise ValueError("Text must not be empty.")

    probabilities = model.predict_proba([clean_text])[0]
    best_idx = int(np.argmax(probabilities))
    return IntentPrediction(label=str(model.classes_[best_idx]), score=float(probabilities[best_idx]))


def save_artifact(path: Path, model: Pipeline, metadata: Dict[str, Any]) -> None:
    """Persist a trained model and metadata to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, path)


def load_artifact(path: Path) -> Tuple[Pipeline, Dict[str, Any]]:
    """Load a trained model artifact and its metadata."""
    artifact = joblib.load(path)
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact["model"], artifact.get("metadata", {})
    return artifact, {}
