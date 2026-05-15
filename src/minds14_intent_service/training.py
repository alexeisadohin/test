from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from minds14_intent_service.data import detect_columns, load_labeled_split, make_text_frame
from minds14_intent_service.text_model import build_text_classifier, save_artifact


def evaluate_predictions(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """Compute weighted and macro classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def train_and_save(
    output_path: Path,
    dataset_name: str,
    language_config: str,
    test_size: float,
    random_state: int,
) -> Dict[str, Any]:
    """Train the baseline text classifier and save it as an artifact."""
    dataset = load_labeled_split(dataset_name, language_config)
    columns = detect_columns(dataset)
    df = make_text_frame(dataset, columns)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[columns.label],
    )

    model = build_text_classifier(random_state=random_state)
    model.fit(train_df[columns.text].tolist(), train_df[columns.label].tolist())

    y_true = test_df[columns.label].tolist()
    y_pred = model.predict(test_df[columns.text].tolist()).tolist()
    metrics = evaluate_predictions(y_true, y_pred)

    metadata: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "language_config": language_config,
        "text_column": columns.text,
        "label_column": columns.label,
        "test_size": test_size,
        "random_state": random_state,
        "labels": sorted(df[columns.label].unique().tolist()),
        "metrics": metrics,
    }
    save_artifact(output_path, model, metadata)
    return metadata
