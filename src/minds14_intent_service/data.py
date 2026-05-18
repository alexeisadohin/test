from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


@dataclass(frozen=True)
class DatasetColumns:
    """Сохраняет имена столбцов обнаруженного набора данных."""

    text: str
    label: str
    audio: Optional[str]


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    """Возвращает первый существующий в датасете столбец из списка кандидатов."""
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def load_labeled_split(dataset_name: str, language_config: str) -> Dataset:
    """Загружает основной размеченный сплит для языковой конфигурации датасета."""
    raw = load_dataset(dataset_name, language_config)
    if isinstance(raw, DatasetDict):
        if "train" in raw:
            return raw["train"]
        return raw[next(iter(raw.keys()))]
    return raw


def detect_columns(dataset: Dataset) -> DatasetColumns:
    """Определяет в датасете текстовый столбец, метку и необязательный аудиостолбец."""
    columns = dataset.column_names
    text_col = first_existing(columns, ["transcription", "text", "utterance", "sentence", "transcript"])
    label_col = first_existing(columns, ["intent_class", "label", "intent"])
    audio_col = first_existing(columns, ["audio", "speech"])

    if text_col is None:
        raise ValueError(f"Text column not found. Available columns: {columns}")
    if label_col is None:
        raise ValueError(f"Intent label column not found. Available columns: {columns}")

    return DatasetColumns(text=text_col, label=label_col, audio=audio_col)


def label_names(dataset: Dataset, label_col: str) -> Optional[Dict[int, str]]:
    """Возвращает отображение числовых меток в имена для столбцов ClassLabel."""
    feature = dataset.features[label_col]
    names = getattr(feature, "names", None)
    if names:
        return {idx: name for idx, name in enumerate(names)}
    return None


def make_text_frame(dataset: Dataset, columns: DatasetColumns) -> pd.DataFrame:
    """Собирает очищенный DataFrame с текстами и метками из датасета Hugging Face."""
    keep = [columns.text, columns.label]
    df = dataset.select_columns(keep).to_pandas().dropna().reset_index(drop=True)

    names = label_names(dataset, columns.label)
    if names is not None:
        df[columns.label] = df[columns.label].map(lambda value: names[int(value)])
    else:
        df[columns.label] = df[columns.label].astype(str)

    df[columns.text] = df[columns.text].astype(str).str.strip()
    df = df[df[columns.text].str.len() > 0].reset_index(drop=True)
    return df
