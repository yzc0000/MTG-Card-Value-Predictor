"""Compare XGBoost and CatBoost on unseen MTG card names.

The target is mana value (CMC) 0-10. Mana cost, card name, and set are never
predictors. Both models receive the same structured and rules-text features.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from mtg_shared_split import (
    DEFAULT_SPLIT, SharedSplit, load_shared_split, save_benchmark_result,
    score_probabilities,
)


SOURCE_COLUMNS = {
    "name", "cmc", "type", "subtypes", "supertypes", "layout", "rarity",
    "colors", "color_identity", "power", "toughness", "loyalty", "text",
}
CARD_TYPES = ("creature", "instant", "sorcery", "artifact", "enchantment",
              "planeswalker", "land", "battle")
TEXT_PATTERNS = {
    "destroy": r"\bdestroy\b", "exile": r"\bexile\b",
    "counter_target": r"\bcounter target\b", "sacrifice": r"\bsacrifice\b",
    "discard": r"\bdiscard\b", "return": r"\breturn\b",
    "tap": r"\btap\b", "damage": r"\bdamage\b",
    "draw": r"\bdraw\b", "scry": r"\bscry\b",
    "surveil": r"\bsurveil\b", "search_library": r"\bsearch your library\b",
    "create": r"\bcreate\b", "token": r"\btoken\b",
    "treasure": r"\btreasure\b", "flying": r"\bflying\b",
    "trample": r"\btrample\b", "haste": r"\bhaste\b",
    "lifelink": r"\blifelink\b", "deathtouch": r"\bdeathtouch\b",
    "equip": r"\bequip\b", "enchant": r"\benchant\b",
    "each": r"\beach\b", "all": r"\ball\b",
    "choose": r"\bchoose\b", "up_to": r"\bup to\b",
}


@dataclass
class BenchmarkData:
    cards: pd.DataFrame
    features: pd.DataFrame
    target: np.ndarray
    groups: np.ndarray
    categorical: list[str]
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    dataset_name: str
    seed: int
    split: SharedSplit


def parse_list(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    raw = str(value).strip()
    if not raw:
        return []
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)):
                return [str(item) for item in parsed if item is not None]
        except (SyntaxError, ValueError):
            return []
    return [raw]


def color_key(identity: object, colors: object) -> str:
    items = parse_list(identity) or parse_list(colors)
    present = {item.strip().upper() for item in items}
    return "".join(color for color in "WUBRG" if color in present) or "C"


def first_subtype(subtypes: object, type_line: object) -> str:
    items = parse_list(subtypes)
    if items:
        return items[0].lower()
    parts = re.split(r"[—–]", str(type_line), maxsplit=1)
    return parts[1].strip().split(" ")[0].lower() if len(parts) == 2 else "__none__"


def make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    x = pd.DataFrame(index=df.index)
    type_line = df["type"].fillna("").astype(str).str.lower()
    rules = df["text"].fillna("").astype(str).str.lower()

    x["layout"] = df["layout"].fillna("__missing__").astype(str).str.lower()
    x["rarity"] = df["rarity"].fillna("__missing__").astype(str).str.lower()
    x["primary_type"] = type_line.str.extract(
        r"\b(creature|instant|sorcery|artifact|enchantment|planeswalker|land|battle)\b",
        expand=False,
    ).fillna("__other__")
    x["primary_subtype"] = [first_subtype(s, t) for s, t in zip(df["subtypes"], df["type"])]
    x["color_identity"] = [color_key(i, c) for i, c in zip(df["color_identity"], df["colors"])]
    categorical = ["layout", "rarity", "primary_type", "primary_subtype", "color_identity"]

    for card_type in CARD_TYPES:
        x[f"is_{card_type}"] = type_line.str.contains(rf"\b{card_type}\b", regex=True).astype("float32")
    for supertype in ("legendary", "basic", "snow"):
        x[f"is_{supertype}"] = type_line.str.contains(rf"\b{supertype}\b", regex=True).astype("float32")
    for color in "WUBRG":
        x[f"has_{color}"] = x["color_identity"].str.contains(color, regex=False).astype("float32")
    x["color_count"] = x["color_identity"].str.replace("C", "", regex=False).str.len().astype("float32")
    for name in ("power", "toughness", "loyalty"):
        x[name] = pd.to_numeric(df[name], errors="coerce").astype("float32")
        x[f"{name}_missing"] = x[name].isna().astype("float32")
    x["rules_chars"] = rules.str.len().astype("float32")
    x["rules_words"] = rules.str.count(r"\b\w+\b").astype("float32")
    x["rules_lines"] = (rules.str.count("\n") + rules.ne("").astype(int)).astype("float32")
    x["rules_digits"] = rules.str.count(r"\d").astype("float32")
    x["rules_mana_symbols"] = rules.str.count(r"\{").astype("float32")
    for name, pattern in TEXT_PATTERNS.items():
        x[f"rules_{name}"] = rules.str.count(pattern).astype("float32")
    return x, categorical


def evaluate(model: object, x_test: pd.DataFrame, y_test: np.ndarray,
             test_groups: np.ndarray) -> tuple[dict, np.ndarray]:
    return score_probabilities(model.predict_proba(x_test), y_test, test_groups,
                               np.asarray(model.classes_, dtype=np.int64))


def xgboost_frames(x: pd.DataFrame, categorical: list[str], train_idx: np.ndarray,
                   val_idx: np.ndarray, test_idx: np.ndarray) -> tuple[pd.DataFrame, ...]:
    frames = [x.iloc[idx].copy() for idx in (train_idx, val_idx, test_idx)]
    for column in categorical:
        categories = sorted(x.iloc[train_idx][column].unique().tolist())
        dtype = CategoricalDtype(categories=categories)
        for frame in frames:
            frame[column] = frame[column].astype(dtype)
    return tuple(frames)


def catboost_frames(x: pd.DataFrame, categorical: list[str], train_idx: np.ndarray,
                    val_idx: np.ndarray, test_idx: np.ndarray) -> tuple[pd.DataFrame, ...]:
    frames = [x.iloc[idx].copy() for idx in (train_idx, val_idx, test_idx)]
    for frame in frames:
        for column in categorical:
            frame[column] = frame[column].astype(str)
    return tuple(frames)


def load_benchmark_data(xlsx: Path, seed: int,
                        split_file: Path = DEFAULT_SPLIT) -> BenchmarkData:
    """Use one feature definition and card-name split for both algorithms."""
    print(f"Loading {xlsx.name}...", flush=True)
    split = load_shared_split(xlsx, split_file, seed)
    print(f"Eligible rows: {len(split.cards):,}; unique card names: "
          f"{len(np.unique(split.groups)):,}", flush=True)
    x, categorical = make_features(split.cards)
    print(f"Features: {x.shape[1]} ({len(categorical)} categorical)", flush=True)
    return BenchmarkData(split.cards, x, split.target, split.groups, categorical,
                         split.train_idx, split.val_idx, split.test_idx,
                         split.dataset_name, seed, split)


def save_result(data: BenchmarkData, output_dir: Path, model_name: str,
                metrics: dict, predicted: np.ndarray) -> Path:
    """Merge one model's test scores into the shared comparison file."""
    metrics["feature_count"] = int(data.features.shape[1])
    return save_benchmark_result(data.split, output_dir, model_name, metrics, predicted)
