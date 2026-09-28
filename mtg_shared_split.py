"""One immutable card-name split and one test-score format for all MTG models."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import GroupShuffleSplit


SOURCE_COLUMNS = {
    "name", "cmc", "type", "subtypes", "supertypes", "layout", "rarity",
    "colors", "color_identity", "power", "toughness", "loyalty", "text",
}
DEFAULT_SPLIT = Path(__file__).resolve().parent / "runs" / "universal_split.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "runs" / "universal_comparison"


@dataclass
class SharedSplit:
    cards: pd.DataFrame
    target: np.ndarray
    groups: np.ndarray
    source_rows: np.ndarray  # Zero-based row positions below the spreadsheet header.
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    dataset_name: str
    dataset_sha256: str
    split_sha256: str
    seed: int


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_hash(manifest: dict) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_shared_split(xlsx: Path, split_file: Path = DEFAULT_SPLIT,
                      seed: int = 42) -> SharedSplit:
    """Load eligible cards and create or verify the same saved split for every model."""
    xlsx, split_file = Path(xlsx), Path(split_file)
    if not xlsx.is_file():
        raise FileNotFoundError(xlsx)
    dataset_sha256 = _file_sha256(xlsx)
    raw = pd.read_excel(xlsx, sheet_name=0,
                        usecols=lambda name: name in SOURCE_COLUMNS, engine="openpyxl")
    missing = SOURCE_COLUMNS.difference(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    cmc = pd.to_numeric(raw["cmc"], errors="coerce")
    names = raw["name"].fillna("").astype(str).str.strip()
    keep = cmc.between(0, 10) & names.ne("")
    source_rows = np.flatnonzero(keep.to_numpy()).astype(np.int64)
    cards = raw.loc[keep].reset_index(drop=True)
    target = cmc.loc[keep].round().astype(np.int64).to_numpy()
    groups = names.loc[keep].str.casefold().to_numpy()
    if not len(cards):
        raise ValueError("No eligible cards: require a nonblank name and CMC in [0, 10]")

    if split_file.exists():
        with split_file.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        split_sha256 = manifest.pop("split_sha256", None)
        if split_sha256 != _manifest_hash(manifest):
            raise ValueError(f"Split manifest was modified: {split_file}")
        if (manifest.get("dataset_sha256") != dataset_sha256
                or manifest.get("seed") != seed
                or manifest.get("version") != 1):
            raise ValueError(f"Split does not match this dataset/seed: {split_file}")
    else:
        first = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        train_idx, temp_idx = next(first.split(cards, target, groups))
        second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
        val_rel, test_rel = next(second.split(cards.iloc[temp_idx], target[temp_idx], groups[temp_idx]))
        val_idx, test_idx = temp_idx[val_rel], temp_idx[test_rel]
        manifest = {
            "version": 1,
            "dataset_sha256": dataset_sha256,
            "dataset_name": xlsx.name,
            "seed": seed,
            "target": "rounded CMC; raw CMC in [0, 10]; nonblank card name",
            "group": "stripped, casefolded card name",
            "train_rows": source_rows[train_idx].tolist(),
            "validation_rows": source_rows[val_idx].tolist(),
            "test_rows": source_rows[test_idx].tolist(),
        }
        split_sha256 = _manifest_hash(manifest)
        split_file.parent.mkdir(parents=True, exist_ok=True)
        with split_file.open("w", encoding="utf-8") as stream:
            json.dump({**manifest, "split_sha256": split_sha256}, stream)

    position = {int(row): i for i, row in enumerate(source_rows)}
    try:
        train_idx = np.array([position[row] for row in manifest["train_rows"]], dtype=np.int64)
        val_idx = np.array([position[row] for row in manifest["validation_rows"]], dtype=np.int64)
        test_idx = np.array([position[row] for row in manifest["test_rows"]], dtype=np.int64)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Split references invalid or ineligible rows: {split_file}") from exc
    all_idx = np.concatenate((train_idx, val_idx, test_idx))
    if (len(all_idx) != len(cards) or len(np.unique(all_idx)) != len(cards)
            or any(len(idx) == 0 for idx in (train_idx, val_idx, test_idx))):
        raise ValueError("Split must partition every eligible row exactly once")
    group_sets = [set(groups[idx]) for idx in (train_idx, val_idx, test_idx)]
    if group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2]:
        raise ValueError("Card names overlap between split partitions")
    print(f"Shared split {split_sha256[:12]}: {len(train_idx):,} train / "
          f"{len(val_idx):,} validation / {len(test_idx):,} test rows", flush=True)
    return SharedSplit(cards, target, groups, source_rows, train_idx, val_idx,
                       test_idx, xlsx.name, dataset_sha256, split_sha256, seed)


def score_probabilities(probabilities: np.ndarray, y_true: np.ndarray,
                        test_groups: np.ndarray, classes: np.ndarray | None = None
                        ) -> tuple[dict, np.ndarray]:
    """Score every model identically on untouched test rows."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    classes = np.arange(11) if classes is None else np.asarray(classes, dtype=np.int64)
    if probabilities.shape != (len(y_true), len(classes)):
        raise ValueError("Probability matrix shape does not match the test set/classes")
    probabilities = np.clip(probabilities, 1e-15, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predicted = classes[probabilities.argmax(axis=1)]
    expected = probabilities @ classes.astype(np.float64)
    by_name = pd.DataFrame({
        "name": test_groups,
        "exact": predicted == y_true,
        "within_one": np.abs(predicted - y_true) <= 1,
        "class_abs_error": np.abs(predicted - y_true),
        "expected_abs_error": np.abs(expected - y_true),
    }).groupby("name", sort=False).mean()
    metrics = {
        "exact_accuracy": float(np.mean(predicted == y_true)),
        "within_one_accuracy": float(np.mean(np.abs(predicted - y_true) <= 1)),
        "class_mae": float(np.mean(np.abs(predicted - y_true))),
        "expected_value_mae": float(np.mean(np.abs(expected - y_true))),
        "name_weighted_exact_accuracy": float(by_name["exact"].mean()),
        "name_weighted_within_one_accuracy": float(by_name["within_one"].mean()),
        "name_weighted_class_mae": float(by_name["class_abs_error"].mean()),
        "name_weighted_expected_value_mae": float(by_name["expected_abs_error"].mean()),
        "test_card_names": int(len(by_name)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "log_loss": float(log_loss(y_true, probabilities, labels=classes)),
    }
    return metrics, predicted


def save_benchmark_result(split: SharedSplit, output_dir: Path, model_name: str,
                          metrics: dict, predicted: np.ndarray) -> Path:
    """Merge one result only when dataset and test split match existing scores."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "dataset": split.dataset_name,
        "dataset_sha256": split.dataset_sha256,
        "split_sha256": split.split_sha256,
        "target": "rounded CMC 0-10",
        "counts": {"train": len(split.train_idx), "validation": len(split.val_idx),
                   "test": len(split.test_idx)},
        "excluded_predictors": ["name", "mana_cost", "set", "cmc"],
        "models": {},
    }
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as stream:
            previous = json.load(stream)
        for key in results.keys() - {"models"}:
            if previous.get(key) != results[key]:
                raise ValueError(f"Comparison has different {key}; use a new output directory")
        results = previous
    results["models"][model_name] = metrics
    test_idx = split.test_idx
    if len(predicted) != len(test_idx):
        raise ValueError("Test predictions do not match the shared test rows")
    pd.DataFrame({"name": split.cards.iloc[test_idx]["name"].to_numpy(),
                  "actual_cmc": split.target[test_idx],
                  "predicted_cmc": predicted}).to_csv(
        output_dir / f"{model_name}_test_predictions.csv", index=False)
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
    lines = [
        "# MTG mana-value model comparison",
        "",
        f"Dataset: `{split.dataset_name}`. Split ID: `{split.split_sha256[:12]}`.",
        "Rows are grouped by normalized card name; no card name crosses train, validation, or test.",
        f"Rows: {len(split.train_idx):,} train / {len(split.val_idx):,} validation / {len(split.test_idx):,} test.",
        "",
        "| Model | Exact accuracy | Within 1 | Class MAE | Expected-value MAE | Name-weighted exact |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("xgboost", "catboost", "compact_mlp", "expanded_mlp"):
        if name not in results["models"]:
            continue
        score = results["models"][name]
        lines.append(f"| {name} | {score['exact_accuracy']:.2%} | "
                     f"{score['within_one_accuracy']:.2%} | "
                     f"{score['class_mae']:.3f} | "
                     f"{score['expected_value_mae']:.3f} | "
                     f"{score['name_weighted_exact_accuracy']:.2%} |")
    lines += ["", "Models use different feature sets and checkpoint-selection rules; "
              "this table compares their complete pipelines, not architecture alone.", ""]
    (output_dir / "COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved comparison to {metrics_path}", flush=True)
    for name, score in results["models"].items():
        print(f"  {name}: exact={score['exact_accuracy']:.2%}, "
              f"within 1={score['within_one_accuracy']:.2%}, "
              f"class MAE={score['class_mae']:.3f}", flush=True)
    return metrics_path
