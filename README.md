# Predicting MTG mana value without mana cost

Can a model estimate a Magic: The Gathering card's mana value from its other printed characteristics? This project treats the question as an 11-class prediction task (rounded mana value, 0–10) and compares four complete pipelines: XGBoost, CatBoost, a compact multilayer perceptron (MLP), and an expanded MLP.

The printed mana cost, card name, set, and target CMC are **not** supplied as predictors. That makes this a prediction of design patterns, not a parser for mana symbols in the cost. The source spreadsheet is not included in this repository.

## Why this is difficult

Mana value is determined by the mana cost, which is intentionally hidden from the models. A card's type, stats, colors, and rules text provide clues, but do not uniquely determine its cost. Similar effects may appear on cards with different costs because of color requirements, timing, restrictions, and design choices. Rules text can also contain mana symbols for abilities, but those are not the card's printed mana cost.

The dataset also contains multiple printings of some cards. A random row split could put the same named card in both training and test sets, inflating the apparent ability to generalize. Finally, the 11 target classes are uneven: in the held-out test set, classes 9 and 10 contain only 25 and 3 rows, respectively. A single headline accuracy cannot describe performance on every class.

## Approach

1. Read card attributes and rules text from a local spreadsheet. Keep rows with a nonblank name and numeric CMC from 0 to 10; round CMC to form the class target.
2. Split by stripped, case-folded card name with seed 42, so no card name crosses training, validation, and test. The saved benchmark has **75,229 / 8,732 / 9,491** rows in those partitions, representing 93,452 eligible rows.
3. Build two feature families. The tree models share 59 structured and rules-text-count features. The MLPs use stable hashed categorical/type cues plus numeric properties and rules-text indicators; the expanded MLP has a larger representation and more text patterns.
4. Fit each model on training data and use validation data for early stopping or checkpoint selection. Evaluate the selected model once on the held-out test rows using the same scoring function.

All normalization and categorical vocabularies that require fitting are based on training rows. The split manifest checks the source spreadsheet checksum and split checksum before reuse. The code excludes `name`, `mana_cost`, `set`, and `cmc` from model input. The group key is used to partition and report scores, not as a predictive feature.

## Four model designs

| Model | Representation | Learner and selection |
| --- | --- | --- |
| **XGBoost** (`train_xgboost.py`) | 59 features, including five categorical columns, type/color indicators, stats, and counts of rules-text effects | Multiclass boosted trees; validation multiclass log loss; early stopping after 60 rounds without improvement. Baseline best iteration: 227 of 600 requested. |
| **CatBoost** (`train_catboost.py`) | The same 59 features; native handling of the five categorical columns | Multiclass boosted trees; validation `MultiClass` loss; early stopping after 60 rounds without improvement. Baseline best iteration: 590 of 1,500 requested. |
| **Compact MLP** (`train_mlp_compact.py`) | 512 hashed categorical dimensions + 45 numeric features (557 inputs) | Hidden widths 512 → 256 → 128 with dropout; class-weighted cross-entropy and AdamW; checkpoint with highest validation exact accuracy. Baseline trained for 150 epochs. |
| **Expanded MLP** (`train_mlp_expanded.py`) | 1,024 hashed categorical dimensions + 80 numeric features (1,104 inputs) | Hidden widths 768 → 512 → 256 → 128 with dropout; class-weighted cross-entropy and AdamW; checkpoint with lowest validation expected-value MAE. |

The tree models are a controlled comparison of XGBoost and CatBoost on the same inputs. The MLPs differ in feature engineering, representation size, architecture, and checkpoint criterion. Consequently, the four-way table below compares **pipelines**, not the isolated effect of model architecture.

## Held-out results

All results below are from the same 9,491-row test partition. The metrics and dataset/split checksums are preserved in [`results/metrics.json`](results/metrics.json). No training run is needed to inspect them.

| Model | Exact accuracy ↑ | Within ±1 ↑ | Class MAE ↓ | Expected-value MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 46.72% | 80.56% | 0.822 | 0.768 |
| **CatBoost** | **49.31%** | **82.86%** | **0.765** | 0.766 | 
| Compact MLP | 47.30% | 80.23% | 0.870 | 0.779 |
| Expanded MLP | 48.67% | 81.07% | 0.803 | **0.753** |

**Interpretation.** CatBoost has the highest exact accuracy, within-one accuracy, and lowest class MAE in this run. The expanded MLP has the lowest expected-value MAE, where the prediction is the probability-weighted average of classes 0–10 rather than the most likely class. Those answer slightly different questions: “which whole-number class?” versus “what is the average predicted mana value?” Neither result establishes one model as universally best.

Name-weighted accuracy averages the exact-match rate within each unique card name, then averages across names. It gives frequently reprinted cards less weight than row-level accuracy does. Its lower values show why the unit of evaluation matters even with a name-grouped split. For metric definitions, class distribution, and interpretation limits, see [EVALUATION.md](EVALUATION.md).

## Repository map

| File | Purpose |
| --- | --- |
| `mtg_shared_split.py` | Dataset eligibility, reproducible name-grouped split, checksum validation, shared scoring, and result export. |
| `mtg_boosting_benchmark.py` | Shared 59-feature construction and train-only categorical handling for both tree models. |
| `train_xgboost.py`, `train_catboost.py` | Separate boosting training entry points. |
| `train_mlp_compact.py`, `train_mlp_expanded.py` | Separate PyTorch feature pipelines, architectures, training loops, and checkpoint selection. |
| `results/metrics.json` | Saved test metrics and benchmark identity; no card rows or model weights. |
| `EVALUATION.md` | Protocol, metric definitions, observed class imbalance, and limitations. |

## Running the code

The code expects a local Excel workbook with the columns named in `mtg_shared_split.py`. The benchmark workbook was named `all_mtg_cards (2).xlsx`; it is not redistributed here. Use Python 3.11. The tabular dependencies are pinned in `requirements-boosting.txt`; install a PyTorch build appropriate for your CPU/CUDA environment for the MLPs.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements-boosting.txt
# Install PyTorch separately for your platform before running the MLPs.
```

From the repository root, run models sequentially (especially when sharing one GPU):

```powershell
$data = '.\all_mtg_cards (2).xlsx'
$out = '.\runs\universal_comparison'
& .\.venv\Scripts\python.exe .\train_xgboost.py --xlsx $data --output-dir $out --device cuda
& .\.venv\Scripts\python.exe .\train_catboost.py --xlsx $data --output-dir $out --device cuda
& .\.venv\Scripts\python.exe .\train_mlp_compact.py --xlsx $data --output-dir $out --out-prefix "$out\compact_mlp" --epochs 150
& .\.venv\Scripts\python.exe .\train_mlp_expanded.py --xlsx $data --output-dir $out --out-prefix "$out\expanded_mlp" --epochs 150
```

The shared split manifest is generated under `runs/` on the first run and verified by the other scripts. The example caps both MLP training runs at 150 epochs; a new run's scores can differ from the saved benchmark because of training stochasticity and checkpoint selection. Boosting scripts support `--device cpu` if CUDA is unavailable. `runs/`, checkpoints, predictions, and the workbook are git-ignored and are not part of the published repository.

## Scope

This is a benchmark on one dataset and one grouped holdout, not a claim that mana value can be inferred exactly from every card's visible properties. The relatively small high-CMC classes, lack of repeated-split uncertainty estimates, and the different MLP feature sets limit stronger conclusions. A useful next experiment would compare per-class errors and confidence calibration across multiple grouped splits, while keeping feature definitions fixed when the aim is to isolate architecture.

