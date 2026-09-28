"""Train an XGBoost classifier for MTG mana value, then test unseen card names."""

import argparse
import time
from pathlib import Path

from xgboost import XGBClassifier
from mtg_shared_split import DEFAULT_SPLIT

from mtg_boosting_benchmark import (
    evaluate,
    load_benchmark_data,
    save_result,
    xgboost_frames,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--xlsx", type=Path, required=True, help="MTG card spreadsheet")
    parser.add_argument("--output-dir", type=Path, required=True, help="Comparison output folder")
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT,
                        help="Shared train/validation/test manifest")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--iterations", type=int, default=600, help="Maximum boosting rounds")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = load_benchmark_data(args.xlsx, args.seed, args.split_file)
    x_train, x_val, x_test = xgboost_frames(
        data.features, data.categorical, data.train_idx, data.val_idx, data.test_idx)
    y = data.target
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Multiclass probabilities support both class and expected-value error measures.
    # Validation log loss controls the 60-round early-stopping rule.
    model = XGBClassifier(
        objective="multi:softprob", num_class=11,
        n_estimators=args.iterations, max_depth=6, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        tree_method="hist", device=args.device,
        enable_categorical=True, max_cat_to_onehot=16,
        eval_metric="mlogloss", early_stopping_rounds=60,
        random_state=args.seed, n_jobs=4,
    )
    print(f"Training XGBoost on {args.device}...", flush=True)
    start = time.monotonic()
    model.fit(x_train, y[data.train_idx],
              eval_set=[(x_val, y[data.val_idx])], verbose=100)
    model.save_model(args.output_dir / "xgboost.json")

    metrics, predicted = evaluate(model, x_test, y[data.test_idx],
                                  data.groups[data.test_idx])
    metrics.update({
        "best_iteration": int(model.best_iteration),
        "fit_seconds": round(time.monotonic() - start, 1),
        "device": args.device,
        "iterations_requested": args.iterations,
    })
    print(f"XGBoost test metrics: {metrics}", flush=True)
    save_result(data, args.output_dir, "xgboost", metrics, predicted)


if __name__ == "__main__":
    main()
