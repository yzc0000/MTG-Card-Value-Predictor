"""Train a CatBoost classifier for MTG mana value, then test unseen card names."""

import argparse
import time
from pathlib import Path

from catboost import CatBoostClassifier
from mtg_shared_split import DEFAULT_SPLIT

from mtg_boosting_benchmark import (
    catboost_frames,
    evaluate,
    load_benchmark_data,
    save_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--xlsx", type=Path, required=True, help="MTG card spreadsheet")
    parser.add_argument("--output-dir", type=Path, required=True, help="Comparison output folder")
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT,
                        help="Shared train/validation/test manifest")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--iterations", type=int, default=1500, help="Maximum boosting rounds")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = load_benchmark_data(args.xlsx, args.seed, args.split_file)
    x_train, x_val, x_test = catboost_frames(
        data.features, data.categorical, data.train_idx, data.val_idx, data.test_idx)
    y = data.target
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device_options = {"task_type": "GPU", "devices": "0"} if args.device == "cuda" else {"task_type": "CPU"}
    model = CatBoostClassifier(
        iterations=args.iterations, depth=6, learning_rate=0.05,
        loss_function="MultiClass", eval_metric="MultiClass",
        random_seed=args.seed, l2_leaf_reg=3, verbose=100,
        thread_count=4, **device_options,
    )
    print(f"Training CatBoost on {args.device}...", flush=True)
    start = time.monotonic()
    # CatBoost handles categorical columns natively. Validation MultiClass loss
    # controls the 60-round early-stopping rule; use_best_model keeps that tree.
    model.fit(x_train, y[data.train_idx], cat_features=data.categorical,
              eval_set=(x_val, y[data.val_idx]), early_stopping_rounds=60,
              use_best_model=True)
    model.save_model(str(args.output_dir / "catboost.cbm"))

    metrics, predicted = evaluate(model, x_test, y[data.test_idx],
                                  data.groups[data.test_idx])
    metrics.update({
        "best_iteration": int(model.best_iteration_),
        "fit_seconds": round(time.monotonic() - start, 1),
        "device": args.device,
        "iterations_requested": args.iterations,
    })
    print(f"CatBoost test metrics: {metrics}", flush=True)
    save_result(data, args.output_dir, "catboost", metrics, predicted)


if __name__ == "__main__":
    main()
