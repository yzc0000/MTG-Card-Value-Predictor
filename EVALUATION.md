# Evaluation protocol and interpretation

## What is measured

The target is rounded CMC in classes 0–10. Eligible rows have a nonblank card name and numeric raw CMC within that range. The benchmark contains 93,452 rows and 29,003 distinct normalized card names. It was partitioned by card name, not by row, with seed 42:

| Partition | Rows | Use |
| --- | ---: | --- |
| Training | 75,229 | Model fitting and train-only normalization/vocabularies |
| Validation | 8,732 | Early stopping or checkpoint selection |
| Test | 9,491 | Final reported score after selection |

The workbook checksum is `c0429f68e2a9a423a5566b9433868a6aae5f9b6e3a4fc694b6ce757b8a891010`; the split checksum is `4b74c9a76097fceca0e754025b2d68e0702f1a7eec97b64ab8e1b312fddb7d05`. `mtg_shared_split.py` verifies a saved manifest before using it. These hashes identify the evaluated data and split; the workbook and row-level predictions are not published here.

## Metric definitions

- **Within-one accuracy:** fraction for which the most probable class differs by no more than one.
- **Class MAE:** mean absolute error of the most probable integer class.
- **Expected-value MAE:** mean absolute error of the probability-weighted average over classes 0–10.
- **Name-weighted within-one accuracy:** mean of per-name within-one rates, giving each distinct test card name equal weight.

All four models use the same scoring implementation in `mtg_shared_split.py`. The saved values are in `results/metrics.json`.

## Saved test comparison

| Model | Within ±1 | Class MAE | Expected-value MAE | Name-weighted within ±1 |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 80.56% | 0.822 | 0.768 | 78.61% |
| CatBoost | 82.86% | 0.765 | 0.766 | 80.67% |
| Compact MLP | 80.23% | 0.870 | 0.779 | 76.97% |
| Expanded MLP | 81.07% | 0.803 | 0.753 | 78.60% |

The test set has 2,901 distinct normalized card names. Rare high-cost classes are especially uncertain: among the 9,491 test rows, CMC 9 has 25 examples and CMC 10 has 3. The recorded test counts by class are 0: 1,583; 1: 959; 2: 1,552; 3: 1,981; 4: 1,476; 5: 1,013; 6: 539; 7: 251; 8: 109; 9: 25; 10: 3. These are descriptive counts, not per-class performance results.

## Limits of the comparison

- Card-name grouping reduces reprint leakage, but this is still one dataset and one split. No confidence interval or repeated-seed claim is made.
- XGBoost and CatBoost share their 59 features. The MLPs use different feature families and checkpoint criteria. The table compares complete pipelines, not architectures in isolation.
- The compact MLP was selected by validation classification performance; the expanded MLP by validation expected-value MAE. This can favor different outcomes even on the same test partition.
- The source data are not included. Matching the exact benchmark requires the same workbook, preprocessing code, and split hash.
- Predictions are not causal explanations of card design, and this work does not establish performance for unseen future releases or every card subtype.

