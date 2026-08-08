# Comparison tiers for the "first contact" datasets

All numbers are means over the same 9 TabArena splits (3 folds x 3 repeats) per dataset.
Data: `tabicl_vs_traditional_by_dataset.csv` (TabICLv2 vs traditional margins, all 51 datasets)
and `naive_xgb_results.csv` (the naive tier, 7 shortlist datasets).

| Tier | What it is |
|---|---|
| Naive XGBoost | True `xgboost` library defaults (100 trees, lr 0.3, no early stopping), single model on a holdout split, no refit, no ensembling, no calibration. Run through the AutoGluon model wrapper so preprocessing matches the other tiers. |
| Traditional, tuned + ensembled | The full TabArena protocol: 201 configurations per model family (default + 200 searched), each config capped at **1 hour** of runtime, 8-fold bagging, post-hoc ensembling across the family's configs. The per-dataset best family is reported (LightGBM / CatBoost / XGBoost / RandomForest). |
| TabPFNv2 (default) | Early-2025-generation TFM, single default config, bagged per the TabArena protocol. |
| TabICLv2 (default) | 2026-generation TFM, single default config, bagged. Typically ~5s fit on a modern GPU. |
| TabFM (default) | The largest single TFM (2026), single default config, bagged. Roughly 10-20x TabICLv2's cost. |

Headline framing this supports: a single untuned TabICLv2 fit beats a 201-config
hyperparameter search with ensembling (1 hour per config), on every one of the
seven shortlist datasets. Describe the tuning budget as "201 configs, 1h per
config" — not as a single 4-hour (or unbounded) run.
