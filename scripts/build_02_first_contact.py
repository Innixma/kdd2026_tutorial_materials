"""Build notebooks/02_first_contact.ipynb.

The tutorial's first hands-on notebook: one small real dataset, three fits of increasing
sophistication — naive XGBoost, AutoGluon-configured XGBoost with bagging, then a single
default TabICLv2 — comparing test ROC AUC and fit time. Run this script to (re)generate
the notebook, then execute it with nbconvert to fill outputs.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "02_first_contact.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# First contact: from a naive baseline to a tabular foundation model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Innixma/kdd2026_tutorial_materials/blob/main/notebooks/02_first_contact.ipynb)

**Taming Structured Data Foundation Models with AutoML — KDD 2026 hands-on tutorial**

In this notebook we take one small real-world dataset and fit it three ways, in increasing
order of sophistication:

1. **Naive XGBoost** — the `xgboost` library with its out-of-the-box defaults, the way a
   first-time user would run it.
2. **XGBoost through AutoGluon, with bagging** — the same model family, but with AutoML-grade
   hyperparameters, early stopping, and 8-fold bagged ensembling.
3. **TabICLv2** — a tabular foundation model (TFM). A single default configuration, no
   hyperparameters to choose, one forward pass through a pretrained network.

The dataset — *polish_companies_bankruptcy* from the [TabArena](https://tabarena.ai)
benchmark — is a binary classification problem: predict whether a Polish company goes
bankrupt within the forecasting horizon, from 64 financial-ratio features (~4,900 rows,
all numeric, heavily imbalanced). It is small enough that every fit below runs in seconds,
and it cleanly separates the three tiers.

> **Runtime**: ~2–4 minutes total on a free Colab GPU runtime (TabICLv2 also runs on CPU,
> just slower). The metric is ROC AUC, where 0.5 is random and 1.0 is perfect.
""")

md("## Setup")

code("""
# On Colab, uncomment the next line (installs take ~2 minutes):
# %pip install -q xgboost autogluon.tabular tabicl openml

import time

import numpy as np
import openml
import pandas as pd
from sklearn.metrics import roc_auc_score
RANDOM_STATE = 0
results = {}  # name -> (test AUC, fit seconds)
""")

md("""
## The dataset

We fetch the dataset from OpenML and use the benchmark task's own first train/test split —
the same rows for all three models, so the comparison is apples-to-apples (and directly
comparable to published TabArena results).
""")

code("""
task = openml.tasks.get_task(363694)  # polish_companies_bankruptcy
X, y = task.get_X_and_y(dataset_format="dataframe")
y = (y == y.cat.categories[1]).astype(int) if hasattr(y, "cat") else y.astype(int)

train_idx, test_idx = task.get_train_test_split_indices(repeat=0, fold=0)
X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
print(f"train: {X_train.shape}, test: {X_test.shape}, positive rate: {y.mean():.3f}")
X_train.head(3)
""")

md("""
## Tier 1 — naive XGBoost

What a first-time user gets: `XGBClassifier()` with library defaults (100 trees, learning
rate 0.3, no early stopping), fit once on the training split.
""")

code("""
from xgboost import XGBClassifier

t0 = time.time()
xgb = XGBClassifier(random_state=RANDOM_STATE)
xgb.fit(X_train, y_train)
fit_s = time.time() - t0

auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
results["Naive XGBoost"] = (auc, fit_s)
print(f"naive XGBoost:  AUC = {auc:.4f}   (fit {fit_s:.1f}s)")
""")

md("""
## Tier 2 — the same model family, done well

Now the identical model family — gradient-boosted trees via XGBoost — but run through
AutoGluon: its curated default hyperparameters (10k trees with early stopping instead of a
fixed 100), plus **8-fold bagging**: eight copies fit on overlapping folds of the training
data whose predictions are averaged. This is the "properly engineered baseline" a
practitioner would build, without any hyperparameter search yet.
""")

code("""
from autogluon.tabular import TabularPredictor

train_df = X_train.copy()
train_df["__label__"] = y_train.values

t0 = time.time()
predictor = TabularPredictor(
    label="__label__", eval_metric="roc_auc", path="ag_xgb_bagged", verbosity=0
).fit(train_df, hyperparameters={"XGB": {}}, num_bag_folds=8)
fit_s = time.time() - t0

auc = roc_auc_score(y_test, predictor.predict_proba(X_test)[1])
results["XGBoost (AutoGluon, bagged)"] = (auc, fit_s)
print(f"AutoGluon XGBoost (bagged):  AUC = {auc:.4f}   (fit {fit_s:.1f}s)")
""")

md("""
## Tier 3 — a tabular foundation model

TabICLv2 is a transformer pretrained on millions of synthetic tabular tasks. There is no
tree-building and no hyperparameter search: `fit` stores the training data (and downloads
the ~100MB checkpoint on first use), and `predict_proba` runs **in-context learning** — the
network reads the training rows and the test rows together and outputs predictions in a
single forward pass.
""")

code("""
from tabicl import TabICLClassifier

t0 = time.time()
ticl = TabICLClassifier()  # one default config; uses GPU if available, else CPU
ticl.fit(X_train, y_train)
fit_s = time.time() - t0

auc = roc_auc_score(y_test, ticl.predict_proba(X_test)[:, 1])
results["TabICLv2 (default)"] = (auc, fit_s)
print(f"TabICLv2:  AUC = {auc:.4f}   (fit {fit_s:.1f}s)")
""")

md("## The comparison")

code("""
comparison = pd.DataFrame(
    [(name, auc, secs) for name, (auc, secs) in results.items()],
    columns=["model", "test AUC", "fit seconds"],
).set_index("model")
comparison["error vs naive"] = (1 - comparison["test AUC"]) / (1 - comparison.loc["Naive XGBoost", "test AUC"])
comparison.round(4)
""")

md("""
### What just happened

- **Naive → engineered**: AutoGluon's hyperparameters, early stopping, and bagging squeeze a
  real improvement out of the *same* model family — this is the value of AutoML engineering,
  and historically it is what separated Kaggle winners from everyone else.
- **Engineered → foundation model**: TabICLv2, with *zero* configuration, jumps well past
  both. On the full TabArena protocol (means over 9 splits; the traditional baselines get a
  201-config hyperparameter search with bagged ensembling), this dataset reads: naive
  XGBoost **0.946** → best traditional model **0.961** → TabICLv2 default **0.984** — the
  foundation model removes ~60% of the best traditional model's remaining error, at a
  fraction of the cost of the search.
- The `fit seconds` column is the punchline for practice: the TFM's "fit" is effectively
  free; its cost is at prediction time, and on datasets this size that cost is seconds.

**Next**: notebook 03 looks at *what* a TFM actually predicts — calibrated probabilities and
full predictive distributions — and why that matters.
""")

nb.cells = cells
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT}")
