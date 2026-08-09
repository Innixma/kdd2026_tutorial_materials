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
# Colab reads these when the notebook is opened and preselects a T4 GPU runtime.
nb.metadata["accelerator"] = "GPU"
nb.metadata["colab"] = {"gpuType": "T4", "provenance": []}
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# First contact: from a naive baseline to a tabular foundation model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Innixma/kdd2026_tutorial_materials/blob/main/notebooks/02_first_contact.ipynb)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/Innixma/kdd2026_tutorial_materials)
[![Tutorial Website](https://img.shields.io/badge/Tutorial-Website-0a7aca?logo=googlechrome&logoColor=white)](https://kdd26-automl-hands-on.github.io/)

**Taming Structured Data Foundation Models with AutoML — KDD 2026 hands-on tutorial**

In this notebook we take one small real-world dataset and fit it three ways, in increasing
order of sophistication:

1. **Naive [XGBoost](https://arxiv.org/abs/1603.02754)** — the `xgboost` library with its out-of-the-box defaults, the way a
   first-time user would run it.
2. **XGBoost through [AutoGluon](https://auto.gluon.ai/stable/index.html), with bagging** — the same model family, but with AutoML-grade
   hyperparameters, early stopping, and 8-fold bagged ensembling.
3. **[TabICLv2](https://arxiv.org/abs/2602.11139)** — a tabular foundation model (TFM). A single default configuration, no
   hyperparameters to choose, one forward pass through a pretrained network.
4. **[TabFM](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/)** — the largest current TFM (Google Research), run with a single ensemble member
   to keep the footprint small. Roughly 10-20x TabICLv2's compute at full ensemble size.

The dataset — *polish_companies_bankruptcy* from the [TabArena](https://tabarena.ai)
benchmark — is a binary classification problem: predict whether a Polish company goes
bankrupt within the forecasting horizon, from 64 financial-ratio features (~4,900 rows,
all numeric, heavily imbalanced). It is small enough that every fit below runs in seconds,
and it cleanly separates the three tiers.

> **Runtime**: ~2–4 minutes total on a free Colab GPU runtime (TabICLv2 also runs on CPU,
> just slower). The metric is ROC AUC, where 0.5 is random and 1.0 is perfect.
""")

md("""
## Tabular foundation models at a glance

Before fitting anything, here is the current TFM landscape — the models, the fit
constraints AutoGluon enforces for each (above the limits a model is skipped rather than
fitted), and, first thing to check in practice, the licenses. Adapted from the
[AutoGluon foundational-models tutorial](https://auto.gluon.ai/stable/tutorials/tabular/tabular-foundational-models.html).

| Model | Key | Released | Max rows | Max features | Max classes | Tasks | License |
|---|---|---|---:|---:|---:|---|---|
| TabPFN-1 | *not in AutoGluon* | 2022-07 | 1,000 | 100 | 10 | classification | Apache-2.0 |
| TabDPT | `TABDPT` | 2024-10 | 100,000 | 2,500 | 160 | all | Apache-2.0 |
| TabPFNv2 | `REALTABPFN-V2` | 2025-01 | 10,000 | 500 | 10 | all | Prior Labs License (commercial use permitted) |
| TabICL | `TABICL` (v1 checkpoint) | 2025-02 | 500,000 | 2,000 | — | all | BSD-3-Clause |
| Mitra | `MITRA` | 2025-07 | 10,000 | 500 | 10 | all | Apache-2.0 |
| RealTabPFN-2.5 | `REALTABPFN-V2.5` | 2025-11 | 100,000 | 2,000 | 10 | all | Commercial license required |
| TabICLv2 | `TABICL` (default) | 2026-02 | 500,000 | 2,000 | — | all | BSD-3-Clause |
| TabPFN-2.6 | `TABPFN-2.6` | 2026-03 | 100,000 | — | 10 | all | Commercial license required |
| TabPFN-3 | `TABPFN-3` | 2026-05 | 500,000 | — | 160 | all | Commercial license required |
| TabDPT-Turbo | `TABDPT-TURBO` | 2026-06 | 100,000 | — | 160 | all | Apache-2.0 |
| Nori | `NORI` | 2026-06 | 50,000 | — | — | regression only | Apache-2.0 |
| TabFM | *not in AutoGluon; `tabfm` library* | 2026-06 | — | — | — | all | Code Apache-2.0; weights noncommercial |
| EXAONE-Tabular | *not in AutoGluon; tabarena wrapper* | 2026-07 | — | — | — | all | Code BSD-3-Clause-LG; weights noncommercial (EXAONE 1.1-NC) |

Notes:

- **Licensing is the first thing to check.** The TabPFN models from 2.5 onwards and TabFM's
  weights are free for research and internal experimentation, but commercial use requires a
  license ([Prior Labs license FAQ](https://docs.priorlabs.ai/models#tabpfn-model-license)).
  TabPFNv2, TabICL/TabICLv2, TabDPT, Mitra, and Nori are all free for commercial use.
- Limits are per fit, measured on the training split; several models are far slower near
  their upper bound than well below it.
- AutoGluon's `extreme` and `noncommercial` presets (notebook 03) bundle these models into
  meta-learned portfolios instead of fitting them one at a time.
""")

md("## Setup")

code("""
# Installs everything the notebook needs (fast via uv; a no-op where the
# packages are already present).
import sys
!command -v uv >/dev/null || pip install -q uv
!uv pip install -q --python {sys.executable} xgboost autogluon.tabular tabicl openml jax "tabfm[pytorch] @ git+https://github.com/google-research/tabfm.git@01004165"

import time

import numpy as np
import openml
import pandas as pd
from sklearn.metrics import roc_auc_score
RANDOM_STATE = 0
results = {}  # name -> (test AUC, total fit + predict seconds)
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
proba = xgb.predict_proba(X_test)[:, 1]
total_s = time.time() - t0

auc = roc_auc_score(y_test, proba)
results["Naive XGBoost"] = (auc, total_s)
print(f"naive XGBoost:  AUC = {auc:.4f}   (fit + predict {total_s:.1f}s)")
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
proba = predictor.predict_proba(X_test)[1]
total_s = time.time() - t0

auc = roc_auc_score(y_test, proba)
results["XGBoost (AutoGluon, bagged)"] = (auc, total_s)
print(f"AutoGluon XGBoost (bagged):  AUC = {auc:.4f}   (fit + predict {total_s:.1f}s)")
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

ticl = TabICLClassifier()  # one default config; uses GPU if available, else CPU

# Untimed warm-up on a few rows of each class: downloads the checkpoint on first
# use and initializes the compute kernels, so the timing below measures the model,
# not the network.
warm_idx = y_train.groupby(y_train).head(8).index
ticl.fit(X_train.loc[warm_idx], y_train.loc[warm_idx])
ticl.predict_proba(X_test.head(8))

t0 = time.time()
ticl.fit(X_train, y_train)
proba = ticl.predict_proba(X_test)[:, 1]
total_s = time.time() - t0

auc = roc_auc_score(y_test, proba)
results["TabICLv2 (default)"] = (auc, total_s)
print(f"TabICLv2:  AUC = {auc:.4f}   (fit + predict {total_s:.1f}s)")
""")


md("""
## Tier 4 — the largest tabular foundation model

TabFM is the biggest TFM on the [TabArena](https://tabarena.ai) leaderboard, where it is the
strongest single model overall. We run it with `n_estimators=1` (a single ensemble member)
so it fits in a small GPU's memory; the full default ensemble is substantially stronger and
substantially more expensive.
""")

code("""
import torch
from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch

device = "cuda" if torch.cuda.is_available() else "cpu"
# Untimed: downloads the pretrained checkpoint from Hugging Face on first use.
network = tabfm_v1_0_0_pytorch.load(model_type="classification", device=device)

t0 = time.time()
tabfm = TabFMClassifier(model=network, n_estimators=1)
tabfm.fit(X_train, y_train)
proba = tabfm.predict_proba(X_test)[:, 1]
total_s = time.time() - t0

auc = roc_auc_score(y_test, proba)
results["TabFM (1 member)"] = (auc, total_s)
print(f"TabFM (1 member):  AUC = {auc:.4f}   (fit + predict {total_s:.1f}s)")
""")

md("## The comparison")

code("""
comparison = pd.DataFrame(
    [(name, auc, secs) for name, (auc, secs) in results.items()],
    columns=["model", "test AUC", "total seconds (fit + predict)"],
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
- The time column is end-to-end (fit + predict, checkpoint download excluded): a TFM's
  "fit" is effectively free and its cost sits at prediction time, yet even the total is
  on par with a single naive XGBoost fit on datasets this size.

TabFM's single-member run above scores **0.9954** — the strongest single model on the
board, taking another large bite out of TabICLv2's remaining error at a fraction of a
second of compute.

**Next**: notebook 03 puts these models to work automatically through AutoGluon. Preview of
where that lands on this same split: the default preset (optimized for `roc_auc`) reaches
**0.9687**, and the `extreme` preset — which composes TabICLv2, TabDPT-Turbo, and a TabPrep
LightGBM into one ensemble with a single `fit()` call — reaches **0.9955**, matching the
hand-run TabFM tier above without you ever choosing a model.
""")

nb.cells = cells
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT}")
