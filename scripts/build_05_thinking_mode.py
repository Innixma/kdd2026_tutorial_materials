"""Build notebooks/05_thinking_mode.ipynb.

TabPFN-3 thinking mode on Amazon_employee_access: first the exhibit that TFMs struggle on
this high-cardinality-categorical dataset while CatBoost dominates, then thinking mode via
the TabPFN API (treated strictly as a capability; no discussion of how it works internally).
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "05_thinking_mode.ipynb"

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
# Thinking mode: cracking the dataset TFMs couldn't

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Innixma/kdd2026_tutorial_materials/blob/main/notebooks/05_thinking_mode.ipynb)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/Innixma/kdd2026_tutorial_materials)
[![Tutorial Website](https://img.shields.io/badge/Tutorial-Website-0a7aca?logo=googlechrome&logoColor=white)](https://kdd26-automl-hands-on.github.io/)

**Taming Structured Data Foundation Models with AutoML — KDD 2026 hands-on tutorial**

*Amazon employee access* is a famous holdout of the pre-TFM era: ~33k rows whose nine
features are all high-cardinality categorical codes (resource ids, manager ids, role
codes). It is CatBoost's home turf — and on the [TabArena](https://tabarena.ai)
leaderboard it is one of the very few datasets where **every** tabular foundation model
loses to well-tuned gradient boosting.

This notebook first reproduces that struggle live (CatBoost vs TabICLv2 and TabPFN-3,
with EXAONE's and TabFM's benchmark scores quoted alongside), then runs
[TabPFN-3](https://priorlabs.ai/technical-reports/tabpfn-3)'s
[thinking mode](https://docs.priorlabs.ai/capabilities/thinking-mode) on the same split.

> **Runtime**: ~10-15 minutes on a Colab T4; the thinking-mode fit runs on the TabPFN API
> (about 5 minutes at high effort) rather than the local GPU.
""")

md("""
## Setup

The install is skipped outside Colab so it never overwrites a locally managed environment.
""")

code("""
import importlib.util

IN_COLAB = importlib.util.find_spec("google.colab") is not None
if IN_COLAB:
    !command -v uv >/dev/null || pip install -q uv
    !uv pip install -q --python {__import__('sys').executable} "autogluon.tabular[tabarena]" openml tabpfn-client

    # The install may replace Colab's preinstalled numpy; the copy already loaded in this
    # kernel then no longer matches the files on disk and imports break. When that happens,
    # restart the runtime once (continue from the next cell after it reconnects).
    import importlib.metadata
    import numpy
    if importlib.metadata.version("numpy") != numpy.__version__:
        print("numpy changed -- restarting the Colab runtime; re-run FROM THE NEXT CELL when it reconnects.")
        import os
        os.kill(os.getpid(), 9)
""")

md("## The dataset\n\nSplit 0 of the official benchmark task, as everywhere in this tutorial.")

code("""
import getpass
import os

import openml
from autogluon.tabular import TabularDataset, TabularPredictor

# --- Prior Labs access token (unlocks the gated TabPFN checkpoints and the API) ---
# 1. Sign up / log in at https://ux.priorlabs.ai
# 2. Accept the license at https://ux.priorlabs.ai/account/licenses
# 3. Copy your access token from https://ux.priorlabs.ai/account
# Tip: save it as a Colab secret named TABPFN_TOKEN to skip the prompt next time.
tabpfn_token = os.environ.get("TABPFN_TOKEN")
if not tabpfn_token:
    try:
        from google.colab import userdata
        tabpfn_token = userdata.get("TABPFN_TOKEN")
    except Exception:
        pass
while not tabpfn_token:
    tabpfn_token = getpass.getpass("Paste your TABPFN_TOKEN and press Enter: ").strip()
os.environ["TABPFN_TOKEN"] = tabpfn_token

task = openml.tasks.get_task(363613)  # Amazon_employee_access
X, y = task.get_X_and_y(dataset_format="dataframe")
train_idx, test_idx = task.get_train_test_split_indices(repeat=0, fold=0)

label = y.name
full_data = X.copy()
full_data[label] = y
train_data = TabularDataset(full_data.iloc[train_idx].reset_index(drop=True))
test_data = TabularDataset(full_data.iloc[test_idx].reset_index(drop=True))
print(f"train: {train_data.shape}, test: {test_data.shape}")
train_data.head(3)
""")

md("""
## Round 1 — the TFM struggle, live

CatBoost against the foundation models, single fits on identical data. Two more TFMs
(EXAONE-Tabular and TabFM) are commented out — extra install and a very large checkpoint
download respectively; their benchmark scores appear in the closing comparison.
""")

code("""
# tabarena's benchmark wrappers drop into the same dict as classes. Install them with:
#   pip install "tabarena[exaone_tabular] @ git+https://github.com/autogluon/tabarena.git#subdirectory=packages/tabarena"
# from tabarena.models.exaone_tabular.model import EXAONETabularModel
# from tabarena.models.tabfm.model import TabFMModel  # ~13GB checkpoint download

predictor = TabularPredictor(label=label, eval_metric="roc_auc").fit(
    train_data,
    hyperparameters={
        "CAT": {},
        "TABICL": {},           # TabICLv2
        "TABPFN-3": {},         # noncommercial weights; via the access token above
        # EXAONETabularModel: {},
        # TabFMModel: {"n_estimators": 1},
    },
    num_gpus=1,
)
predictor.leaderboard(test_data)
""")

md("""
The pattern the benchmark shows across nine splits holds in one glance: **CatBoost on top,
every foundation model behind it**. Wide, high-cardinality categorical spaces are the
corner of tabular learning where gradient boosting still rules and in-context learning has
struggled.

## Round 2 — thinking mode

Same split, one change: TabPFN-3 through the API with
[thinking mode](https://docs.priorlabs.ai/capabilities/thinking-mode) enabled. It spends
substantially more compute at fit time (about five minutes here), steered toward the metric
you declare.
""")

code("""
import time

import tabpfn_client
from sklearn.metrics import roc_auc_score
from tabpfn_client import TabPFNClassifier

tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
y_train_bin = (y_train == y_train.cat.categories[1]).astype(int) if hasattr(y_train, "cat") else y_train
y_test_bin = (y_test == y_test.cat.categories[1]).astype(int) if hasattr(y_test, "cat") else y_test

clf = TabPFNClassifier(
    thinking_mode=True,
    thinking_effort="high",
    thinking_metric="roc_auc",
)
t0 = time.time()
clf.fit(X_train, y_train_bin)
fit_s = time.time() - t0
proba = clf.predict_proba(X_test)[:, 1]
print(f"TabPFN-3 (thinking): AUC = {roc_auc_score(y_test_bin, proba):.4f}   (fit {fit_s:.0f}s)")
""")

md("""
## What just happened

One model, one switch, and the unclimbable wall is climbed: thinking mode does not just
close the TFM-vs-CatBoost gap on this dataset — it comes out ahead. On the full benchmark
protocol (mean over nine splits), the picture is: TabFM **0.858**, EXAONE-Tabular
**0.872**, CatBoost (tuned + ensembled) **0.882** — and thinking **0.885**, the only
single model above the CatBoost family on this dataset, trailing just the full AutoGluon
systems on the TabArena leaderboard.

The takeaways for practice:

- The metric argument matters here as everywhere (`thinking_metric="roc_auc"` — the lesson
  from notebook 02).
- Thinking mode trades fit-time compute for accuracy; reach for it when a dataset sits in
  a known TFM weak spot or when the last points of a metric are valuable. Details and
  options (`thinking_effort`, `thinking_timeout_s`) are in the
  [docs](https://docs.priorlabs.ai/capabilities/thinking-mode); the model itself is
  described in the [TabPFN-3 technical report](https://priorlabs.ai/technical-reports/tabpfn-3).
- It runs through the TabPFN API (`tabpfn-client`), not the local `tabpfn` package.

**Next**: [notebook 06](https://colab.research.google.com/github/Innixma/kdd2026_tutorial_materials/blob/main/notebooks/06_noniid_validation.ipynb) tackles data where rows aren't IID — honest validation with grouped and temporal splits.
""")

nb.cells = cells
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT}")
