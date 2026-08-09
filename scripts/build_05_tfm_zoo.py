"""Build notebooks/05_tfm_zoo.ipynb.

Fitting and predicting with many tabular foundation models through one AutoGluon
TabularPredictor: the built-in TFM model keys, plus tabarena's wrappers (EXAONE-Tabular,
TabFM) dropped into the same hyperparameters dict. The install cell is Colab-guarded so a
local run never overwrites a development tabarena/autogluon install.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "05_tfm_zoo.ipynb"

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
# The TFM zoo: one predictor, every foundation model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Innixma/kdd2026_tutorial_materials/blob/main/notebooks/05_tfm_zoo.ipynb)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/Innixma/kdd2026_tutorial_materials)
[![Tutorial Website](https://img.shields.io/badge/Tutorial-Website-0a7aca?logo=googlechrome&logoColor=white)](https://kdd26-automl-hands-on.github.io/)

**Taming Structured Data Foundation Models with AutoML — KDD 2026 hands-on tutorial**

Every tabular foundation model ships with its own library, its own API quirks, its own
preprocessing expectations, VRAM appetite, and problem-type constraints. Running them one by
one — as we did by hand in notebook 02 — does not scale past a demo.

This notebook shows the alternative: **AutoGluon as an optimized model zoo for TFMs**. One
`TabularPredictor`, one `hyperparameters` dict listing the models you want, and AutoGluon
handles the rest — per-model preprocessing, GPU/memory management, constraint checks
(models that don't support the problem type are skipped, not crashed), fit scheduling, a
shared validation protocol, and an ensemble over everything at the end. The zoo is also
*extensible*: [tabarena](https://github.com/autogluon/tabarena)'s benchmark wrappers are
AutoGluon model classes, so models that haven't shipped in an AutoGluon release yet (like
EXAONE-Tabular or TabFM) drop into the same dict.

> **Runtime**: ~5-10 minutes on a Colab T4, most of it checkpoint downloads on first use.
""")

md("""
## Setup

The install is skipped outside Colab so it never overwrites a locally managed environment.
`autogluon.tabular[tabarena]` brings the built-in TFMs' dependencies; `tabarena` itself adds
the benchmark's model wrappers (with the `exaone_tabular` extra for EXAONE's library).
""")

code("""
import importlib.util

IN_COLAB = importlib.util.find_spec("google.colab") is not None
if IN_COLAB:
    !command -v uv >/dev/null || pip install -q uv
    !uv pip install -q --python {__import__('sys').executable} "autogluon.tabular[tabarena]" openml \\
        "tabarena[exaone_tabular] @ git+https://github.com/autogluon/tabarena.git"
""")

md("## The dataset\n\nSame task as notebooks 02 and 03 — *polish_companies_bankruptcy*, official benchmark split — so every number is comparable across the tutorial.")

code("""
import openml
from autogluon.tabular import TabularDataset, TabularPredictor

task = openml.tasks.get_task(363694)  # polish_companies_bankruptcy
X, y = task.get_X_and_y(dataset_format="dataframe")
train_idx, test_idx = task.get_train_test_split_indices(repeat=0, fold=0)

full_data = X.copy()
full_data[y.name] = y
train_data = TabularDataset(full_data.iloc[train_idx].reset_index(drop=True))
test_data = TabularDataset(full_data.iloc[test_idx].reset_index(drop=True))
print(f"train: {train_data.shape}, test: {test_data.shape}")
""")

md("""
## One dict, many foundation models

AutoGluon's registry addresses each built-in model by a string key. The zoo below runs
several TFMs side by side — and includes one deliberate misfit: Nori is regression-only, so
on this binary task AutoGluon quietly drops it from the fit plan instead of crashing,
exactly the constraint handling you would otherwise write yourself.

Two groups are present but commented out, each for a practical reason you will hit in the
wild: the TabPFN family's checkpoints are gated on Hugging Face (run
`huggingface-cli login` with a token that has accepted the model terms, then uncomment),
and TabFM's checkpoint is a very large download.

Models that live outside the AutoGluon release drop into the same dict as classes:
tabarena's `EXAONETabularModel` wrapper works like any built-in. And the zoo is not only
TFMs — the commented block at the bottom lists the classical toolkit (boosted trees,
forests, linear models, neural nets) that shares the same interface. The full roster of
built-in models and their keys is in the
[AutoGluon model docs](https://auto.gluon.ai/stable/api/autogluon.tabular.models.html).
""")

code("""
from tabarena.models.exaone_tabular.model import EXAONETabularModel

# from tabarena.models.tabfm.model import TabFMModel  # ~13GB checkpoint download; uncomment if you have the time/disk

hyperparameters = {
    # Built into AutoGluon (string keys):
    "TABICL": {},           # TabICLv2
    "TABDPT": {},
    "TABDPT-TURBO": {},
    "NORI": {},             # regression-only: dropped from the fit plan on this binary task
    # TabPFN family: gated Hugging Face checkpoints -- authenticate first (`huggingface-cli login`),
    # accept the model terms on the hub, then uncomment. TabPFN-3 weights are also noncommercial.
    # "TABPFN-2.6": {},
    # "REALTABPFN-V2.5": {},
    # "TABPFN-3": {},
    # From tabarena's model wrappers (classes):
    EXAONETabularModel: {},
    # TabFMModel: {"n_estimators": 1},
    # The same zoo also holds the entire classical toolkit -- uncomment any of these to add
    # them to the exact same fit/leaderboard/ensemble flow:
    # "GBM": {},        # LightGBM
    # "XGB": {},        # XGBoost
    # "CAT": {},        # CatBoost
    # "GBM_PREP": {},   # TabPrep LightGBM (feature-engineering pipeline + LightGBM)
    # "EBM": {},        # Explainable Boosting Machine
    # "RF": {},         # RandomForest
    # "XT": {},         # ExtraTrees
    # "KNN": {},
    # "LR": {},         # Linear model
    # "REALMLP": {},
    # "TABM": {},
    # "NN_TORCH": {},   # AutoGluon's torch MLP
    # "FASTAI": {},     # fastai tabular NN
}

predictor = TabularPredictor(label="company_bankrupt", eval_metric="roc_auc").fit(
    train_data,
    hyperparameters=hyperparameters,
)
""")

md("## The leaderboard\n\nEvery model was fit under the same validation protocol, and `WeightedEnsemble_L2` blends them — the zoo's models are strongest *together*.")

code("""
predictor.leaderboard(test_data)
""")

md("""
Predictions come from the ensemble by default, or from any single zoo member by name:
""")

code("""
proba_ensemble = predictor.predict_proba(test_data)
proba_tabicl = predictor.predict_proba(test_data, model="TabICL")
proba_ensemble.head()
""")

md("""
## Why this is the practical answer

- **One interface for every generation.** The dict above spans 2025-era TabDPT to 2026-era
  TabICLv2 and EXAONE (and the TabPFN family once authenticated) — same fit call, same
  leaderboard, same predict API.
- **The engineering is amortized.** VRAM-aware scheduling, per-model preprocessing,
  problem-type constraints, seed handling, and ensembling are implemented once in the zoo,
  not once per model — this is what "taming" TFMs means in practice.
- **New models are one class away.** Anything wrapped as an AutoGluon model (every method
  benchmarked on [TabArena](https://tabarena.ai) already is) joins the zoo without waiting
  for a release.
- The presets from notebook 03 (`extreme`, `noncommercial`) are exactly this zoo with a
  meta-learned shopping list: a portfolio of configs chosen on TabArena, plus bagging.

**Next**: notebook 04 looks at where all these models come from — the synthetic prior.
""")

nb.cells = cells
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT}")
