"""Build notebooks/06_noniid_validation.ipynb.

Non-IID validation in AutoGluon: a naive random split vs `validation_structure` on the
mice-protein dataset (72 mice, 15 repeated measurements each), where the naive split's
validation score is wrong by more than an order of magnitude and inverts model selection.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "06_noniid_validation.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
# CPU-only notebook: no accelerator metadata, so Colab opens a plain CPU runtime.
nb.metadata["colab"] = {"provenance": []}
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# When rows aren't IID: honest validation with `validation_structure`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Innixma/kdd2026_tutorial_materials/blob/main/notebooks/06_noniid_validation.ipynb)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/Innixma/kdd2026_tutorial_materials)
[![Tutorial Website](https://img.shields.io/badge/Tutorial-Website-0a7aca?logo=googlechrome&logoColor=white)](https://kdd26-automl-hands-on.github.io/)

**Taming Structured Data Foundation Models with AutoML — KDD 2026 hands-on tutorial**

Every validation split so far assumed rows are independent. Much real tabular data is not:
repeated measurements per patient, transactions per customer, readings over time. Split
such data randomly and the "held-out" rows share their entity with training rows — the
model recognizes *the mouse*, not the biology, and the validation score becomes fiction.

The dataset: **mice protein expression** — 72 mice, 15 measurements each (1,080 rows),
77 protein levels, 8 classes. New data in deployment means *new mice*, so honest
evaluation must hold out whole mice. We build one honest, mouse-disjoint test set, then
fit AutoGluon twice on the identical training data:

1. **Naive** — the default random holdout, as if rows were IID.
2. **Grouped** — `validation_structure={"group_on": "mouse"}`, which makes every internal
   split mouse-disjoint.

> **Runtime**: ~2 minutes, CPU only.
""")

md("## Setup")

code("""
# Installs everything the notebook needs (fast via uv; a no-op where already present).
import sys
!command -v uv >/dev/null || pip install -q uv
!uv pip install -q --python {sys.executable} autogluon.tabular openml

import numpy as np
import openml
from autogluon.tabular import TabularPredictor
""")

md("""
## The data, and an honest test set

OpenML's copy carries the measurement id (`MouseID`, e.g. `309_1` … `309_15`); the part
before the underscore identifies the mouse. We hold out 30% of the *mice* — not 30% of the
rows — as the test set both runs share.
""")

code("""
ds = openml.datasets.get_dataset(40966)  # MiceProtein
df, *_ = ds.get_data(include_row_id=True)
df["mouse"] = df["MouseID"].astype(str).str.split("_").str[0]
df = df.drop(columns=["MouseID"])

rng = np.random.default_rng(0)
mice = df["mouse"].unique()
test_mice = set(rng.choice(mice, size=int(0.3 * len(mice)), replace=False))
train = df[~df["mouse"].isin(test_mice)].reset_index(drop=True)
test = df[df["mouse"].isin(test_mice)].reset_index(drop=True)
print(f"{df['mouse'].nunique()} mice, {len(train)} train rows, {len(test)} test rows ({len(test_mice)} held-out mice)")
""")

md("""
## Run 1 — the naive split

We drop the `mouse` column (a naive user wouldn't think of it as a feature or a split key)
and let AutoGluon use its default random holdout.
""")

code("""
naive = TabularPredictor(label="class", eval_metric="log_loss", path="noniid_naive", verbosity=0).fit(
    train.drop(columns=["mouse"]),
    hyperparameters={"GBM": {}, "RF": {}, "XGB": {}},
)
naive.leaderboard(test.drop(columns=["mouse"]))
""")

md("""
Read that leaderboard carefully — it is a double disaster:

- **The scores are fiction.** Validation says log-loss ≈ 0.07; the mouse-disjoint test says
  ≈ 1.9. The estimate is off by more than **25×**, because every "held-out" row had
  siblings from the same mouse in training.
- **Model selection is inverted.** Validation ranks the boosted trees above RandomForest;
  on genuinely new mice, RandomForest is by far the best and the boosted trees are the
  worst. The leaked split doesn't just misestimate — it picks the wrong model.

## Run 2 — declare the structure

One argument fixes both problems: `validation_structure` tells AutoGluon the rows are
grouped by `mouse`, and every internal split — holdout or bagged folds — becomes
group-disjoint. (For temporal data the analogous key is `time_on`; for both at once,
`group_time_on`.)
""")

code("""
grouped = TabularPredictor(label="class", eval_metric="log_loss", path="noniid_grouped", verbosity=0).fit(
    train,
    hyperparameters={"GBM": {}, "RF": {}, "XGB": {}},
    validation_structure={"group_on": "mouse"},
)
grouped.leaderboard(test)
""")

md("""
Now validation and test agree: the estimate is honest (within a factor ~1.2 rather than
25×), RandomForest correctly rises in the ranking, and the final selected model scores
substantially better on the honest test set than the naive run's choice — the naive run
didn't just *report* the wrong number, it *shipped* a worse model.

## Takeaways

- If your rows share entities (patients, customers, devices, stores) or arrive over time,
  IID validation silently breaks — and TFMs' in-context memorization makes them at least as
  exposed to this as trees.
- `validation_structure` is declarative: `group_on`, `time_on`, `group_time_on`,
  `stratify_on` — one dict, and holdouts, bagged folds, and ensembling all honor it.
- Benchmarks need this too: [TabArena](https://tabarena.ai)'s BeyondArena tasks carry
  grouped and temporal splits as first-class citizens, and leaderboards there are computed
  on structure-respecting splits — 39 of its datasets declare a group or time column.

**Next**: with fitting, priors, the zoo, and honest validation covered, you have the full
toolkit — the remaining sessions put it to work on the benchmark itself.
""")

nb.cells = cells
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT}")
