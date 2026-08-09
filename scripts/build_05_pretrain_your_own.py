"""Build notebooks/05_pretrain_your_own.ipynb.

Pretrain a working TabPFN-style model in about a minute with nanoTabPFN: fetch the ~400
lines of model + training code, download the prior data dump, train live, and evaluate the
freshly pretrained model on real data against a classical baseline.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "05_pretrain_your_own.ipynb"

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
# Pretrain your own TFM in a minute

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Innixma/kdd2026_tutorial_materials/blob/main/notebooks/05_pretrain_your_own.ipynb)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/Innixma/kdd2026_tutorial_materials)
[![Tutorial Website](https://img.shields.io/badge/Tutorial-Website-0a7aca?logo=googlechrome&logoColor=white)](https://kdd26-automl-hands-on.github.io/)

**Taming Structured Data Foundation Models with AutoML — KDD 2026 hands-on tutorial**

Notebook 04 generated the synthetic datasets TFMs are pretrained on. This notebook closes
the loop: we **pretrain an actual TabPFN-style transformer, live**, on exactly that kind of
data — and then watch it classify real tumors it has never seen anything like.

The vehicle is [nanoTabPFN](https://github.com/automl/nanoTabPFN)
([paper](https://arxiv.org/abs/2511.03634)): the TabPFN v2 architecture and training loop in
under 400 lines. Restricted to small data, one minute of pretraining on a single GPU is
enough to be competitive with classical baselines — roughly **160,000× less pretraining
compute** than the real TabPFNv2, which is the same recipe scaled up.

> **Runtime**: ~5 minutes on a Colab T4, most of it downloading the 1GB prior-data dump;
> the pretraining itself is about a minute.
""")

md("""
## Setup

We fetch nanoTabPFN's two source files straight from the repository and the pregenerated
prior dump (300,000 synthetic datasets of 150 rows × 5 features) from figshare.
""")

code("""
import sys
!command -v uv >/dev/null || pip install -q uv
!uv pip install -q --python {sys.executable} schedulefree h5py scikit-learn

import urllib.request

for f in ["model.py", "train.py"]:
    urllib.request.urlretrieve(f"https://raw.githubusercontent.com/automl/nanoTabPFN/main/{f}", f)

import os
if not os.path.exists("300k_150x5_2.h5"):
    !curl -sL -o 300k_150x5_2.h5 "https://ndownloader.figshare.com/files/58932628?private_link=63fc1ada93e42e388e63"
print("prior dump:", round(os.path.getsize("300k_150x5_2.h5") / 1e9, 2), "GB")
""")

md("""
## Pretrain

The model is a 3-layer, 96-dimensional transformer with TabPFN's two-dimensional attention
(across rows and across features). Every training batch is a fresh set of synthetic
prediction tasks from the dump: the model sees labeled rows as context and learns to
predict the held-out rows' labels — *learning to learn*, never seeing the same dataset
twice.
""")

code("""
import torch
from model import NanoTabPFNClassifier, NanoTabPFNModel
from train import PriorDumpDataLoader, eval, get_default_device, set_randomness_seed, train

set_randomness_seed(0)
device = get_default_device()
print("device:", device)

model = NanoTabPFNModel(
    embedding_size=96,
    num_attention_heads=4,
    mlp_hidden_size=192,
    num_layers=3,
    num_outputs=2,
)
print(sum(p.numel() for p in model.parameters()), "parameters")

prior = PriorDumpDataLoader("300k_150x5_2.h5", num_steps=2500, batch_size=32, device=device)
model, history = train(model, prior, lr=4e-3, steps_per_eval=25)
""")

md("""
## Did it learn to learn?

The pretrained network is wrapped in a scikit-learn interface and applied — with **no
further training** — to the breast-cancer dataset: 30 real medical features it has never
seen, only ever having lived on 5-feature synthetic tables. Logistic regression provides
the classical reference point.
""")

code("""
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    *load_breast_cancer(return_X_y=True), test_size=0.5, random_state=0
)

nano = NanoTabPFNClassifier(model, device)
nano.fit(X_train, y_train)
nano_auc = roc_auc_score(y_test, nano.predict_proba(X_test)[:, 1])

logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_train, y_train)
logreg_auc = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])

print(f"nanoTabPFN (1 minute of pretraining): AUC = {nano_auc:.4f}")
print(f"logistic regression (fit on this data): AUC = {logreg_auc:.4f}")
""")

md("""
## What just happened

A transformer that has **never seen a real dataset** — and never gets gradient updates on
this one — classifies tumors at 0.96 AUC via a single forward pass. Logistic regression,
fit directly to the data, still wins here (breast cancer is a friendly, near-linear
problem), but that is not the point: the point is that *one minute* of pretraining on
synthetic tables produces genuine learning-to-learn. That is the entire TFM thesis in
miniature, reproduced on your GPU:

- The **prior** (notebook 04) supplies endless synthetic prediction tasks.
- Pretraining across them teaches the architecture *how to learn from a table*, rather than
  any particular table.
- Scale the same recipe up — bigger model, richer prior, weeks instead of a minute — and you
  get TabPFN, TabICLv2, and TabFM, the models from notebooks 01-03.

Explore further in [nanoTabPFN](https://github.com/automl/nanoTabPFN): the
[paper](https://arxiv.org/abs/2511.03634) benchmarks this small setting properly, and
`experiment.ipynb` reproduces those results.

**Next**: [notebook 06](https://colab.research.google.com/github/Innixma/kdd2026_tutorial_materials/blob/main/notebooks/06_thinking_mode.ipynb) pushes a full-scale model further — thinking mode on the dataset TFMs couldn't crack.
""")

nb.cells = cells
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT}")
