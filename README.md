# Taming Structured Data Foundation Models with AutoML: A Hands-On Guide

Materials for the [KDD 2026](https://kdd.org/kdd2026/) hands-on tutorial — see the
[tutorial website](https://kdd26-automl-hands-on.github.io/) for the schedule, speakers,
and slides.

## Notebooks

| Notebook | What it covers | Open in Colab |
|---|---|---|
| [`01_first_contact.ipynb`](notebooks/01_first_contact.ipynb) | Four tiers on one real dataset: naive XGBoost, AutoGluon-bagged XGBoost, TabICLv2, and TabFM — opening with the TFM at-a-glance table | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Innixma/kdd2026_tutorial_materials/blob/main/notebooks/01_first_contact.ipynb) |
| [`02_autogluon_essentials.ipynb`](notebooks/02_autogluon_essentials.ipynb) | AutoGluon essentials: fit, predict, evaluate, inspect, and the `extreme` preset with TFMs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Innixma/kdd2026_tutorial_materials/blob/main/notebooks/02_autogluon_essentials.ipynb) |
| [`03_tfm_zoo.ipynb`](notebooks/03_tfm_zoo.ipynb) | AutoGluon as an optimized model zoo for TFMs: many foundation models, one predictor | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Innixma/kdd2026_tutorial_materials/blob/main/notebooks/03_tfm_zoo.ipynb) |
| [`04_inside_the_prior.ipynb`](notebooks/04_inside_the_prior.ipynb) | The synthetic prior TFMs are pretrained on, generated live via nanotabicl | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Innixma/kdd2026_tutorial_materials/blob/main/notebooks/04_inside_the_prior.ipynb) |
| [`05_pretrain_your_own.ipynb`](notebooks/05_pretrain_your_own.ipynb) | Pretrain a working TabPFN-style model in about a minute with nanoTabPFN | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Innixma/kdd2026_tutorial_materials/blob/main/notebooks/05_pretrain_your_own.ipynb) |
| [`06_thinking_mode.ipynb`](notebooks/06_thinking_mode.ipynb) | TabPFN-3 thinking mode conquers the high-cardinality-categorical dataset where TFMs lose to CatBoost | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Innixma/kdd2026_tutorial_materials/blob/main/notebooks/06_thinking_mode.ipynb) |
| [`07_interpretability.ipynb`](notebooks/07_interpretability.ipynb) | Explaining TabPFN-3 predictions with shapiq Shapley values, plus the KV-cache `fit_mode` that makes repeated inference fast | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Innixma/kdd2026_tutorial_materials/blob/main/notebooks/07_interpretability.ipynb) |
| [`08_noniid_validation.ipynb`](notebooks/08_noniid_validation.ipynb) | Non-IID data: a naive split vs `validation_structure` on grouped data, where the naive estimate is off 25x | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Innixma/kdd2026_tutorial_materials/blob/main/notebooks/08_noniid_validation.ipynb) |
| [`09_timeseries_forecasting.ipynb`](notebooks/09_timeseries_forecasting.ipynb) | Time series forecasting with AutoGluon | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Innixma/kdd2026_tutorial_materials/blob/main/notebooks/09_timeseries_forecasting.ipynb) |

The notebooks open on a free Colab T4 GPU runtime by default.

### Additional notebooks

External notebooks worth working through alongside the tutorial:

- [TabPFN local demo](https://github.com/PriorLabs/TabPFN/blob/main/examples/notebooks/TabPFN_Demo_Local.ipynb) — Prior Labs' official demo: local-GPU vs API backends, classification and regression.
- [AutoGluon: tabular foundational models](https://auto.gluon.ai/stable/tutorials/tabular/tabular-foundational-models.html) — the official AutoGluon tutorial for the TFMs used in this repo.
- [TabICL tutorial gallery](https://github.com/soda-inria/tabicl/tree/main/tutorials) — getting started, probabilistic classification, interpretability (SHAP), fine-tuning, unsupervised use, and forecasting.
- [nanotabicl](https://github.com/soda-inria/nanotabicl) — the minimal TabICL prior and model, the source behind notebook 04.
- [nanoTabPFN experiment notebook](https://github.com/automl/nanoTabPFN/blob/main/experiment.ipynb) — pretrain a working TabPFN-style model in about a minute on one GPU, in under 500 lines ([paper](https://arxiv.org/abs/2511.03634)).
- [TFM-Playground](https://github.com/automl/TFM-Playground) — a playground for experimenting with tabular foundation models.
- [AutoGluon: deployment optimization](https://auto.gluon.ai/stable/tutorials/tabular/advanced/tabular-deployment.html) — shrinking and shipping a fitted predictor for production inference.

## Layout

- `notebooks/` — the hands-on notebooks, one per tutorial module, numbered in session order.
- `slides/` — lecture slides for the non-hands-on segments.
- `scripts/` — setup and utility scripts (environment checks, model-weight prefetching, dataset preparation).
- `data/` — small datasets used in the exercises. Anything above a few MB should be downloaded by a script rather than committed.

## Environment

The notebooks target `autogluon.tabular[tabarena]` with a GPU (a Colab T4 is sufficient for the attendee exercises). Pinned dependencies live in `requirements.txt`.

```bash
pip install -r requirements.txt
python scripts/check_environment.py
```

Run the environment check (and the model-weight prefetch it performs) before the session: the tabular foundation model checkpoints are several hundred MB and should never be downloaded over conference Wi-Fi.
