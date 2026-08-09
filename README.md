# Taming Structured Data Foundation Models with AutoML: A Hands-On Guide

Materials for the KDD 2026 hands-on tutorial.

## Notebooks

| Notebook | What it covers | Open in Colab |
|---|---|---|
| [`02_first_contact.ipynb`](notebooks/02_first_contact.ipynb) | Four tiers on one real dataset: naive XGBoost, AutoGluon-bagged XGBoost, TabICLv2, and TabFM | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Innixma/kdd2026_tutorial_materials/blob/main/notebooks/02_first_contact.ipynb) |
| [`03_autogluon_essentials.ipynb`](notebooks/03_autogluon_essentials.ipynb) | AutoGluon essentials: fit, predict, evaluate, inspect, and the `extreme` preset with TFMs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Innixma/kdd2026_tutorial_materials/blob/main/notebooks/03_autogluon_essentials.ipynb) |
| [`04_inside_the_prior.ipynb`](notebooks/04_inside_the_prior.ipynb) | The synthetic prior TFMs are pretrained on, generated live via nanotabicl | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Innixma/kdd2026_tutorial_materials/blob/main/notebooks/04_inside_the_prior.ipynb) |
| [`05_tfm_zoo.ipynb`](notebooks/05_tfm_zoo.ipynb) | AutoGluon as an optimized model zoo for TFMs: many foundation models, one predictor, incl. tabarena wrappers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Innixma/kdd2026_tutorial_materials/blob/main/notebooks/05_tfm_zoo.ipynb) |
| [`kdd-tutorial-timeseries.ipynb`](notebooks/kdd-tutorial-timeseries.ipynb) | Time series forecasting with AutoGluon | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Innixma/kdd2026_tutorial_materials/blob/main/notebooks/kdd-tutorial-timeseries.ipynb) |

The notebooks open on a free Colab T4 GPU runtime by default.

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
