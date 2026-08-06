# Taming Structured Data Foundation Models with AutoML: A Hands-On Guide

Materials for the KDD 2026 hands-on tutorial.

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
