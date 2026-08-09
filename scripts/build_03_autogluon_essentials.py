"""Build notebooks/03_autogluon_essentials.ipynb.

An adaptation of the official AutoGluon "Tabular Essentials" tutorial for the KDD 2026
tutorial: same teaching flow (fit -> predict -> evaluate -> inspect -> maximize), but on
polish_companies_bankruptcy (the dataset from notebook 02, official benchmark split), with
the regression section removed and `presets="best"` replaced by `presets="extreme"`.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "03_autogluon_essentials.ipynb"

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
# AutoGluon Tabular — Essential Functionality

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Innixma/kdd2026_tutorial_materials/blob/main/notebooks/03_autogluon_essentials.ipynb)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/Innixma/kdd2026_tutorial_materials)
[![Tutorial Website](https://img.shields.io/badge/Tutorial-Website-0a7aca?logo=googlechrome&logoColor=white)](https://kdd26-automl-hands-on.github.io/)

**Taming Structured Data Foundation Models with AutoML — KDD 2026 hands-on tutorial**

*Adapted from the official [AutoGluon Tabular Essentials tutorial](https://auto.gluon.ai/stable/tutorials/tabular/tabular-essentials.html), on the dataset from notebook 02.*

Notebook 02 compared individual models by hand. This notebook shows the AutoML way: how
AutoGluon's `TabularPredictor` produces a highly accurate model in 3 lines of code — and how
its `extreme` preset puts the tabular foundation models you just met to work automatically.

We keep working on *polish_companies_bankruptcy*: predict whether a Polish company goes
bankrupt, from 64 financial-ratio features. Same official benchmark split as notebook 02, so
every score here is directly comparable to the staircase we built there.

> **Runtime**: the default fit takes ~1 minute on CPU; the `extreme` fit at the end wants a
> GPU (any Colab GPU runtime works) and takes ~10 minutes with the time limit set below.
""")

md("## TabularPredictor\n\nTo start, import AutoGluon's `TabularPredictor` and `TabularDataset` classes:")

code("""
# Installs everything the notebook needs, including the tabular foundation models used
# by the `extreme` preset (fast via uv; a no-op where already present).
import sys
!command -v uv >/dev/null || pip install -q uv
!uv pip install -q --python {sys.executable} "autogluon.tabular[tabarena]" openml

from autogluon.tabular import TabularDataset, TabularPredictor
""")

md("""
### Loading the data

`TabularDataset` is a convenience wrapper around a [pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)
and the same methods can be applied to both. We fetch the dataset from OpenML and split it
with the benchmark task's own first train/test split — the same rows as notebook 02.
""")

code("""
import openml

task = openml.tasks.get_task(363694)  # polish_companies_bankruptcy
X, y = task.get_X_and_y(dataset_format="dataframe")
train_idx, test_idx = task.get_train_test_split_indices(repeat=0, fold=0)

full_data = X.copy()
full_data[y.name] = y
train_data = TabularDataset(full_data.iloc[train_idx].reset_index(drop=True))
test_data = TabularDataset(full_data.iloc[test_idx].reset_index(drop=True))
print(f"train: {train_data.shape}, test: {test_data.shape}")
train_data.head()
""")

md("""
Each row corresponds to one company; the columns are financial ratios from its annual report
(profitability, liquidity, leverage, ...). We predict whether the company goes bankrupt
within the forecasting horizon, indicated by the `company_bankrupt` column. Note the class
imbalance — bankruptcies are rare, which will matter when we choose an evaluation metric.
""")

code("""
label = "company_bankrupt"
print(f"Unique classes: {list(train_data[label].unique())}")
print(f"Positive rate: {(train_data[label] == 'Yes').mean():.3f}")
""")

md("""
AutoGluon works with raw data, meaning you don't need to perform any data preprocessing
before fitting AutoGluon. We actively recommend that you avoid performing operations such as
missing value imputation or one-hot-encoding, as AutoGluon has dedicated logic to handle
these situations automatically.

### Pick the evaluation metric first

One decision is worth making *before* fitting: what metric your application actually cares
about. With ~7% positives, accuracy is nearly useless here — a model that answers "No
bankruptcy" for every company is already ~93% accurate. For imbalanced screening problems
the ranking quality matters, so we use `roc_auc`. Passing it to `TabularPredictor` makes
AutoGluon optimize everything — validation, model selection, ensembling — for *that* metric
instead of the default (accuracy for binary classification).

### Training

Now we initialize and fit AutoGluon's TabularPredictor in one line of code:
""")

code("""
predictor = TabularPredictor(label=label, eval_metric="roc_auc").fit(train_data)
""")

md("""
That's it! We now have a TabularPredictor that is able to make predictions on new data.

### Prediction

We can now use our trained models to make predictions on the held-out test companies:
""")

code("""
y_pred = predictor.predict(test_data)
y_pred.head()  # Predictions
""")

code("""
y_pred_proba = predictor.predict_proba(test_data)
y_pred_proba.head()  # Prediction probabilities
""")

md("### Evaluation\n\nNext, we can evaluate the predictor on the (labeled) test data:")

code("""
predictor.evaluate(test_data)
""")

md("""
`evaluate` leads with the metric the predictor optimizes (`roc_auc`), alongside auxiliary
metrics. Note how high `accuracy` looks despite the mediocre `recall` — the majority-class
trap from the metric discussion above; had we optimized accuracy, the leaderboard below
would rank models by a number that barely reflects screening quality.

We can also evaluate each model individually:
""")

code("""
predictor.leaderboard(test_data)
""")

md("""
### Loading a trained predictor

The predictor is saved to disk automatically; you can load it in a new session (or on a new
machine) by pointing `TabularPredictor.load()` at its path:
""")

code("""
predictor.path  # The path on disk where the predictor is saved
""")

code("""
# predictor = TabularPredictor.load(predictor.path)
""")

md("""
## Description of fit()

Since there are only two possible values of the `company_bankrupt` variable, this was a
binary classification problem. AutoGluon infers that automatically (along with the type of
each feature, missing-data handling, and rescaling); had we not passed `eval_metric`, it
would also have defaulted the metric to accuracy — which is exactly what we did not want
here.

We did not specify separate validation data, so AutoGluon chose a train/validation split
automatically. Rather than a single model, AutoGluon trains multiple models and ensembles
them together to obtain superior predictive performance — no hyperparameters for you to
specify.

We can view what properties AutoGluon automatically inferred about our prediction task:
""")

code("""
print("AutoGluon infers problem type is: ", predictor.problem_type)
print("AutoGluon identified the following types of features:")
print(predictor.feature_metadata)
""")

md("""
To better understand our trained predictor, we can estimate the overall importance of each
feature via permutation importance — how much the score would drop if the feature's values
were shuffled:
""")

code("""
predictor.feature_importance(test_data).head(10)
""")

md("""
Negative `importance` values mean the model may improve if re-fit without that feature.

When we call `predict()`, AutoGluon automatically predicts with the model that displayed the
best performance on validation data (i.e. the weighted ensemble):
""")

code("""
predictor.model_best
""")

md("""
We can instead specify which model to use for predictions like this:

```python
predictor.predict(test_data, model="LightGBM")
```

You can get the list of trained models via `.leaderboard()` or `.model_names()`:
""")

code("""
predictor.model_names()
""")

md("""
## Presets

The scores above used AutoGluon's default preset (`medium`) and default metric. For serious
usage, pick a preset deliberately:

| Preset  | Model Quality                                        | Use Cases | Fit Time (Ideal) | Inference Time (vs medium) | Disk Usage |
|:--------|:-----------------------------------------------------|:----------|:-----------------|:---------------------------|:-----------|
| extreme | **Far better** than best on datasets <100000 samples | (New in v1.6) The absolute cutting edge. Incorporates recent tabular foundation models Nori, TabICLv2, and TabDPT-Turbo. Every model is free for commercial use. Requires a GPU for best results. | 1x | 8x | 2x |
| noncommercial | **Far better** than best on datasets <100000 samples | (New in v1.6) `extreme` plus TabPFN-3, a frontier tabular foundation model created by Prior Labs. Commercial use of TabPFN-3 requires a license from Prior Labs ([license FAQ](https://docs.priorlabs.ai/models#tabpfn-model-license)). Requires a GPU for best results. | 1x | 8x | 2x |
| best    | State-of-the-art (SOTA), much better than high       | When accuracy is what matters and no GPU is available. Has been used to win numerous Kaggle competitions. | 16x+ | 32x+ | 16x+ |
| high    | Better than good                                     | A very powerful, portable solution with fast inference. | 16x+ | 4x | 2x |
| good    | Stronger than any other AutoML framework             | Highly portable, very fast inference. | 16x | 2x | 0.1x |
| medium  | Competitive with other top AutoML frameworks         | Initial prototyping, establishing a performance baseline. | 1x | 1x | 1x |

**If you have a GPU, start with `extreme`.** It is meta-learned from
[TabArena](https://tabarena.ai) and is far better than `best` on datasets below 100,000
samples, while training faster and producing a smaller predictor. Install its dependencies
with `pip install autogluon[tabarena]`. Without a GPU, start with `best`.

## Maximizing predictive performance

**Note:** You should not call `fit()` with entirely default arguments if you are
benchmarking AutoGluon-Tabular or hoping to maximize its accuracy! To get the best
predictive accuracy with AutoGluon, you should generally use it like this:
""")

code("""
time_limit = 600  # for quick demonstration only; set this to the longest time you are willing to wait (in seconds)
predictor = TabularPredictor(label, eval_metric="roc_auc").fit(
    train_data, time_limit=time_limit, presets="extreme"
)
""")

code("""
predictor.leaderboard(test_data)
""")

md("""
This command implements the following strategy to maximize accuracy:

- Specify `presets="extreme"`, which fits a portfolio of tabular foundation models and
  gradient-boosted trees — meta-learned from TabArena — and ensembles them with
  stacking/bagging. The default `presets="medium"` produces less accurate models but
  facilitates faster prototyping.
- Provide `eval_metric` to `TabularPredictor()` if you know what metric will be used to
  evaluate predictions in your application, as we did from the very first fit (other options
  include `f1`, `log_loss`, `mean_absolute_error`, ...).
- Include all your data in `train_data` and do not provide `tuning_data` (AutoGluon will
  split the data more intelligently to fit its needs).
- Do not specify the `hyperparameter_tune_kwargs` argument (counterintuitively,
  hyperparameter tuning is not the best way to spend a limited training budget — model
  ensembling is often superior, and notebook 02's tuning-trajectory figures show why).
- Do not specify the `hyperparameters` argument (allow AutoGluon to adaptively select which
  models/hyperparameters to use).
- Set `time_limit` to the longest amount of time you are willing to wait.

### Where this lands on notebook 02's staircase

On this exact split, notebook 02 measured: naive XGBoost **0.9628** → AutoGluon-bagged
XGBoost **0.9670** → a single TabICLv2 **0.9838**; the TabArena artifacts put a bagged TabFM
at **0.9952**. The `extreme` leaderboard above shows what an automatically composed
portfolio of foundation models and trees achieves with one `fit()` call — check the
`score_test` of the best model against those numbers.

**Next**: notebook 04 opens up what a TFM actually predicts — calibrated probabilities and
full predictive distributions.
""")

nb.cells = cells
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT}")
