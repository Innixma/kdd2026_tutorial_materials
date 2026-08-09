# Notebooks

One notebook per tutorial module, numbered in session order:

1. `01_first_contact.ipynb` — naive XGBoost, AutoGluon-bagged XGBoost, TabICLv2, and TabFM on one dataset, opening with the TFM at-a-glance table (constraints, licenses).
2. `02_autogluon_essentials.ipynb` — AutoGluon essentials on the same dataset; metric choice first, `extreme` preset last.
3. `03_tfm_zoo.ipynb` — many foundation models behind one `TabularPredictor`, bagged and ensembled.
4. `04_inside_the_prior.ipynb` — the synthetic prior TFMs are pretrained on, generated live (CPU runtime).
5. `05_pretrain_your_own.ipynb` — pretrain a working TabPFN-style model in about a minute with nanoTabPFN.
6. `06_thinking_mode.ipynb` — TabPFN-3 thinking mode on the dataset where TFMs lose to CatBoost (needs a Prior Labs token).
7. `07_interpretability.ipynb` — Shapley explanations for TabPFN-3 predictions via shapiq, plus the KV-cache `fit_mode` story.
8. `08_noniid_validation.ipynb` — naive vs `validation_structure` splits on grouped data (CPU runtime).
9. `09_timeseries_forecasting.ipynb` — forecasting with AutoGluon.

Keep saved outputs in the committed notebooks so attendees who fall behind can read along.
