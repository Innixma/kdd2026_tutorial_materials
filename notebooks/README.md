# Notebooks

One notebook per tutorial module, numbered in session order:

1. `01_first_contact.ipynb` — naive XGBoost, AutoGluon-bagged XGBoost, TabICLv2, and TabFM on one dataset, opening with the TFM at-a-glance table (constraints, licenses).
2. `02_autogluon_essentials.ipynb` — AutoGluon essentials on the same dataset; metric choice first, `extreme` preset last.
3. `03_inside_the_prior.ipynb` — the synthetic prior TFMs are pretrained on, generated live (CPU runtime).
4. `04_tfm_zoo.ipynb` — many foundation models behind one `TabularPredictor`, bagged and ensembled.
5. `05_thinking_mode.ipynb` — TabPFN-3 thinking mode on the dataset where TFMs lose to CatBoost (needs a Prior Labs token).
6. `06_noniid_validation.ipynb` — naive vs `validation_structure` splits on grouped data (CPU runtime).
7. `07_timeseries_forecasting.ipynb` — forecasting with AutoGluon.

Keep saved outputs in the committed notebooks so attendees who fall behind can read along.
