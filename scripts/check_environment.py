"""Verify the tutorial environment and prefetch the model checkpoints.

Run this before the session, on the network you will actually use. It checks the
installed packages and GPU, then downloads every tabular foundation model checkpoint
the notebooks need so nothing is fetched during the exercises.
"""

from __future__ import annotations

import importlib
import shutil
import sys

REQUIRED_MODULES = [
    "autogluon.tabular",
    "tabpfn",
    "tabicl",
    "tabdpt",
]


def check_modules() -> bool:
    ok = True
    for name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "?")
            print(f"  [ok] {name} {version}")
        except ImportError as err:
            print(f"  [MISSING] {name}: {err}")
            ok = False
    return ok


def check_gpu() -> None:
    try:
        import torch
    except ImportError:
        print("  [MISSING] torch")
        return
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  [ok] GPU: {device} ({vram_gb:.0f} GB VRAM)")
    else:
        print("  [WARNING] No CUDA GPU detected. The notebooks run, but in-context inference is 12-63x slower on CPU.")


def check_disk() -> None:
    free_gb = shutil.disk_usage(".").free / 1e9
    status = "ok" if free_gb > 10 else "WARNING"
    print(f"  [{status}] {free_gb:.0f} GB free disk (checkpoints + fitted predictors need ~10 GB)")


def prefetch_checkpoints() -> None:
    """Trigger each library's checkpoint download by fitting on a toy dataset."""
    import numpy as np
    import pandas as pd

    from autogluon.tabular import TabularPredictor

    rng = np.random.default_rng(0)
    toy = pd.DataFrame(
        {
            "x1": rng.normal(size=200),
            "x2": rng.choice(list("abc"), size=200),
            "label": rng.integers(0, 2, size=200),
        },
    )
    print("  Fitting a toy predictor to download the model checkpoints (a few minutes on first run)...")
    TabularPredictor(label="label", verbosity=0).fit(
        toy,
        hyperparameters={"TABPFN": {}, "TABICL": {}, "TABDPT": {}},
        num_bag_folds=0,
    )
    print("  [ok] checkpoints cached")


if __name__ == "__main__":
    print("Modules:")
    modules_ok = check_modules()
    print("Hardware:")
    check_gpu()
    check_disk()
    if not modules_ok:
        print("\nInstall the missing packages first: pip install -r requirements.txt")
        sys.exit(1)
    print("Checkpoints:")
    prefetch_checkpoints()
    print("\nEnvironment ready.")
