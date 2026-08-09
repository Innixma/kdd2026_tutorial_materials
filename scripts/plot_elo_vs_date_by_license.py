"""TabArena Elo vs release date, split by license class (permissive vs noncommercial).

The license-story variant of tabarena's elo-vs-date-introduced figure: dots are model
families colored by family type (same colors as the interactive plots), marker shape
encodes the license class, and two step frontiers trace the best-so-far Elo per class.
Reference pipelines (AutoGluon presets) are excluded: the story is about model licenses.

Elo source: the stock-TabArenaContext leaderboard (`tabarena_context_default` eval run).
License classes follow the AutoGluon foundation-models tutorial table plus the wrappers'
docstring `License:` lines; families that cannot be confidently classified are excluded
and listed on stdout.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from tabarena.contexts.tabarena._tabarena_method_metadata_2025_11_01 import ag_140_eq_4h8c_metadata
from tabarena.contexts.tabarena.methods import tabarena_method_metadata_collection
from tabarena.plot.plot_pareto_focus import FAMILY_COLORS
from tabarena.website.website_format import get_model_family

LEADERBOARDS = {
    "all": "/home/nick_priorlabs_ai/workspace_tabpfn_plus/code/external/tabarena/tmp_scripts/eval_output/tabarena_context_default/tabarena_leaderboard.csv",
    "medium": "/home/nick_priorlabs_ai/workspace_tabpfn_plus/code/external/tabarena/tmp_scripts/eval_output/tabarena_context_default_medium/tabarena_leaderboard.csv",
    "small": "/home/nick_priorlabs_ai/workspace_tabpfn_plus/code/external/tabarena/tmp_scripts/eval_output/tabarena_context_default_small/tabarena_leaderboard.csv",
    "!tiny": "/home/nick_priorlabs_ai/workspace_tabpfn_plus/code/external/tabarena/tmp_scripts/eval_output/tabarena_context_default_nottiny/tabarena_leaderboard.csv",
    "tiny": "/home/nick_priorlabs_ai/workspace_tabpfn_plus/code/external/tabarena/tmp_scripts/eval_output/tabarena_context_default_tiny/tabarena_leaderboard.csv",
}
OUT_DIR = Path(__file__).resolve().parents[1] / "slides"

#: display-name (family label) -> license class. Sources: the AutoGluon
#: tabular-foundational-models tutorial table and the wrappers' `License:` docstrings.
LICENSE_CLASS = {
    # Noncommercial weights (commercial use requires a license)
    "RealTabPFN-2.5": "noncommercial",
    "TabPFN-2.6": "noncommercial",
    "TabPFN-3": "noncommercial",
    "EXAONE-Tabular": "noncommercial",
    # Permissive (Apache/MIT/BSD or explicitly commercial-permitted)
    "TabPFNv2": "permissive",  # Prior Labs License, commercial use permitted
    "TabICL": "permissive",
    "TabICLv2": "permissive",
    "TabDPT": "permissive",
    "TabDPT-Turbo": "permissive",
    "Nori": "permissive",
    "Nori-30M": "permissive",
    "TabFM": "noncommercial",  # code Apache-2.0, weights noncommercial
    "TabSwift": "permissive",
    "Mitra": "permissive",
    "OrionMSP": "permissive",
    "LimiX": "permissive",
    "iLTM": "permissive",
    "RealMLP": "permissive",
    "TabM": "permissive",
    "ModernNCA": "permissive",
    "LightGBM": "permissive",
    "CatBoost": "permissive",
    "XGBoost": "permissive",
    "RandomForest": "permissive",
    "ExtraTrees": "permissive",
    "EBM": "permissive",
    "KNN": "permissive",
    "Linear": "permissive",
    "TorchMLP": "permissive",
    "FastaiMLP": "permissive",
    "ChimeraBoost": "permissive",
    "BetaTabPFN": "permissive",
    "SAP-RPT-OSS": "permissive",
    "TabFlex": "permissive",
    "TabSTAR": "permissive",
    "xRFM": "permissive",
    "PerpetualBooster": "permissive",  # AGPL-3.0: copyleft, but commercial use is permitted
}

#: AutoGluon preset license classes: a preset is noncommercial if any bundled model is.
REFERENCE_LICENSE = {
    "AutoGluon 1.4 (best, 4h)": "permissive",
    "AutoGluon 1.4 (extreme, 4h)": "permissive",
    "AutoGluon 1.5 (extreme, 4h)": "permissive",  # bundles RealTabPFN-2, not 2.5
    "AutoGluon 1.6 (extreme, 4h)": "permissive",
    "AutoGluon 1.6 (noncommercial, 4h)": "noncommercial",
}

MARKERS = {"permissive": "o", "noncommercial": "D"}
FRONTIER_STYLE = {"permissive": dict(ls="-", label="Best permissive"), "noncommercial": dict(ls="--", label="Best noncommercial")}
INK = "#333333"


def build_frame(include_reference_pipelines: bool = False, subset: str = "all") -> pd.DataFrame:
    lb = pd.read_csv(LEADERBOARDS[subset])
    meta = pd.DataFrame(
        {
            "ta_name": m.method,
            "ta_suite": m.suite,
            "date_introduced": getattr(m, "date_introduced", None),
            "display_name": getattr(m, "display_name", None) or m.method,
        }
        for m in [*tabarena_method_metadata_collection.method_metadata_lst, ag_140_eq_4h8c_metadata]
    ).drop_duplicates(subset=["ta_name", "ta_suite"])

    df = lb.merge(meta, on=["ta_name", "ta_suite"], how="left")
    date = df["date_introduced"].astype("string")
    date = date.mask(date.str.len() == 4, date + "-01")
    date = date.mask(date.str.len() == 7, date + "-01")
    df["_date"] = pd.to_datetime(date, format="%Y-%m-%d", errors="coerce")
    df = df[df["_date"].notna() & df["elo"].notna()]
    df["_label"] = df["display_name"].fillna(df["ta_name"])
    df = df.sort_values("elo").drop_duplicates(subset=["_label"], keep="last")

    df["family"] = df["_label"].map(get_model_family)
    license_map = dict(LICENSE_CLASS)
    if include_reference_pipelines:
        license_map.update(REFERENCE_LICENSE)
    else:
        df = df[df["family"] != "Reference Pipeline"]
    df["license"] = df["_label"].map(license_map)
    excluded = sorted(df[df["license"].isna()]["_label"])
    if excluded:
        print(f"excluded (license unverified): {excluded}")
    return df[df["license"].notna()].sort_values("_date")


def main(
    include_reference_pipelines: bool = False,
    file_stem: str = "elo_vs_date_by_license",
    x_start: str = "2013-06-01",
    y_min: float | None = None,
    subset: str = "all",
) -> None:
    df = build_frame(include_reference_pipelines=include_reference_pipelines, subset=subset)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    for lic, group in df.groupby("license"):
        for family, sub in group.groupby("family"):
            ax.scatter(
                sub["_date"], sub["elo"],
                marker=MARKERS[lic], s=90,
                c=FAMILY_COLORS.get(family, FAMILY_COLORS["Other"]),
                edgecolor="white", linewidth=0.6, zorder=3,
            )

    # Step frontiers: best-so-far Elo per license class, extended to the axis end.
    # Computed on the full history; the x-axis is clipped to the modern era below, so
    # the permissive frontier enters the frame at its pre-2014 level (the tree-era best).
    label_points = []
    x_end = df["_date"].max() + pd.Timedelta(days=45)
    perm = df[df["license"] == "permissive"]
    perm_front = perm[perm["elo"] >= perm["elo"].cummax()]
    for lic, style in FRONTIER_STYLE.items():
        sub = df[df["license"] == lic]
        front = sub[sub["elo"] >= sub["elo"].cummax()]
        xs = list(front["_date"]) + [x_end]
        ys = list(front["elo"]) + [front["elo"].iloc[-1]]
        if lic == "noncommercial" and not front.empty:
            # Start on the permissive frontier at the divergence date: until its first
            # exclusive model, the noncommercial track IS the permissive one.
            t0 = front["_date"].iloc[0]
            carried = perm_front[perm_front["_date"] <= t0]
            if not carried.empty:
                xs = [t0, *xs]
                ys = [carried["elo"].iloc[-1], *ys]
        ax.step(xs, ys, where="post", color=INK, lw=2, zorder=2, **style)
        label_points.append(front)

    x_start = pd.Timestamp(x_start)
    ax.set_xlim(x_start, x_end)
    if y_min is not None:
        ax.set_ylim(bottom=y_min)

    # Direct labels on the frontier-advancing methods, as data-coordinate texts so
    # adjustText can rearrange them within the axes.
    front_all = pd.concat(label_points)
    front_all = front_all[front_all["_date"] >= x_start]
    texts = [
        ax.text(p["_date"], p["elo"], p["_label"], fontsize=8.5, color=INK, zorder=5)
        for _, p in front_all.iterrows()
    ]
    try:
        from adjustText import adjust_text

        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.7),
        )
    except ImportError:
        pass

    ax.set_xlabel("Model release date")
    ax.set_ylabel("TabArena Elo (best per family)")
    subset_suffix = "" if subset == "all" else f" ({subset} datasets)"
    ax.set_title(f"Tabular model Elo by release date: permissive vs noncommercial licenses{subset_suffix}", fontsize=13)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, alpha=0.25, lw=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    visible = df[(df["_date"] >= x_start) & (df["elo"] >= (y_min if y_min is not None else -1e9))]
    visible_families = set(visible["family"])
    family_handles = [
        plt.Line2D([], [], marker="s", ls="", color=c, label=f)
        for f, c in FAMILY_COLORS.items()
        if f in visible_families
    ]
    lic_handles = [
        plt.Line2D([], [], marker=MARKERS[lic], ls=FRONTIER_STYLE[lic]["ls"], color=INK, label=FRONTIER_STYLE[lic]["label"])
        for lic in FRONTIER_STYLE
    ]
    ax.legend(handles=family_handles + lic_handles, loc="upper left", fontsize=9, frameon=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{file_stem}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved to {OUT_DIR}/{file_stem}.png and .pdf")


if __name__ == "__main__":
    main()
    main(include_reference_pipelines=True, file_stem="elo_vs_date_by_license_with_autogluon")
    # Zoomed variant: the modern era only, from AutoGluon 1.4 "best" onward, weak methods hidden.
    main(
        include_reference_pipelines=True,
        file_stem="elo_vs_date_by_license_zoom",
        x_start="2023-09-01",
        y_min=1420,
    )
    main(
        include_reference_pipelines=True,
        file_stem="elo_vs_date_by_license_zoom_medium",
        x_start="2023-09-01",
        y_min=1420,
        subset="medium",
    )
    main(
        include_reference_pipelines=True,
        file_stem="elo_vs_date_by_license_zoom_small",
        x_start="2023-09-01",
        y_min=1420,
        subset="small",
    )
    main(
        include_reference_pipelines=True,
        file_stem="elo_vs_date_by_license_zoom_nottiny",
        x_start="2023-09-01",
        y_min=1420,
        subset="!tiny",
    )
    main(
        include_reference_pipelines=True,
        file_stem="elo_vs_date_by_license_zoom_tiny",
        x_start="2023-09-01",
        y_min=1370,
        subset="tiny",
    )
