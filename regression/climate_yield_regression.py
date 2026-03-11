"""
climate_yield_regression.py
---------------------------
OLS regression of climate variables (ET, temperature, precipitation) on
grape yield (tons/acre) for the selected counties present in the yield
dataset, across all available years.

Data sources (within this repository):
  - Yield:   Emily/Emily/grape_normalized_yield_selected_counties.csv
  - Climate: county_vineyard_eto_annual.csv  (annual ETo & precip by county)
  - Climate: county_vineyard_eto_monthly.csv (monthly ETo & precip by county)

NOTE: The current climate dataset only contains ET and precipitation.
      Precipitation values are all 0.0, and temperature is not available.
      The regression therefore uses ET as the sole climate predictor with
      meaningful variation.  Results should be interpreted with this
      limitation in mind.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
YIELD_CSV = ROOT / "Emily" / "Emily" / "grape_normalized_yield_selected_counties.csv"
ANNUAL_CSV = ROOT / "county_vineyard_eto_annual.csv"
MONTHLY_CSV = ROOT / "county_vineyard_eto_monthly.csv"
OUT_DIR = Path(__file__).resolve().parent / "output"


def load_yield() -> pd.DataFrame:
    """Load yield data and melt from wide to long format."""
    df = pd.read_csv(YIELD_CSV)
    df = df.melt(id_vars="county", var_name="year", value_name="yield_tons_per_acre")
    df["year"] = df["year"].astype(int)
    df = df.dropna(subset=["yield_tons_per_acre"])
    return df


def load_annual_climate() -> pd.DataFrame:
    """Load annual ETo and precipitation by county."""
    df = pd.read_csv(ANNUAL_CSV)
    df = df.rename(columns={"annual_eto_in": "eto_annual_in",
                            "annual_precip_in": "precip_annual_in"})
    return df[["year", "county", "eto_annual_in", "precip_annual_in"]]


def load_monthly_climate() -> pd.DataFrame:
    """Load monthly climate data and compute growing-season summaries.

    Growing season is defined as April–October (months 4–10) for California
    vineyards.  We compute:
      - eto_gs_in      : total growing-season ETo (inches)
      - precip_gs_in   : total growing-season precipitation (inches)
      - eto_gs_mean_mo : mean monthly ETo during growing season
    """
    df = pd.read_csv(MONTHLY_CSV)
    gs = df[df["month"].between(4, 10)]

    agg = gs.groupby(["year", "county"]).agg(
        eto_gs_in=("eto_in", "sum"),
        precip_gs_in=("precip_in", "sum"),
        eto_gs_mean_mo=("eto_in", "mean"),
    ).reset_index()
    return agg


def merge_data() -> pd.DataFrame:
    """Merge yield with annual and growing-season climate data."""
    yld = load_yield()
    ann = load_annual_climate()
    gs = load_monthly_climate()

    merged = yld.merge(ann, on=["year", "county"], how="inner")
    merged = merged.merge(gs, on=["year", "county"], how="left")
    merged = merged.sort_values(["county", "year"]).reset_index(drop=True)
    return merged


def run_regressions(df: pd.DataFrame) -> dict:
    """Run OLS regressions and return results dict."""
    results = {}

    # ── 1. Pooled OLS: yield ~ ETo (annual) ───────────────────────────────
    X = sm.add_constant(df["eto_annual_in"])
    y = df["yield_tons_per_acre"]
    model = sm.OLS(y, X).fit()
    results["pooled_annual_eto"] = model

    # ── 2. Pooled OLS: yield ~ ETo (growing season) ───────────────────────
    subset_gs = df.dropna(subset=["eto_gs_in"])
    if len(subset_gs) > 3:
        X_gs = sm.add_constant(subset_gs["eto_gs_in"])
        y_gs = subset_gs["yield_tons_per_acre"]
        model_gs = sm.OLS(y_gs, X_gs).fit()
        results["pooled_gs_eto"] = model_gs

    # ── 3. Pooled OLS with county fixed effects ───────────────────────────
    county_dummies = pd.get_dummies(df["county"], drop_first=True, dtype=float)
    X_fe = sm.add_constant(pd.concat([df[["eto_annual_in"]], county_dummies], axis=1))
    model_fe = sm.OLS(y, X_fe).fit()
    results["fe_county_annual_eto"] = model_fe

    # ── 4. Per-county simple regressions ──────────────────────────────────
    for county, grp in df.groupby("county"):
        if len(grp) < 3:
            continue
        X_c = sm.add_constant(grp["eto_annual_in"])
        y_c = grp["yield_tons_per_acre"]
        model_c = sm.OLS(y_c, X_c).fit()
        results[f"county_{county}"] = model_c

    return results


def print_results(results: dict, df: pd.DataFrame) -> str:
    """Format regression results as a readable report."""
    lines = []
    lines.append("=" * 78)
    lines.append("REGRESSION ANALYSIS: CLIMATE VARIABLES ON GRAPE YIELD")
    lines.append("=" * 78)
    lines.append("")

    # Data summary
    lines.append("DATA SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Counties : {sorted(df['county'].unique())}")
    lines.append(f"  Years    : {sorted(df['year'].unique())}")
    lines.append(f"  N obs    : {len(df)}")
    lines.append("")

    # Variable summary
    lines.append("VARIABLE STATISTICS")
    lines.append("-" * 40)
    for col in ["yield_tons_per_acre", "eto_annual_in", "precip_annual_in"]:
        if col in df.columns:
            lines.append(f"  {col}:")
            lines.append(f"    mean={df[col].mean():.3f}  std={df[col].std():.3f}  "
                         f"min={df[col].min():.3f}  max={df[col].max():.3f}")
    lines.append("")

    # Data limitations
    lines.append("DATA LIMITATIONS")
    lines.append("-" * 40)
    if df["precip_annual_in"].sum() == 0:
        lines.append("  * Precipitation: ALL VALUES ARE 0.0 — excluded from regression")
    lines.append("  * Temperature: NOT AVAILABLE in current dataset")
    lines.append("  * ET (evapotranspiration) is the only climate predictor with variation")
    lines.append("  * ET is driven by temperature, solar radiation, humidity, and wind,")
    lines.append("    so it serves as a partial proxy for temperature effects")
    lines.append("")

    # Regression results
    for name, model in results.items():
        lines.append("=" * 78)
        lines.append(f"MODEL: {name}")
        lines.append("=" * 78)
        lines.append(model.summary().as_text())
        lines.append("")

    return "\n".join(lines)


def create_plots(df: pd.DataFrame, results: dict):
    """Generate diagnostic and exploratory plots."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # ── 1. Scatter: ETo vs Yield by county ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for county, grp in df.groupby("county"):
        ax.scatter(grp["eto_annual_in"], grp["yield_tons_per_acre"],
                   label=county, s=60, alpha=0.8)
    # Overall trend line
    m = results.get("pooled_annual_eto")
    if m:
        x_range = np.linspace(df["eto_annual_in"].min(), df["eto_annual_in"].max(), 100)
        ax.plot(x_range, m.params["const"] + m.params["eto_annual_in"] * x_range,
                "k--", alpha=0.6, label=f"OLS (R²={m.rsquared:.3f})")
    ax.set_xlabel("Annual ETo (inches)")
    ax.set_ylabel("Yield (tons/acre)")
    ax.set_title("Annual ETo vs Grape Yield by County")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scatter_eto_vs_yield.png", dpi=150)
    plt.close(fig)

    # ── 2. Per-county scatter with individual trend lines ─────────────────
    counties = sorted(df["county"].unique())
    n_counties = len(counties)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
    axes = axes.flatten()
    for i, county in enumerate(counties):
        ax = axes[i]
        grp = df[df["county"] == county]
        ax.scatter(grp["eto_annual_in"], grp["yield_tons_per_acre"], s=50)
        key = f"county_{county}"
        if key in results:
            m_c = results[key]
            x_r = np.linspace(grp["eto_annual_in"].min(), grp["eto_annual_in"].max(), 50)
            ax.plot(x_r, m_c.params["const"] + m_c.params["eto_annual_in"] * x_r,
                    "r-", alpha=0.7)
            ax.set_title(f"{county} (R²={m_c.rsquared:.3f})")
        else:
            ax.set_title(county)
        ax.set_xlabel("Annual ETo (in)")
        ax.set_ylabel("Yield (t/ac)")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Per-County: Annual ETo vs Grape Yield", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scatter_per_county.png", dpi=150)
    plt.close(fig)

    # ── 3. Yield time series by county ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for county, grp in df.groupby("county"):
        ax.plot(grp["year"], grp["yield_tons_per_acre"], "o-", label=county)
    ax.set_xlabel("Year")
    ax.set_ylabel("Yield (tons/acre)")
    ax.set_title("Grape Yield Over Time by County")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "yield_timeseries.png", dpi=150)
    plt.close(fig)

    # ── 4. Residual plot for pooled model ─────────────────────────────────
    m = results.get("pooled_annual_eto")
    if m:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(m.fittedvalues, m.resid, alpha=0.7)
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Fitted — Pooled OLS (yield ~ ETo)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "residuals_pooled.png", dpi=150)
        plt.close(fig)

    # ── 5. Correlation heatmap ────────────────────────────────────────────
    num_cols = ["yield_tons_per_acre", "eto_annual_in"]
    if "eto_gs_in" in df.columns:
        num_cols.append("eto_gs_in")
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                square=True, ax=ax)
    ax.set_title("Correlation Matrix")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"  Plots saved to {OUT_DIR}/")


def main():
    print("Loading and merging data...")
    df = merge_data()

    print(f"Merged dataset: {len(df)} observations")
    print(f"Counties: {sorted(df['county'].unique())}")
    print(f"Years:    {sorted(df['year'].unique())}")
    print()

    print("Running regressions...")
    results = run_regressions(df)

    report = print_results(results, df)
    print(report)

    # Save report
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "regression_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    # Save merged dataset
    csv_path = OUT_DIR / "merged_climate_yield.csv"
    df.to_csv(csv_path, index=False)
    print(f"Merged data saved to {csv_path}")

    print("\nGenerating plots...")
    create_plots(df, results)

    print("\nDone.")


if __name__ == "__main__":
    main()
