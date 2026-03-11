"""
climate_yield_regression.py
---------------------------
OLS regression of climate variables (ET, temperature, precipitation) on
grape yield (tons/acre) for the selected counties present in the yield
dataset, across all available years.

Data sources (within regression/ folder):
  - Yield:  grape_normalized_yield_selected_counties1.csv
  - ET:     county_vineyard_eto_annual.csv
  - Temp:   County temp years.xlsx
  - Precip: ca_county_precip.xlsx
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
REG_DIR = Path(__file__).resolve().parent
YIELD_CSV = REG_DIR / "grape_normalized_yield_selected_counties1.csv"
ETO_CSV = REG_DIR / "county_vineyard_eto_annual.csv"
TEMP_XLSX = REG_DIR / "County temp years.xlsx"
PRECIP_XLSX = REG_DIR / "ca_county_precip.xlsx"
OUT_DIR = REG_DIR / "output"


# ── data loading ───────────────────────────────────────────────────────────

def load_yield() -> pd.DataFrame:
    """Load yield data and melt from wide to long format."""
    df = pd.read_csv(YIELD_CSV)
    df = df.melt(id_vars="county", var_name="year", value_name="yield_tons_per_acre")
    df["year"] = df["year"].astype(int)
    df = df.dropna(subset=["yield_tons_per_acre"])
    return df


def load_eto() -> pd.DataFrame:
    """Load annual ETo by county."""
    df = pd.read_csv(ETO_CSV)
    df = df.rename(columns={"annual_eto_in": "eto_annual_in"})
    return df[["year", "county", "eto_annual_in"]]


def load_temperature() -> pd.DataFrame:
    """Load annual average temperature by county from NOAA data.

    The xlsx has columns: Year, County, Value (e.g. '57.4°F'), Anomaly, Rank,
    Mean(1901-2000).  We parse °F values to float and strip ' County' from
    county names.
    """
    df = pd.read_excel(TEMP_XLSX)
    df = df.dropna(subset=["County", "Value"])
    # Strip ' County' suffix to match other datasets
    df["county"] = df["County"].str.replace(r"\s*County$", "", regex=True).str.strip()
    # Parse temperature: remove '°F' and convert to float
    df["temp_avg_f"] = df["Value"].astype(str).str.replace("°F", "", regex=False).astype(float)
    df = df.rename(columns={"Year": "year"})
    return df[["year", "county", "temp_avg_f"]]


def load_precipitation() -> pd.DataFrame:
    """Load monthly precipitation and compute annual total.

    The xlsx has columns: county_name, year, Jan..Dec, plus an unnamed avg column.
    We sum Jan–Dec to get annual total precipitation (inches).
    """
    df = pd.read_excel(PRECIP_XLSX)
    df = df.dropna(subset=["county_name"])
    month_cols = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    df["precip_annual_in"] = df[month_cols].sum(axis=1)
    # Growing season precipitation (Apr–Oct)
    gs_cols = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]
    df["precip_gs_in"] = df[gs_cols].sum(axis=1)
    df = df.rename(columns={"county_name": "county"})
    return df[["year", "county", "precip_annual_in", "precip_gs_in"]]


def merge_data() -> pd.DataFrame:
    """Merge yield with ET, temperature, and precipitation data."""
    yld = load_yield()
    eto = load_eto()
    temp = load_temperature()
    precip = load_precipitation()

    merged = yld.merge(eto, on=["year", "county"], how="left")
    merged = merged.merge(temp, on=["year", "county"], how="left")
    merged = merged.merge(precip, on=["year", "county"], how="left")
    merged = merged.sort_values(["county", "year"]).reset_index(drop=True)
    return merged


# ── regressions ────────────────────────────────────────────────────────────

def run_regressions(df: pd.DataFrame) -> dict:
    """Run OLS regressions and return results dict."""
    results = {}
    y = df["yield_tons_per_acre"]

    # ── 1. Multiple regression: yield ~ ETo + Temp + Precip ───────────────
    predictors = ["eto_annual_in", "temp_avg_f", "precip_annual_in"]
    subset = df.dropna(subset=predictors)
    if len(subset) > len(predictors) + 1:
        X = sm.add_constant(subset[predictors])
        model = sm.OLS(subset["yield_tons_per_acre"], X).fit()
        results["full_model_eto_temp_precip"] = model

    # ── 2. Simple regressions for each predictor ─────────────────────────
    for pred in predictors:
        sub = df.dropna(subset=[pred])
        if len(sub) > 3:
            X = sm.add_constant(sub[pred])
            model = sm.OLS(sub["yield_tons_per_acre"], X).fit()
            results[f"simple_{pred}"] = model

    # ── 3. Multiple regression with county fixed effects ─────────────────
    subset = df.dropna(subset=predictors)
    if len(subset) > len(predictors) + 2:
        county_dummies = pd.get_dummies(subset["county"], drop_first=True, dtype=float)
        X_fe = sm.add_constant(
            pd.concat([subset[predictors].reset_index(drop=True),
                       county_dummies.reset_index(drop=True)], axis=1)
        )
        model_fe = sm.OLS(subset["yield_tons_per_acre"].reset_index(drop=True), X_fe).fit()
        results["fe_county_full"] = model_fe

    # ── 4. Per-county regressions ────────────────────────────────────────
    for county, grp in df.groupby("county"):
        sub = grp.dropna(subset=predictors)
        if len(sub) < len(predictors) + 2:
            # Not enough observations for multiple regression; try simple ETo
            sub_eto = grp.dropna(subset=["eto_annual_in"])
            if len(sub_eto) >= 3:
                X_c = sm.add_constant(sub_eto["eto_annual_in"])
                model_c = sm.OLS(sub_eto["yield_tons_per_acre"], X_c).fit()
                results[f"county_{county}_eto_only"] = model_c
            continue
        X_c = sm.add_constant(sub[predictors])
        model_c = sm.OLS(sub["yield_tons_per_acre"], X_c).fit()
        results[f"county_{county}"] = model_c

    return results


def print_results(results: dict, df: pd.DataFrame) -> str:
    """Format regression results as a readable report."""
    lines = []
    lines.append("=" * 78)
    lines.append("REGRESSION ANALYSIS: ET, TEMPERATURE & PRECIPITATION ON GRAPE YIELD")
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
    for col in ["yield_tons_per_acre", "eto_annual_in", "temp_avg_f", "precip_annual_in", "precip_gs_in"]:
        if col in df.columns:
            sub = df[col].dropna()
            lines.append(f"  {col} (n={len(sub)}):")
            lines.append(f"    mean={sub.mean():.3f}  std={sub.std():.3f}  "
                         f"min={sub.min():.3f}  max={sub.max():.3f}")
    lines.append("")

    # Missing data check
    lines.append("DATA COVERAGE")
    lines.append("-" * 40)
    for col in ["eto_annual_in", "temp_avg_f", "precip_annual_in"]:
        n_avail = df[col].notna().sum()
        n_miss = df[col].isna().sum()
        lines.append(f"  {col}: {n_avail} available, {n_miss} missing")
    lines.append("")

    # Regression results
    for name, model in results.items():
        lines.append("=" * 78)
        lines.append(f"MODEL: {name}")
        lines.append("=" * 78)
        lines.append(model.summary().as_text())
        lines.append("")

    return "\n".join(lines)


# ── plots ──────────────────────────────────────────────────────────────────

def create_plots(df: pd.DataFrame, results: dict):
    """Generate diagnostic and exploratory plots."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # ── 1. Scatter matrix: all predictors vs yield ───────────────────────
    pred_cols = ["eto_annual_in", "temp_avg_f", "precip_annual_in"]
    avail_cols = [c for c in pred_cols if c in df.columns and df[c].notna().any()]

    fig, axes = plt.subplots(1, len(avail_cols), figsize=(6 * len(avail_cols), 5))
    if len(avail_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, avail_cols):
        sub = df.dropna(subset=[col])
        for county, grp in sub.groupby("county"):
            ax.scatter(grp[col], grp["yield_tons_per_acre"], label=county, s=60, alpha=0.8)
        ax.set_xlabel(col)
        ax.set_ylabel("Yield (tons/acre)")
        ax.set_title(f"Yield vs {col}")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("Climate Predictors vs Grape Yield", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scatter_predictors_vs_yield.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 2. Per-county panels for each predictor ──────────────────────────
    counties = sorted(df["county"].unique())
    for col in avail_cols:
        fig, axes_grid = plt.subplots(2, 3, figsize=(14, 8))
        axes_flat = axes_grid.flatten()
        for i, county in enumerate(counties):
            ax = axes_flat[i]
            grp = df[df["county"] == county].dropna(subset=[col])
            ax.scatter(grp[col], grp["yield_tons_per_acre"], s=50)
            if len(grp) >= 3:
                slope, intercept, r, p, se = stats.linregress(grp[col], grp["yield_tons_per_acre"])
                x_r = np.linspace(grp[col].min(), grp[col].max(), 50)
                ax.plot(x_r, intercept + slope * x_r, "r-", alpha=0.7)
                ax.set_title(f"{county} (r={r:.3f}, p={p:.3f})")
            else:
                ax.set_title(county)
            ax.set_xlabel(col)
            ax.set_ylabel("Yield (t/ac)")
        for j in range(len(counties), len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.suptitle(f"Per-County: {col} vs Grape Yield", fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"scatter_per_county_{col}.png", dpi=150)
        plt.close(fig)

    # ── 3. Yield time series by county ───────────────────────────────────
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

    # ── 4. Residual plot for full model ──────────────────────────────────
    m = results.get("full_model_eto_temp_precip")
    if m:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(m.fittedvalues, m.resid, alpha=0.7)
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Fitted — Full Model (yield ~ ETo + Temp + Precip)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "residuals_full_model.png", dpi=150)
        plt.close(fig)

    # ── 5. Correlation heatmap ───────────────────────────────────────────
    num_cols = ["yield_tons_per_acre"] + avail_cols
    if "precip_gs_in" in df.columns:
        num_cols.append("precip_gs_in")
    corr = df[num_cols].dropna().corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                square=True, ax=ax)
    ax.set_title("Correlation Matrix: Yield & Climate Variables")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"  Plots saved to {OUT_DIR}/")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print("Loading and merging data...")
    df = merge_data()

    print(f"Merged dataset: {len(df)} observations")
    print(f"Counties: {sorted(df['county'].unique())}")
    print(f"Years:    {sorted(df['year'].unique())}")
    print()
    print("Columns:", df.columns.tolist())
    print()
    print(df.head(10).to_string())
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
