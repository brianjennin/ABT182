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
YIELD_CSV = REG_DIR / "grape_normalized_yield_selected_counties.csv"
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
    """Generate regression output and yield time-series plots."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    pred_cols = ["eto_annual_in", "temp_avg_f", "precip_annual_in"]
    labels = {"eto_annual_in": "Annual ETo (in)",
              "temp_avg_f": "Avg Temperature (°F)",
              "precip_annual_in": "Annual Precip (in)"}

    # ── 1. Regression output: scatter + OLS line per predictor ───────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col in zip(axes, pred_cols):
        sub = df.dropna(subset=[col])
        for county, grp in sub.groupby("county"):
            ax.scatter(grp[col], grp["yield_tons_per_acre"], s=50, alpha=0.8, label=county)
        slope, intercept, r, p, se = stats.linregress(sub[col], sub["yield_tons_per_acre"])
        x_line = np.linspace(sub[col].min(), sub[col].max(), 100)
        ax.plot(x_line, intercept + slope * x_line, "k--", linewidth=1.5)
        sign = "+" if intercept >= 0 else "-"
        ax.text(0.05, 0.95,
                f"y = {slope:.3f}x {sign} {abs(intercept):.2f}\n"
                f"R² = {r**2:.3f},  p = {p:.4f}",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.set_xlabel(labels.get(col, col), fontsize=11)
        ax.set_ylabel("Yield (tons/acre)", fontsize=11)
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("OLS Regression: Climate Variables vs Grape Yield", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "regression_output.png", dpi=150)
    plt.close(fig)

    # ── 2. Yield time series by county ───────────────────────────────────
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
