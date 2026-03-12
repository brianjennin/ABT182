"""Bare-bones regression plots: three climate predictors vs yield, plus yield time series."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REG_DIR = Path(__file__).resolve().parent
OUT_DIR = REG_DIR / "output"


def load_data() -> pd.DataFrame:
    """Load and merge yield, ETo, temperature, and precipitation data."""
    # Yield (wide -> long)
    yld = pd.read_csv(REG_DIR / "grape_normalized_yield_selected_counties.csv")
    yld = yld.melt(id_vars="county", var_name="year", value_name="yield_tons_per_acre")
    yld["year"] = yld["year"].astype(int)
    yld = yld.dropna(subset=["yield_tons_per_acre"])

    # ETo
    eto = pd.read_csv(REG_DIR / "county_vineyard_eto_annual.csv")
    eto = eto.rename(columns={"annual_eto_in": "eto_annual_in"})
    eto = eto[["year", "county", "eto_annual_in"]]

    # Temperature
    temp = pd.read_excel(REG_DIR / "County temp years.xlsx")
    temp = temp.dropna(subset=["County", "Value"])
    temp["county"] = temp["County"].str.replace(r"\s*County$", "", regex=True).str.strip()
    temp["temp_avg_f"] = temp["Value"].astype(str).str.replace("°F", "", regex=False).astype(float)
    temp = temp.rename(columns={"Year": "year"})[["year", "county", "temp_avg_f"]]

    # Precipitation
    precip = pd.read_excel(REG_DIR / "ca_county_precip.xlsx")
    precip = precip.dropna(subset=["county_name"])
    month_cols = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    precip["precip_annual_in"] = precip[month_cols].sum(axis=1)
    precip = precip.rename(columns={"county_name": "county"})[["year", "county", "precip_annual_in"]]

    # Merge
    df = yld.merge(eto, on=["year", "county"], how="left")
    df = df.merge(temp, on=["year", "county"], how="left")
    df = df.merge(precip, on=["year", "county"], how="left")
    return df.sort_values(["county", "year"]).reset_index(drop=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    pred_cols = ["eto_annual_in", "temp_avg_f", "precip_annual_in"]
    labels = {"eto_annual_in": "Annual ETo (in)",
              "temp_avg_f": "Avg Temperature (\u00b0F)",
              "precip_annual_in": "Annual Precip (in)"}

    # -- 3-panel regression scatter plot --
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
                f"R\u00b2 = {r**2:.3f},  p = {p:.4f}",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.set_xlabel(labels[col], fontsize=11)
        ax.set_ylabel("Yield (tons/acre)", fontsize=11)
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("OLS Regression: Climate Variables vs Grape Yield", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "regression_output.png", dpi=150)
    plt.close(fig)

    # -- Yield time series --
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

    # -- Climate variable time series (1x3) --
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col in zip(axes, pred_cols):
        for county, grp in df.dropna(subset=[col]).groupby("county"):
            ax.plot(grp["year"], grp[col], "o-", label=county)
        ax.set_xlabel("Year")
        ax.set_ylabel(labels[col])
        ax.set_title(f"{labels[col]} Over Time")
    axes[0].legend(loc="best", fontsize=7)
    fig.suptitle("Climate Variables Over Time by County", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "climate_timeseries.png", dpi=150)
    plt.close(fig)

    print(f"Saved plots to {OUT_DIR}/")


if __name__ == "__main__":
    main()
