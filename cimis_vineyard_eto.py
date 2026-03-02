"""
cimis_vineyard_eto.py
---------------------
Pure-Python replacement for the ArcGIS Pro + Spatial-CIMIS GeoTIFF workflow.

Original workflow (ArcGIS):
  1. Huge annual GeoTIFFs (366 bands of daily ETo) → Zonal Statistics over
     vineyard polygons → aggregate to CA county averages.

New workflow (this script):
  1. Read DWR i15 Crop Mapping GDBs with geopandas (no ArcGIS required).
  2. Filter for vineyard polygons; compute area in California Albers (m²).
  3. Download Census TIGER ZCTA boundaries once and cache locally; spatial-
     join vineyard polygon centroids to assign each polygon a zip code.
  4. Build area-based weights: fraction of county vineyard area per zip code.
  5. Query the CIMIS Web API (daily ASCE ETo + precipitation by zip code).
  6. Sum daily ETo → monthly totals per zip code.
  7. Area-weighted mean of monthly ETo/precip → one row per county × month.
  8. Write CSVs: per-zip daily cache, county monthly, county annual.

Years supported: 2014, 2016, 2018-2024  (matches DWR crop mapping releases)

Requirements
------------
    pip install geopandas pyogrio fiona requests pandas numpy shapely

CIMIS API key (free)
--------------------
Register at https://cimis.water.ca.gov, log in, scroll to bottom of your
account page, and click "Get AppKey".  Then either:
  • Set the USER SETTINGS variable CIMIS_APP_KEY below, OR
  • Export it as an environment variable:  export CIMIS_APP_KEY="your-key"
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import box

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================
# USER SETTINGS — edit these before running
# ============================================================

# Folder containing all downloaded DWR .gdb files
GDB_FOLDER = Path(r"C:\Users\brian\ABT 182 Project\Grant\DWRraw\Data\onlyGDB")

# Where outputs (CSVs, ZCTA cache) are written
OUTPUT_DIR = Path(r"C:\Users\brian\ABT 182 Project\Output")

# CIMIS Web API key — register free at https://cimis.water.ca.gov
CIMIS_APP_KEY = os.getenv("CIMIS_APP_KEY", "YOUR-APP-KEY-HERE")

# Years to process
TARGET_YEARS = [2014, 2016, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

# Maximum zip codes kept per county per year, ranked by vineyard area.
# None = keep all (most accurate but more API calls).
# Set to e.g. 5 to drastically cut runtime while retaining major areas.
MAX_ZIPS_PER_COUNTY = None

# Seconds to pause between CIMIS API calls (avoid rate-limiting)
API_DELAY_SECONDS = 1.2


# ============================================================
# DWR GDB CONFIG — mirrors the ArcGIS extract_vineyards.py filters
# ============================================================

YEAR_CONFIG: dict[int, dict] = {
    2014: {
        "gdb":          "i15_Crop_Mapping_2014.gdb",
        "fc":           "i15_Crop_Mapping_2014",
        "filter_col":   "DWR_Standard_Legend",
        "filter_value": "V | VINEYARD",
        "filter_op":    "eq",          # exact match
        "county_col":   "COUNTY",
        "area_col":     "ACRES",       # fallback if Shape_Area unavailable
    },
    2016: {
        "gdb":          "i15_Crop_Mapping_2016.gdb",
        "fc":           "i15_Crop_Mapping_2016",
        "filter_col":   "CROPTYP2",
        "filter_value": "V",
        "filter_op":    "startswith",
        "county_col":   "COUNTY",
        "area_col":     "ACRES",
    },
    2018: {
        "gdb":          "i15_Crop_Mapping_2018.gdb",
        "fc":           "i15_Crop_Mapping_2018",
        "filter_col":   "CROPTYP2",
        "filter_value": "V",
        "filter_op":    "startswith",
        "county_col":   "COUNTY",
        "area_col":     "ACRES",
    },
    2019: {
        "gdb":          "i15_Crop_Mapping_2019.gdb",
        "fc":           "i15_Crop_Mapping_2019",
        "filter_col":   "CROPTYP2",
        "filter_value": "V",
        "filter_op":    "startswith",
        "county_col":   "COUNTY",
        "area_col":     "ACRES",
    },
    2020: {
        "gdb":          "i15_Crop_Mapping_2020.gdb",
        "fc":           "i15_Crop_Mapping_2020",
        "filter_col":   "CROPTYP2",
        "filter_value": "V",
        "filter_op":    "startswith",
        "county_col":   "COUNTY",
        "area_col":     "ACRES",
    },
    2021: {
        "gdb":          "i15_Crop_Mapping_2021.gdb",
        "fc":           "i15_Crop_Mapping_2021",
        "filter_col":   "CROPTYP2",
        "filter_value": "V",
        "filter_op":    "startswith",
        "county_col":   "COUNTY",
        "area_col":     "ACRES",
    },
    2022: {
        "gdb":          "i15_crop_mapping_2022.gdb",
        "fc":           "i15_Crop_Mapping_2022",
        "filter_col":   "CROPTYP2",
        "filter_value": "V",
        "filter_op":    "startswith",
        "county_col":   "COUNTY",
        "area_col":     "ACRES",
    },
    2023: {
        "gdb":          "i15_Crop_Mapping_2023_Provisional_20241127.gdb",
        "fc":           "i15_Crop_Mapping_2023_Provisional",
        "filter_col":   "CROPTYP2",
        "filter_value": "V",
        "filter_op":    "startswith",
        "county_col":   "COUNTY",
        "area_col":     "ACRES",
    },
    2024: {
        "gdb":          "i15_Crop_Mapping_2024_Provisional_20251208.gdb",
        "fc":           "i15_Crop_Mapping_2024_Provisional",
        "filter_col":   "CROPTYP2",
        "filter_value": "V",
        "filter_op":    "startswith",
        "county_col":   "COUNTY",
        "area_col":     "ACRES",
    },
}


# ============================================================
# STEP 1 — VINEYARD EXTRACTION
# ============================================================

def extract_vineyards(year: int) -> gpd.GeoDataFrame | None:
    """
    Read the DWR i15 GDB for *year* and return vineyard polygons as a
    GeoDataFrame in WGS-84 (EPSG:4326) with columns:
      county, area_m2, year, geometry
    Returns None if the GDB is missing or unreadable.
    """
    cfg = YEAR_CONFIG[year]
    gdb_path = GDB_FOLDER / cfg["gdb"]

    if not gdb_path.exists():
        log.warning(f"[{year}] GDB not found — skipping: {gdb_path}")
        return None

    log.info(f"[{year}] Reading layer '{cfg['fc']}' ...")
    try:
        gdf = gpd.read_file(str(gdb_path), layer=cfg["fc"], engine="pyogrio")
    except Exception as exc:
        log.error(f"[{year}] Failed to read GDB: {exc}")
        return None

    # --- Apply vineyard filter ---
    col = cfg["filter_col"]
    val = cfg["filter_value"]
    if col not in gdf.columns:
        log.warning(f"[{year}] Column '{col}' not found; available: {list(gdf.columns)}")
        return None

    if cfg["filter_op"] == "eq":
        mask = gdf[col] == val
    else:  # startswith
        mask = gdf[col].str.startswith(val, na=False)

    vineyards = gdf[mask].copy()
    if vineyards.empty:
        log.warning(f"[{year}] No vineyard polygons matched filter.")
        return None
    log.info(f"[{year}] {len(vineyards):,} vineyard polygons matched.")

    # --- Standardise columns ---
    county_col = cfg["county_col"]
    vineyards = vineyards[[county_col, "geometry"]].copy()
    vineyards = vineyards.rename(columns={county_col: "county"})
    vineyards["year"] = year

    # Reproject to California Albers for accurate area (m²)
    vineyards_albers = vineyards.to_crs("EPSG:3310")
    vineyards["area_m2"] = vineyards_albers.geometry.area

    # Final CRS: WGS-84 for spatial join with ZCTA
    vineyards = vineyards.to_crs("EPSG:4326")

    return vineyards[["county", "area_m2", "year", "geometry"]]


# ============================================================
# STEP 2 — ZIP CODE ASSIGNMENT VIA CENSUS ZCTA BOUNDARIES
# ============================================================

# California bounding box (WGS-84) — used to trim the national ZCTA file
_CA_BBOX = box(-124.6, 32.4, -113.9, 42.1)

def _get_zcta_cache_path() -> Path:
    return OUTPUT_DIR / "ca_zcta_cache.gpkg"


def get_ca_zcta() -> gpd.GeoDataFrame:
    """
    Return California ZCTA (zip-code tabulation area) boundaries.
    Downloads the 2023 Census TIGER file on first run (~75 MB) and caches
    the California subset as a GeoPackage for fast subsequent loads.
    """
    cache = _get_zcta_cache_path()
    if cache.exists():
        log.info(f"Loading cached ZCTA boundaries from {cache} ...")
        return gpd.read_file(str(cache))

    log.info("Downloading Census TIGER 2023 ZCTA boundaries (one-time, ~75 MB) ...")
    url = (
        "https://www2.census.gov/geo/tiger/TIGER2023/ZCTA520/"
        "tl_2023_us_zcta520.zip"
    )
    try:
        all_zcta = gpd.read_file(url, engine="pyogrio")
    except Exception as exc:
        raise RuntimeError(
            f"Could not download ZCTA file from Census. "
            f"Check your internet connection or download manually:\n{url}\n{exc}"
        ) from exc

    # Keep only zip codes within (or touching) California's bounding box
    all_zcta = all_zcta.to_crs("EPSG:4326")
    ca_zcta = all_zcta[all_zcta.intersects(_CA_BBOX)].copy()
    ca_zcta = ca_zcta[["ZCTA5CE20", "geometry"]].rename(
        columns={"ZCTA5CE20": "zip_code"}
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ca_zcta.to_file(str(cache), driver="GPKG")
    log.info(f"Cached {len(ca_zcta):,} California-area ZCTAs → {cache}")
    return ca_zcta


def assign_zip_codes(
    vineyards: gpd.GeoDataFrame,
    zcta: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Spatial-join each vineyard polygon centroid to a ZCTA boundary.
    Returns a DataFrame with columns: year, county, zip_code, area_m2.
    Polygons whose centroid falls outside any ZCTA are dropped with a warning.
    """
    centroids = vineyards.copy()
    centroids["geometry"] = vineyards.geometry.centroid

    joined = gpd.sjoin(centroids, zcta[["zip_code", "geometry"]], how="left", predicate="within")

    missing = joined["zip_code"].isna().sum()
    if missing:
        log.warning(
            f"{missing:,} vineyard centroids did not fall within any ZCTA and will be excluded. "
            "This usually happens near county borders — the count is typically small."
        )

    joined = joined.dropna(subset=["zip_code"])
    return joined[["year", "county", "zip_code", "area_m2"]].copy()


# ============================================================
# STEP 3 — CIMIS WEB API QUERIES
# ============================================================

_CIMIS_URL = "https://et.water.ca.gov/api/data"


def _fetch_daily_eto(
    zip_code: str,
    start_date: str,
    end_date: str,
    app_key: str,
    max_retries: int = 4,
) -> pd.DataFrame:
    """
    Single CIMIS API call for *one* zip code over a date range.
    Returns a DataFrame with columns: date (datetime), eto_in, precip_in.
    Returns an empty DataFrame on failure.
    """
    params = {
        "appKey":       app_key,
        "targets":      zip_code,
        "startDate":    start_date,
        "endDate":      end_date,
        "dataItems":    "day-asce-eto,day-precip",
        "prioritizeSCS": "Y",    # prefer Spatial CIMIS (full statewide coverage)
        "unitOfMeasure": "E",   # English units — inches
    }

    wait = 2.0
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(_CIMIS_URL, params=params, timeout=60)
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt == max_retries:
                log.error(f"    CIMIS API failed after {max_retries} retries for zip {zip_code}: {exc}")
                return pd.DataFrame()
            log.warning(f"    Retry {attempt + 1}/{max_retries} for zip {zip_code} (wait {wait:.0f}s) ...")
            time.sleep(wait)
            wait *= 2

    try:
        payload = resp.json()
    except ValueError:
        log.error(f"    Non-JSON response for zip {zip_code}")
        return pd.DataFrame()

    records = []
    try:
        providers = payload["Data"]["Providers"]
        for provider in providers:
            for rec in provider.get("Records", []):
                raw_date = rec.get("Date", "")
                if not raw_date:
                    continue

                def _parse(field: str) -> float | None:
                    v = rec.get(field, {})
                    val = v.get("Value") if isinstance(v, dict) else None
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        return None

                records.append({
                    "date":      pd.to_datetime(raw_date),
                    "eto_in":    _parse("DayAsceEto"),
                    "precip_in": _parse("DayPrecip"),
                })
    except (KeyError, TypeError) as exc:
        log.warning(f"    Unexpected CIMIS response structure for zip {zip_code}: {exc}")

    return pd.DataFrame(records)


def query_year_monthly(zip_code: str, year: int, app_key: str) -> pd.DataFrame:
    """
    Query CIMIS for a full calendar year and return *monthly* totals.
    Splits the year into two half-year requests to stay within API record limits.
    Returns DataFrame with columns: zip_code, year, month, eto_in, precip_in.
    """
    # Split into two half-year requests (≤183 daily records each)
    halves = [
        (f"{year}-01-01", f"{year}-06-30"),
        (f"{year}-07-01", f"{year}-12-31"),
    ]
    dfs = []
    for start, end in halves:
        df = _fetch_daily_eto(zip_code, start, end, app_key)
        if not df.empty:
            dfs.append(df)
        time.sleep(0.3)  # small pause between the two half-year calls

    if not dfs:
        return pd.DataFrame()

    daily = pd.concat(dfs, ignore_index=True)
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month

    monthly = (
        daily.groupby(["year", "month"])
        .agg(eto_in=("eto_in", "sum"), precip_in=("precip_in", "sum"))
        .reset_index()
    )
    monthly["zip_code"] = zip_code
    return monthly


# ============================================================
# STEP 4 — AREA-WEIGHTED COUNTY AGGREGATION
# ============================================================

def compute_weights(vine_zip: pd.DataFrame) -> pd.DataFrame:
    """
    For each (year, county, zip_code), calculate the fraction of that
    county's total vineyard area that falls within the zip code.
    Used as weights when averaging ETo across zip codes within a county.
    """
    county_total = (
        vine_zip.groupby(["year", "county"])["area_m2"]
        .sum()
        .rename("county_total_m2")
        .reset_index()
    )
    zip_area = (
        vine_zip.groupby(["year", "county", "zip_code"])["area_m2"]
        .sum()
        .rename("zip_area_m2")
        .reset_index()
    )
    w = zip_area.merge(county_total, on=["year", "county"])
    w["weight"] = w["zip_area_m2"] / w["county_total_m2"]
    return w


def aggregate_to_county(
    eto_by_zip: pd.DataFrame,
    weights: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine ETo data with area weights and compute:
      • county_monthly — county × year × month with area-weighted mean ETo & precip
      • county_annual  — county × year with annual totals

    Returns (county_monthly, county_annual).
    """
    merged = eto_by_zip.merge(
        weights[["year", "county", "zip_code", "weight"]],
        on=["year", "zip_code"],
    )
    merged["eto_wt"]    = merged["eto_in"]    * merged["weight"]
    merged["precip_wt"] = merged["precip_in"] * merged["weight"]

    county_monthly = (
        merged.groupby(["year", "county", "month"])
        .agg(
            eto_in    = ("eto_wt",    "sum"),
            precip_in = ("precip_wt", "sum"),
        )
        .reset_index()
        .sort_values(["county", "year", "month"])
    )

    county_annual = (
        county_monthly.groupby(["year", "county"])
        .agg(
            annual_eto_in    = ("eto_in",    "sum"),
            annual_precip_in = ("precip_in", "sum"),
        )
        .reset_index()
        .sort_values(["county", "year"])
    )
    # Simple aridity / water-demand proxy: ETo - precip
    county_annual["water_deficit_in"] = (
        county_annual["annual_eto_in"] - county_annual["annual_precip_in"]
    )

    return county_monthly, county_annual


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if CIMIS_APP_KEY == "YOUR-APP-KEY-HERE":
        log.error(
            "No CIMIS API key set.\n"
            "  1. Register free at https://cimis.water.ca.gov\n"
            "  2. Log in → account page → 'Get AppKey'\n"
            "  3. Set CIMIS_APP_KEY in this script or export it as an env variable."
        )
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Step 1 — Extract vineyard polygons from each year's DWR GDB         #
    # ------------------------------------------------------------------ #
    log.info("=" * 60)
    log.info("STEP 1 — Extracting vineyard polygons from DWR GDBs")
    log.info("=" * 60)

    all_gdfs: list[gpd.GeoDataFrame] = []
    for year in TARGET_YEARS:
        gdf = extract_vineyards(year)
        if gdf is not None:
            all_gdfs.append(gdf)

    if not all_gdfs:
        log.error("No vineyard data extracted. Verify GDB_FOLDER path.")
        return

    vineyards = gpd.GeoDataFrame(
        pd.concat(all_gdfs, ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )
    log.info(f"Total vineyard polygons across all years: {len(vineyards):,}")

    # ------------------------------------------------------------------ #
    # Step 2 — Assign zip codes via Census ZCTA spatial join              #
    # ------------------------------------------------------------------ #
    log.info("")
    log.info("=" * 60)
    log.info("STEP 2 — Assigning zip codes via Census ZCTA boundaries")
    log.info("=" * 60)

    zcta = get_ca_zcta()
    vine_zip = assign_zip_codes(vineyards, zcta)

    zip_table_path = OUTPUT_DIR / "vineyard_zip_area.csv"
    vine_zip.to_csv(zip_table_path, index=False)
    log.info(f"Vineyard–zip area table saved → {zip_table_path}")
    log.info(f"Unique zip codes with vineyards: {vine_zip['zip_code'].nunique():,}")

    # ------------------------------------------------------------------ #
    # Step 3 — Query CIMIS API                                            #
    # ------------------------------------------------------------------ #
    log.info("")
    log.info("=" * 60)
    log.info("STEP 3 — Querying CIMIS API for monthly ETo by zip code")
    log.info("=" * 60)

    weights = compute_weights(vine_zip)

    # Determine which (zip, year) pairs to query
    if MAX_ZIPS_PER_COUNTY is not None:
        # Keep only the top-N zip codes per county by vineyard area
        top = (
            vine_zip.groupby(["year", "county", "zip_code"])["area_m2"]
            .sum()
            .reset_index()
            .sort_values("area_m2", ascending=False)
            .groupby(["year", "county"])
            .head(MAX_ZIPS_PER_COUNTY)
        )
        query_pairs = top[["zip_code", "year"]].drop_duplicates()
        log.info(
            f"MAX_ZIPS_PER_COUNTY={MAX_ZIPS_PER_COUNTY}: "
            f"reduced to {len(query_pairs):,} zip×year queries."
        )
    else:
        query_pairs = vine_zip[["zip_code", "year"]].drop_duplicates()
        log.info(f"Total zip×year queries: {len(query_pairs):,}")

    # Check for a resume cache (useful if the run is interrupted)
    cache_path = OUTPUT_DIR / "cimis_daily_cache.csv"
    if cache_path.exists():
        log.info(f"Resuming from existing cache: {cache_path}")
        cached = pd.read_csv(cache_path)
        cached["date"] = pd.to_datetime(cached["date"])
        already_done = set(
            zip(cached["zip_code"].astype(str), cached["year"].astype(int))
        )
    else:
        cached = pd.DataFrame()
        already_done: set[tuple] = set()

    new_results: list[pd.DataFrame] = []
    remaining = query_pairs[
        ~query_pairs.apply(
            lambda r: (str(r["zip_code"]), int(r["year"])) in already_done, axis=1
        )
    ]
    log.info(f"Remaining queries (not yet cached): {len(remaining):,}")

    for i, (_, row) in enumerate(remaining.iterrows(), 1):
        zc, yr = str(row["zip_code"]), int(row["year"])
        log.info(f"  [{i}/{len(remaining)}] zip={zc}, year={yr}")
        monthly = query_year_monthly(zc, yr, CIMIS_APP_KEY)
        if not monthly.empty:
            new_results.append(monthly)
        # Periodically save cache so we can resume if interrupted
        if i % 20 == 0 and new_results:
            _save_cache(cache_path, cached, new_results)
        time.sleep(API_DELAY_SECONDS)

    # Final cache save
    all_eto = _save_cache(cache_path, cached, new_results)

    if all_eto.empty:
        log.error("No ETo data retrieved. Check your API key and zip codes.")
        return

    # ------------------------------------------------------------------ #
    # Step 4 — Aggregate to county × month                               #
    # ------------------------------------------------------------------ #
    log.info("")
    log.info("=" * 60)
    log.info("STEP 4 — Area-weighted aggregation to county level")
    log.info("=" * 60)

    county_monthly, county_annual = aggregate_to_county(all_eto, weights)

    monthly_path = OUTPUT_DIR / "county_vineyard_eto_monthly.csv"
    annual_path  = OUTPUT_DIR / "county_vineyard_eto_annual.csv"

    county_monthly.to_csv(monthly_path, index=False)
    county_annual.to_csv(annual_path, index=False)

    log.info(f"County monthly ETo → {monthly_path}")
    log.info(f"County annual  ETo → {annual_path}")

    # ------------------------------------------------------------------ #
    # Summary table                                                       #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("SUMMARY — Annual vineyard ETo by county (inches)")
    print("=" * 70)
    pivot = county_annual.pivot_table(
        index="county", columns="year", values="annual_eto_in"
    ).round(1)
    print(pivot.to_string())

    print("\n" + "=" * 70)
    print("SUMMARY — Annual water deficit (ETo − precip) by county (inches)")
    print("=" * 70)
    pivot_def = county_annual.pivot_table(
        index="county", columns="year", values="water_deficit_in"
    ).round(1)
    print(pivot_def.to_string())

    print(f"\nOutputs written to: {OUTPUT_DIR}")
    print("Done.")


def _save_cache(
    cache_path: Path,
    existing: pd.DataFrame,
    new_results: list[pd.DataFrame],
) -> pd.DataFrame:
    """Append new_results to the existing cache DataFrame, write to disk, return combined."""
    if not new_results:
        return existing if not existing.empty else pd.DataFrame()

    combined = pd.concat(
        [df for df in [existing] + new_results if not df.empty],
        ignore_index=True,
    )
    combined.to_csv(cache_path, index=False)
    log.info(f"  Cache updated ({len(combined):,} records) → {cache_path}")
    return combined


if __name__ == "__main__":
    main()
