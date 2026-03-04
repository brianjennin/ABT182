"""
cimis_vineyard_eto.py
---------------------
Pure-Python replacement for the ArcGIS Pro + Spatial-CIMIS GeoTIFF workflow.

Original workflow (ArcGIS):
  1. Huge annual GeoTIFFs (366 bands of daily ETo) → Zonal Statistics over
     vineyard polygons → aggregate to CA county averages.

New workflow (this script):
  1. Read DWR i15 Crop Mapping GDBs with geopandas (no ArcGIS required).
  2. Filter for vineyard polygons; compute polygon area in California Albers.
  3. Download Census TIGER ZCTA boundaries once and cache locally; spatial-
     join vineyard polygon centroids to assign each polygon a zip code.
  4. Build area-based weights: fraction of county/AVA vineyard area per zip code.
  5. Query the CIMIS Web API (daily ASCE ETo + precip by zip code) and
     sum daily → monthly totals per zip code.
  6. Area-weighted mean of monthly ETo/precip → one value per county × month
     AND one value per American Viticultural Area (AVA) × month.
  7. Download CA county boundaries (Census TIGER) and load CA AVA boundaries
     from the bundled CA_avas.geojson; join ETo data to each.
  8. Write a GeoPackage with four ready-to-use ArcGIS Pro layers:
       • county_annual_eto  — one row per county × year
       • county_monthly_eto — one row per county × year × month
       • ava_annual_eto     — one row per AVA × year
       • ava_monthly_eto    — one row per AVA × year × month
     Also writes companion CSVs for non-spatial analysis.

  AVA aggregation note: AVAs have an internal geographic and viticultural logic
  (climate, soils, topography) that counties lack.  A vineyard centroid that
  falls within nested AVAs (e.g., Napa Valley ⊂ North Coast) contributes to
  BOTH — each AVA gets an independent area-weighted ET estimate.

Years supported: 2014, 2016, 2018–2024  (matches DWR crop mapping releases)

─────────────────────────────────────────────────────────────────────────────
HOW TO OPEN IN ARCGIS PRO
─────────────────────────────────────────────────────────────────────────────
1. Add to map
   Catalog pane → right-click vineyard_water_stress.gpkg → Add to Current Map
   → choose "county_annual_eto"

2. Enable time slider (year animation)
   Layer Properties → Time tab → check "Filter layer content based on
   attribute values" → Field: year → set Start / End years

3. Choropleth symbology
   Symbology pane → Primary symbology = Graduated Colors
   → Field: water_deficit_in  (annual ETo − precip, inches)
   → or choose annual_eto_in, annual_precip_in

4. For monthly breakdown
   Add "county_monthly_eto" layer; pivot chart or use the eto_mo01…eto_mo12
   fields as series in a chart, or join to the annual layer by county+year.

─────────────────────────────────────────────────────────────────────────────
REQUIREMENTS
─────────────────────────────────────────────────────────────────────────────
    pip install -r requirements.txt

CIMIS API key (free)
    Register at https://cimis.water.ca.gov → log in → account page →
    scroll to bottom → "Get AppKey".
    Then either set CIMIS_APP_KEY below or:  export CIMIS_APP_KEY="your-key"
"""

from __future__ import annotations

import logging
import os
import time
import warnings
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from curl_cffi import requests
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

# All outputs (GeoPackage, CSVs, boundary caches) go here
OUTPUT_DIR = Path(r"C:\Users\brian\ABT 182 Project\WaterStressOutput")

# California AVA boundaries — bundled GeoJSON in this repository
AVA_GEOJSON = Path(__file__).parent / "CA_avas.geojson"

# CIMIS Web API key — register free at https://cimis.water.ca.gov
CIMIS_APP_KEY = os.getenv("CIMIS_APP_KEY", "ef985ad9-17bc-4032-995c-1a6441c088c1")

# Years to process
TARGET_YEARS = [2014, 2016, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

# Maximum zip codes kept per county per year, ranked by vineyard area.
# None = keep all (most accurate but more API calls).
# Set e.g. 5 to cut runtime while still covering major vineyard areas.
MAX_ZIPS_PER_COUNTY = None

# Seconds to pause between CIMIS API calls (be polite / avoid rate-limiting)
API_DELAY_SECONDS = 2.0

# Number of API batches to run concurrently.
# Keep at 1 — CIMIS WAF blocks concurrent requests (each batch already fires
# 2 sub-requests for the two half-years, so 2 workers = 4 simultaneous hits).
CONCURRENT_BATCHES = 1


# ============================================================
# DWR GDB CONFIG — mirrors the ArcGIS extract_vineyards.py filters
# ============================================================

YEAR_CONFIG: dict[int, dict] = {
    2014: {
        "gdb":          "i15_Crop_Mapping_2014.gdb",
        "fc":           "i15_Crop_Mapping_2014",
        "filter_col":   "DWR_Standard_Legend",
        "filter_value": "V | VINEYARD",
        "filter_op":    "eq",
        "county_col":   "COUNTY",
    },
    2016: {
        "gdb":          "i15_Crop_Mapping_2016.gdb",
        "fc":           "i15_Crop_Mapping_2016",
        "filter_col":   "CROPTYP2",
        "filter_value": "V",
        "filter_op":    "startswith",
        "county_col":   "COUNTY",
    },
    2018: {
        "gdb":          "i15_Crop_Mapping_2018.gdb",
        "fc":           "i15_Crop_Mapping_2018",
        "filter_col":   "CROPTYP2",
        "filter_value": "V",
        "filter_op":    "startswith",
        "county_col":   "COUNTY",
    },
    2019: {
        "gdb":          "i15_Crop_Mapping_2019.gdb",
        "fc":           "i15_Crop_Mapping_2019",
        "filter_col":   "CROPTYP2",
        "filter_value": "V",
        "filter_op":    "startswith",
        "county_col":   "COUNTY",
    },
    2020: {
        "gdb":          "i15_Crop_Mapping_2020.gdb",
        "fc":           "i15_Crop_Mapping_2020",
        "filter_col":   "CROPTYP2",
        "filter_value": "V",
        "filter_op":    "startswith",
        "county_col":   "COUNTY",
    },
    2021: {
        "gdb":          "i15_Crop_Mapping_2021.gdb",
        "fc":           "i15_Crop_Mapping_2021",
        "filter_col":   "CROPTYP2",
        "filter_value": "V",
        "filter_op":    "startswith",
        "county_col":   "COUNTY",
    },
    2022: {
        "gdb":          "i15_crop_mapping_2022.gdb",
        "fc":           "i15_Crop_Mapping_2022",
        "filter_col":   "CROPTYP2",
        "filter_value": "V",
        "filter_op":    "startswith",
        "county_col":   "COUNTY",
    },
    2023: {
        "gdb":          "i15_Crop_Mapping_2023_Provisional_20241127.gdb",
        "fc":           "i15_Crop_Mapping_2023_Provisional",
        "filter_col":   "CROPTYP2",
        "filter_value": "V",
        "filter_op":    "startswith",
        "county_col":   "COUNTY",
    },
    2024: {
        "gdb":          "i15_Crop_Mapping_2024_Provisional_20251208.gdb",
        "fc":           "i15_Crop_Mapping_2024_Provisional",
        "filter_col":   "CROPTYP2",
        "filter_value": "V",
        "filter_op":    "startswith",
        "county_col":   "COUNTY",
    },
}


# ============================================================
# STEP 1 — VINEYARD EXTRACTION
# ============================================================

def extract_vineyards(year: int) -> gpd.GeoDataFrame | None:
    """
    Read the DWR i15 GDB for *year* and return vineyard polygons as a
    GeoDataFrame in WGS-84 with columns: county, area_m2, year, geometry.
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

    col = cfg["filter_col"]
    val = cfg["filter_value"]
    if col not in gdf.columns:
        log.warning(f"[{year}] Column '{col}' not found. Available: {list(gdf.columns)}")
        return None

    mask = (gdf[col] == val) if cfg["filter_op"] == "eq" else gdf[col].str.startswith(val, na=False)
    vineyards = gdf[mask].copy()

    if vineyards.empty:
        log.warning(f"[{year}] No vineyard polygons matched filter.")
        return None
    log.info(f"[{year}] {len(vineyards):,} vineyard polygons matched.")

    # Resolve county column — older GDBs use different names
    county_col = cfg["county_col"]
    if county_col not in vineyards.columns:
        _candidates = ["COUNTY", "County", "county", "CO_NAME", "COUNTY_NAME",
                       "CNTY", "COUNTYNAME", "COUNTY_CD", "CO_CNTY"]
        county_col = next((c for c in _candidates if c in vineyards.columns), None)
        if county_col:
            log.warning(
                f"[{year}] Configured county_col '{cfg['county_col']}' not found; "
                f"using '{county_col}' instead. Update YEAR_CONFIG to silence this."
            )
        else:
            log.warning(
                f"[{year}] No county column found. "
                f"Available columns: {list(vineyards.columns)}\n"
                f"       Update 'county_col' in YEAR_CONFIG for this year."
            )
            return None

    vineyards = vineyards[[county_col, "geometry"]].rename(columns={county_col: "county"})
    vineyards["year"] = year

    # Area in m² using California Albers projection
    vineyards["area_m2"] = vineyards.to_crs("EPSG:3310").geometry.area

    return vineyards.to_crs("EPSG:4326")[["county", "area_m2", "year", "geometry"]]


# ============================================================
# STEP 2 — ZIP CODE ASSIGNMENT VIA CENSUS ZCTA BOUNDARIES
# ============================================================

_CA_BBOX = box(-124.6, 32.4, -113.9, 42.1)  # California bounding box


def get_ca_zcta() -> gpd.GeoDataFrame:
    """
    Return California ZCTA (zip-code tabulation area) boundaries.
    Downloads from Census TIGER once (~75 MB) and caches as a GeoPackage.
    """
    cache = OUTPUT_DIR / "ca_zcta_cache.gpkg"
    if cache.exists():
        log.info("Loading cached ZCTA boundaries ...")
        return gpd.read_file(str(cache))

    log.info("Downloading Census TIGER ZCTA boundaries (one-time, ~75 MB) ...")
    url = "https://www2.census.gov/geo/tiger/TIGER2023/ZCTA520/tl_2023_us_zcta520.zip"
    try:
        all_zcta = gpd.read_file(url, engine="pyogrio").to_crs("EPSG:4326")
    except Exception as exc:
        raise RuntimeError(f"Could not download ZCTA file. Check your internet connection.\n{exc}") from exc

    ca_zcta = all_zcta[all_zcta.intersects(_CA_BBOX)][["ZCTA5CE20", "geometry"]].copy()
    ca_zcta = ca_zcta.rename(columns={"ZCTA5CE20": "zip_code"})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ca_zcta.to_file(str(cache), driver="GPKG")
    log.info(f"Cached {len(ca_zcta):,} California ZCTAs → {cache}")
    return ca_zcta


def assign_zip_codes(vineyards: gpd.GeoDataFrame, zcta: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Spatial-join each vineyard polygon centroid to a ZCTA boundary.
    Returns DataFrame with columns: year, county, zip_code, area_m2.
    """
    centroids = vineyards.copy()
    centroids["geometry"] = vineyards.geometry.centroid
    joined = gpd.sjoin(centroids, zcta[["zip_code", "geometry"]], how="left", predicate="within")
    missing = joined["zip_code"].isna().sum()
    if missing:
        log.warning(f"{missing:,} vineyard centroids fell outside any ZCTA and were excluded.")
    joined = joined.dropna(subset=["zip_code"])
    return joined[["year", "county", "zip_code", "area_m2", "_poly_id"]].copy()


# ============================================================
# STEP 2b — AVA BOUNDARY ASSIGNMENT
# ============================================================

def load_ca_avas() -> gpd.GeoDataFrame:
    """
    Load California AVA boundaries from the bundled CA_avas.geojson.
    Returns GeoDataFrame with columns: ava_id, ava_name, geometry (WGS-84).
    """
    if not AVA_GEOJSON.exists():
        raise FileNotFoundError(
            f"CA_avas.geojson not found at {AVA_GEOJSON}.\n"
            "Make sure you have cloned the full repository."
        )
    avas = gpd.read_file(str(AVA_GEOJSON)).to_crs("EPSG:4326")
    avas = avas[["ava_id", "name", "geometry"]].rename(columns={"name": "ava_name"})
    log.info(f"Loaded {len(avas):,} California AVAs from {AVA_GEOJSON.name}")
    return avas


def assign_avas(vineyards: gpd.GeoDataFrame, avas: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Spatial-join each vineyard polygon centroid to AVA boundaries.

    Because AVAs nest (e.g. Napa Valley ⊂ North Coast), a centroid can match
    multiple AVAs and will appear once per matching AVA.  Each AVA gets an
    independent water demand estimate from the vineyards inside it.

    Returns DataFrame with columns: _poly_id, ava_id, ava_name.
    """
    centroids = vineyards[["_poly_id", "geometry"]].copy()
    centroids["geometry"] = vineyards.geometry.centroid

    joined = gpd.sjoin(
        centroids,
        avas[["ava_id", "ava_name", "geometry"]],
        how="left",
        predicate="within",
    )

    outside = joined["ava_id"].isna().sum()
    if outside:
        log.info(
            f"  {outside:,} vineyard centroids outside all CA AVAs "
            "(expected for non-wine-region vineyards — excluded from AVA output)."
        )

    in_ava = joined.dropna(subset=["ava_id"])
    log.info(
        f"  {len(in_ava):,} vineyard-AVA assignments "
        f"({in_ava['ava_id'].nunique()} unique AVAs covered)."
    )
    return in_ava[["_poly_id", "ava_id", "ava_name"]].copy()


# ============================================================
# STEP 3 — CIMIS WEB API QUERIES
# ============================================================

_CIMIS_URL = "https://et.water.ca.gov/api/data"

# CIMIS hard limit is 1,750 records/request; each zip contributes ~182 records
# per half-year, so batch up to 9 zips per call (9 × 182 = 1,638 < 1,750).
_BATCH_SIZE = 9


def _fetch_daily_eto_batch(
    zip_codes: list[str], start_date: str, end_date: str, app_key: str, max_retries: int = 3
) -> pd.DataFrame:
    """
    Batch CIMIS API call for up to _BATCH_SIZE zip codes at once.
    Uses JSON format; returns daily DataFrame with columns:
    zip_code, date, eto_in, precip_in.
    """
    targets = ",".join(zip_codes)
    url = (
        f"{_CIMIS_URL}?appKey={app_key}"
        f"&targets={targets}"
        f"&startDate={start_date}"
        f"&endDate={end_date}"
        f"&dataItems=day-asce-eto,day-precip"
        f"&unitOfMeasure=E"
    )
    wait = 5.0
    resp = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, timeout=90, impersonate="chrome124")
            # WAF block — two known variants:
            #   1) 200 OK + body contains "Request Rejected"
            #   2) 200 OK + body is a generic HTML block page (<!DOCTYPE html …)
            body_start = resp.text[:500]
            if "Request Rejected" in body_start or body_start.lstrip().startswith(("<!DOCTYPE", "<html")):
                if attempt == max_retries:
                    log.error(f"    WAF blocked zips {targets[:60]!r} — skipping batch")
                    return pd.DataFrame()
                log.warning(f"    WAF rejection, retry {attempt+1}/{max_retries} (wait {wait:.0f}s) ...")
                time.sleep(wait)
                wait *= 3
                continue
            # 404 = no station/spatial data for these zips — permanent, skip immediately
            if resp.status_code == 404 or "404" in str(getattr(resp, 'status_code', '')):
                log.debug(f"    No CIMIS data for zips {targets[:60]!r} (404), skipping")
                return pd.DataFrame()
            # 403 = WAF / rate-limit — back off longer before retrying
            if resp.status_code == 403:
                if attempt == max_retries:
                    log.error(f"    403 Forbidden for zips {targets[:60]!r} — skipping batch")
                    return pd.DataFrame()
                log.warning(f"    403 rate-limit, retry {attempt+1}/{max_retries} (wait {wait:.0f}s) ...")
                time.sleep(wait)
                wait *= 3
                continue
            resp.raise_for_status()
            break
        except Exception as exc:
            if "404" in str(exc):
                log.debug(f"    No CIMIS data for zips {targets[:60]!r} (404), skipping")
                return pd.DataFrame()
            if attempt == max_retries:
                log.error(f"    CIMIS API failed for zips {targets[:60]!r}: {exc}")
                return pd.DataFrame()
            log.warning(f"    Retry {attempt+1}/{max_retries} (wait {wait:.0f}s): {exc}")
            time.sleep(wait)
            wait *= 3

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        log.error(f"    Unparseable XML for zips {targets[:60]!r}: {resp.text[:300]!r}")
        return pd.DataFrame()

    records = []
    for rec in root.findall(".//record"):
        raw_date = rec.get("date", "")
        zip_code = rec.get("zip-code", "").strip()
        if not raw_date or not zip_code:
            continue

        def _val(tag: str, _rec: ET.Element = rec) -> float | None:
            el = _rec.find(tag)
            if el is None or not (el.text or "").strip():
                return None
            try:
                return float(el.text.strip())
            except (TypeError, ValueError):
                return None

        records.append({
            "zip_code":  zip_code,
            "date":      pd.to_datetime(raw_date),
            "eto_in":    _val("day-asce-eto"),
            "precip_in": _val("day-precip"),
        })

    return pd.DataFrame(records)


def query_year_monthly_batch(zip_codes: list[str], year: int, app_key: str) -> pd.DataFrame:
    """
    Batch query for multiple zip codes for a full calendar year.
    Splits into two half-year requests to stay within the 1,750-record limit.
    Returns DataFrame with columns: zip_code, year, month, eto_in, precip_in.
    """
    halves = [(f"{year}-01-01", f"{year}-06-30"), (f"{year}-07-01", f"{year}-12-31")]
    dfs = []
    for start, end in halves:
        df = _fetch_daily_eto_batch(zip_codes, start, end, app_key)
        if not df.empty:
            dfs.append(df)
        time.sleep(0.2)

    if not dfs:
        return pd.DataFrame()

    daily = pd.concat(dfs, ignore_index=True)
    monthly = (
        daily.assign(year=daily["date"].dt.year, month=daily["date"].dt.month)
        .groupby(["zip_code", "year", "month"])
        .agg(eto_in=("eto_in", "sum"), precip_in=("precip_in", "sum"))
        .reset_index()
    )
    return monthly


# ============================================================
# STEP 4 — AREA-WEIGHTED COUNTY AGGREGATION
# ============================================================

def compute_weights(vine_zip: pd.DataFrame) -> pd.DataFrame:
    county_total = (
        vine_zip.groupby(["year", "county"])["area_m2"].sum().rename("county_total_m2").reset_index()
    )
    zip_area = (
        vine_zip.groupby(["year", "county", "zip_code"])["area_m2"].sum().rename("zip_area_m2").reset_index()
    )
    w = zip_area.merge(county_total, on=["year", "county"])
    w["weight"] = w["zip_area_m2"] / w["county_total_m2"]
    return w


def aggregate_to_county(
    eto_by_zip: pd.DataFrame, weights: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (county_monthly, county_annual) as plain DataFrames.
    county_annual includes water_deficit_in = annual_eto_in − annual_precip_in.
    """
    merged = eto_by_zip.merge(weights[["year", "county", "zip_code", "weight"]], on=["year", "zip_code"])
    merged["eto_wt"]    = merged["eto_in"]    * merged["weight"]
    merged["precip_wt"] = merged["precip_in"] * merged["weight"]

    county_monthly = (
        merged.groupby(["year", "county", "month"])
        .agg(eto_in=("eto_wt", "sum"), precip_in=("precip_wt", "sum"))
        .reset_index()
        .sort_values(["county", "year", "month"])
    )
    county_annual = (
        county_monthly.groupby(["year", "county"])
        .agg(annual_eto_in=("eto_in", "sum"), annual_precip_in=("precip_in", "sum"))
        .reset_index()
        .sort_values(["county", "year"])
    )
    county_annual["water_deficit_in"] = (
        county_annual["annual_eto_in"] - county_annual["annual_precip_in"]
    )
    return county_monthly, county_annual


# ============================================================
# STEP 4b — AREA-WEIGHTED AVA AGGREGATION
# ============================================================

def compute_ava_weights(vine_ava: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-zip-code weights within each AVA × year.
    weight = (vineyard area from zip X inside AVA) / (total vineyard area inside AVA)
    A zip code that appears in multiple AVAs gets a separate weight for each.
    """
    ava_total = (
        vine_ava.groupby(["year", "ava_id"])["area_m2"]
        .sum()
        .rename("ava_total_m2")
        .reset_index()
    )
    zip_area = (
        vine_ava.groupby(["year", "ava_id", "zip_code"])["area_m2"]
        .sum()
        .rename("zip_area_m2")
        .reset_index()
    )
    w = zip_area.merge(ava_total, on=["year", "ava_id"])
    w["weight"] = w["zip_area_m2"] / w["ava_total_m2"]
    return w


def aggregate_to_ava(
    eto_by_zip: pd.DataFrame,
    ava_weights: pd.DataFrame,
    vine_ava: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Area-weighted aggregation of zip-level ETo to AVA level.
    Returns (ava_monthly, ava_annual).
    ava_annual includes water_deficit_in = annual_eto_in − annual_precip_in.
    """
    ava_names = vine_ava[["ava_id", "ava_name"]].drop_duplicates()

    merged = eto_by_zip.merge(
        ava_weights[["year", "ava_id", "zip_code", "weight"]], on=["year", "zip_code"]
    )
    merged["eto_wt"]    = merged["eto_in"]    * merged["weight"]
    merged["precip_wt"] = merged["precip_in"] * merged["weight"]

    ava_monthly = (
        merged.groupby(["year", "ava_id", "month"])
        .agg(eto_in=("eto_wt", "sum"), precip_in=("precip_wt", "sum"))
        .reset_index()
        .merge(ava_names, on="ava_id")
        .sort_values(["ava_id", "year", "month"])
    )
    ava_annual = (
        ava_monthly.groupby(["year", "ava_id"])
        .agg(annual_eto_in=("eto_in", "sum"), annual_precip_in=("precip_in", "sum"))
        .reset_index()
        .merge(ava_names, on="ava_id")
        .sort_values(["ava_id", "year"])
    )
    ava_annual["water_deficit_in"] = (
        ava_annual["annual_eto_in"] - ava_annual["annual_precip_in"]
    )
    return ava_monthly, ava_annual


# ============================================================
# STEP 5 — COUNTY BOUNDARIES + GEOPACKAGE FOR ARCGIS PRO
# ============================================================

def get_ca_counties() -> gpd.GeoDataFrame:
    """
    Return California county boundaries (Census TIGER 2023).
    Downloads once (~7 MB) and caches locally.
    """
    cache = OUTPUT_DIR / "ca_counties_cache.gpkg"
    if cache.exists():
        log.info("Loading cached CA county boundaries ...")
        return gpd.read_file(str(cache))

    log.info("Downloading Census TIGER county boundaries (one-time, ~7 MB) ...")
    url = "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip"
    try:
        all_counties = gpd.read_file(url, engine="pyogrio")
    except Exception as exc:
        raise RuntimeError(f"Could not download county boundaries.\n{exc}") from exc

    # FIPS state code 06 = California
    ca = all_counties[all_counties["STATEFP"] == "06"][["NAME", "geometry"]].copy()
    ca = ca.rename(columns={"NAME": "county"}).to_crs("EPSG:4326")
    ca.to_file(str(cache), driver="GPKG")
    log.info(f"Cached {len(ca)} CA counties → {cache}")
    return ca


def _normalize_county(s: pd.Series) -> pd.Series:
    """Title-case and strip common suffixes so DWR names match Census names."""
    return s.str.strip().str.title().str.replace(r"\s+County$", "", regex=True)


def build_geopackage(
    county_monthly: pd.DataFrame,
    county_annual: pd.DataFrame,
    ca_counties: gpd.GeoDataFrame,
    output_path: Path,
    ava_monthly: pd.DataFrame | None = None,
    ava_annual: pd.DataFrame | None = None,
    ca_avas: gpd.GeoDataFrame | None = None,
) -> None:
    """
    Write four layers to a GeoPackage that are immediately usable in ArcGIS Pro.

    Layer 1 — county_annual_eto
        One row per county × year.
        Columns: county, year, annual_eto_in, annual_precip_in, water_deficit_in
        → Enable ArcGIS Pro time slider on the 'year' field.
        → Symbolize with Graduated Colors on water_deficit_in.

    Layer 2 — county_monthly_eto
        One row per county × year with monthly ETo spread into columns
        eto_mo01 … eto_mo12 (inches) and precip_mo01 … precip_mo12.
        → Use for bar/line charts per county, or Join Field workflows.

    Layer 3 — ava_annual_eto  (written when ava_annual and ca_avas are provided)
        One row per AVA × year.  Same fields as county_annual_eto.
        → Compare water stress across American Viticultural Areas.

    Layer 4 — ava_monthly_eto  (written when ava_monthly and ca_avas are provided)
        One row per AVA × year with eto_mo01 … eto_mo12 columns.
    """
    # Normalise county name for joining
    ca_counties = ca_counties.copy()
    ca_counties["county_key"] = _normalize_county(ca_counties["county"])

    # ── Layer 1: annual, time-aware ──────────────────────────────────────────
    annual = county_annual.copy()
    annual["county_key"] = _normalize_county(annual["county"])

    annual_geo = ca_counties.merge(annual, on="county_key", how="inner", suffixes=("_census", ""))
    annual_geo = annual_geo.drop(columns=["county_key", "county_census"], errors="ignore")

    # ArcGIS Pro time slider needs a proper date column; add Jan 1 of each year
    annual_geo["date"] = pd.to_datetime(annual_geo["year"].astype(str) + "-01-01")

    # Round numeric fields for tidiness
    for col in ["annual_eto_in", "annual_precip_in", "water_deficit_in"]:
        if col in annual_geo.columns:
            annual_geo[col] = annual_geo[col].round(2)

    # ── Layer 2: monthly pivot ───────────────────────────────────────────────
    monthly = county_monthly.copy()
    monthly["county_key"] = _normalize_county(monthly["county"])

    eto_wide = monthly.pivot_table(
        index=["year", "county_key"], columns="month", values="eto_in"
    ).reset_index()
    eto_wide.columns = [
        f"eto_mo{c:02d}" if isinstance(c, int) else c for c in eto_wide.columns
    ]

    precip_wide = monthly.pivot_table(
        index=["year", "county_key"], columns="month", values="precip_in"
    ).reset_index()
    precip_wide.columns = [
        f"pr_mo{c:02d}" if isinstance(c, int) else c for c in precip_wide.columns
    ]

    wide = eto_wide.merge(precip_wide, on=["year", "county_key"])
    wide["county_key"] = wide["county_key"].astype(str)

    monthly_geo = ca_counties.merge(wide, on="county_key", how="inner")
    monthly_geo = monthly_geo.drop(columns=["county_key"], errors="ignore")
    monthly_geo["date"] = pd.to_datetime(monthly_geo["year"].astype(str) + "-01-01")

    # Round
    num_cols = [c for c in monthly_geo.columns if c.startswith(("eto_mo", "pr_mo"))]
    monthly_geo[num_cols] = monthly_geo[num_cols].round(2)

    # ── Write GeoPackage ─────────────────────────────────────────────────────
    if output_path.exists():
        output_path.unlink()   # overwrite cleanly

    annual_geo.to_file(str(output_path), layer="county_annual_eto",  driver="GPKG")
    monthly_geo.to_file(str(output_path), layer="county_monthly_eto", driver="GPKG")

    log.info(f"GeoPackage written → {output_path}")
    log.info(f"  county_annual_eto  : {len(annual_geo):,} features  ({annual_geo['county'].nunique()} counties × {annual_geo['year'].nunique()} years)")
    log.info(f"  county_monthly_eto : {len(monthly_geo):,} features  (monthly ETo as eto_mo01…eto_mo12)")

    # ── Optional AVA layers ───────────────────────────────────────────────────
    if ava_annual is not None and ava_monthly is not None and ca_avas is not None:
        avas = ca_avas.copy()

        # ── AVA Layer 1: annual ───────────────────────────────────────────────
        ann_ava = ava_annual.copy()
        ava_annual_geo = avas.merge(ann_ava, on="ava_id", how="inner")
        ava_annual_geo["date"] = pd.to_datetime(ava_annual_geo["year"].astype(str) + "-01-01")
        for col in ["annual_eto_in", "annual_precip_in", "water_deficit_in"]:
            if col in ava_annual_geo.columns:
                ava_annual_geo[col] = ava_annual_geo[col].round(2)

        # ── AVA Layer 2: monthly pivot ────────────────────────────────────────
        mon_ava = ava_monthly.copy()
        ava_eto_wide = mon_ava.pivot_table(
            index=["year", "ava_id"], columns="month", values="eto_in"
        ).reset_index()
        ava_eto_wide.columns = [
            f"eto_mo{c:02d}" if isinstance(c, int) else c for c in ava_eto_wide.columns
        ]
        ava_precip_wide = mon_ava.pivot_table(
            index=["year", "ava_id"], columns="month", values="precip_in"
        ).reset_index()
        ava_precip_wide.columns = [
            f"pr_mo{c:02d}" if isinstance(c, int) else c for c in ava_precip_wide.columns
        ]
        ava_wide = ava_eto_wide.merge(ava_precip_wide, on=["year", "ava_id"])
        ava_names = ava_monthly[["ava_id", "ava_name"]].drop_duplicates()
        ava_wide = ava_wide.merge(ava_names, on="ava_id")

        ava_monthly_geo = avas.merge(ava_wide, on="ava_id", how="inner")
        ava_monthly_geo["date"] = pd.to_datetime(ava_monthly_geo["year"].astype(str) + "-01-01")
        num_cols = [c for c in ava_monthly_geo.columns if c.startswith(("eto_mo", "pr_mo"))]
        ava_monthly_geo[num_cols] = ava_monthly_geo[num_cols].round(2)

        ava_annual_geo.to_file(str(output_path),  layer="ava_annual_eto",  driver="GPKG", mode="a")
        ava_monthly_geo.to_file(str(output_path), layer="ava_monthly_eto", driver="GPKG", mode="a")

        log.info(f"  ava_annual_eto     : {len(ava_annual_geo):,} features  ({ava_annual_geo['ava_id'].nunique()} AVAs × {ava_annual_geo['year'].nunique()} years)")
        log.info(f"  ava_monthly_eto    : {len(ava_monthly_geo):,} features  (monthly ETo as eto_mo01…eto_mo12)")

    _print_arcgis_instructions(output_path)


def _print_arcgis_instructions(gpkg_path: Path) -> None:
    print()
    print("=" * 70)
    print("HOW TO USE IN ARCGIS PRO")
    print("=" * 70)
    print(f"GeoPackage: {gpkg_path}")
    print()
    print("1. ADD TO MAP")
    print("   Catalog pane → navigate to the .gpkg file → expand it →")
    print("   drag 'county_annual_eto' onto your map.")
    print()
    print("2. CHOROPLETH (graduated colors)")
    print("   Symbology pane → Primary symbology = Graduated Colors")
    print("   Field options:")
    print("     water_deficit_in  → annual ETo minus precip (proxy for water stress)")
    print("     annual_eto_in     → total reference ET demand")
    print("     annual_precip_in  → total precipitation")
    print()
    print("3. YEAR ANIMATION (time slider)")
    print("   Layer Properties → Time tab")
    print("   ✓ Filter layer content based on attribute values")
    print("   Start field: date   (or use 'year' as a range field)")
    print("   Step interval: 1 Year")
    print("   → Use the Time Slider toolbar to step through 2014–2024.")
    print()
    print("4. MONTHLY BREAKDOWN")
    print("   Add 'county_monthly_eto' layer.")
    print("   Fields eto_mo01 … eto_mo12 = monthly ETo (inches).")
    print("   Right-click layer → Create Chart → Bar Chart → select month fields.")
    print()
    print("5. AVA LAYERS")
    print("   'ava_annual_eto'  — same as county_annual_eto but by American")
    print("   Viticultural Area (AVA). Nested AVAs (e.g. Napa Valley ⊂ North")
    print("   Coast) each get their own independent ET estimate.")
    print("   'ava_monthly_eto' — monthly pivot by AVA.")
    print("   Symbolize on water_deficit_in or annual_eto_in; enable time slider.")
    print("=" * 70)


# ============================================================
# CACHE HELPER
# ============================================================

def _save_cache(cache_path: Path, existing: pd.DataFrame, new_results: list[pd.DataFrame]) -> pd.DataFrame:
    if not new_results:
        return existing if not existing.empty else pd.DataFrame()
    combined = pd.concat([df for df in [existing] + new_results if not df.empty], ignore_index=True)
    combined.to_csv(cache_path, index=False)
    log.info(f"  Cache updated ({len(combined):,} records) → {cache_path}")
    return combined


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if CIMIS_APP_KEY in ("YOUR-APP-KEY-HERE",):
        log.error(
            "No CIMIS API key set.\n"
            "  1. Register free at https://cimis.water.ca.gov\n"
            "  2. Log in → account page → 'Get AppKey'\n"
            "  3. Paste the key into CIMIS_APP_KEY above, or:\n"
            "     Windows: set CIMIS_APP_KEY=your-key\n"
            "     Mac/Linux: export CIMIS_APP_KEY=your-key"
        )
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _vine_zip_cache = OUTPUT_DIR / "vineyard_zip_area.csv"
    _vine_ava_cache = OUTPUT_DIR / "vineyard_ava_zip_area.csv"

    if _vine_zip_cache.exists() and _vine_ava_cache.exists():
        # ── Steps 1–2: Load from cache ────────────────────────────────────────
        log.info("Loading cached vineyard–zip and vineyard–AVA tables (delete CSVs to re-extract) ...")
        vine_zip = pd.read_csv(_vine_zip_cache, dtype={"zip_code": str})
        vine_ava = pd.read_csv(_vine_ava_cache, dtype={"zip_code": str})
        ca_avas  = load_ca_avas()
        log.info(f"vineyard_zip_area.csv      → {vine_zip.shape[0]:,} rows")
        log.info(f"vineyard_ava_zip_area.csv  → {vine_ava.shape[0]:,} rows  ({vine_ava['ava_id'].nunique()} AVAs)")
    else:
        # ── Step 1: Extract vineyard polygons ─────────────────────────────────
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

        vineyards = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), geometry="geometry", crs="EPSG:4326")
        vineyards["_poly_id"] = range(len(vineyards))
        log.info(f"Total vineyard polygons across all years: {len(vineyards):,}")

        # ── Step 2a: Assign zip codes ─────────────────────────────────────────
        log.info("")
        log.info("=" * 60)
        log.info("STEP 2a — Assigning zip codes via Census ZCTA boundaries")
        log.info("=" * 60)

        zcta = get_ca_zcta()
        vine_zip = assign_zip_codes(vineyards, zcta)
        log.info(f"Unique zip codes with vineyards: {vine_zip['zip_code'].nunique():,}")

        # ── Step 2b: Assign AVAs ──────────────────────────────────────────────
        log.info("")
        log.info("=" * 60)
        log.info("STEP 2b — Assigning AVAs from CA_avas.geojson")
        log.info("=" * 60)

        ca_avas = load_ca_avas()
        ava_assignments = assign_avas(vineyards, ca_avas)

        vine_ava = vine_zip.merge(ava_assignments, on="_poly_id", how="inner")
        vine_ava = vine_ava.drop(columns=["_poly_id"])

        vine_zip = vine_zip.drop(columns=["_poly_id"])
        vine_zip.to_csv(_vine_zip_cache, index=False)
        vine_ava.to_csv(_vine_ava_cache, index=False)
        log.info(f"vineyard_zip_area.csv      → {vine_zip.shape[0]:,} rows")
        log.info(f"vineyard_ava_zip_area.csv  → {vine_ava.shape[0]:,} rows  ({vine_ava['ava_id'].nunique()} AVAs)")

    # ── Step 3: Query CIMIS API ───────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("STEP 3 — Querying CIMIS API for monthly ETo by zip code")
    log.info("=" * 60)

    weights = compute_weights(vine_zip)

    if MAX_ZIPS_PER_COUNTY is not None:
        top = (
            vine_zip.groupby(["year", "county", "zip_code"])["area_m2"].sum().reset_index()
            .sort_values("area_m2", ascending=False)
            .groupby(["year", "county"]).head(MAX_ZIPS_PER_COUNTY)
        )
        query_pairs = top[["zip_code", "year"]].drop_duplicates()
        log.info(f"MAX_ZIPS_PER_COUNTY={MAX_ZIPS_PER_COUNTY} → {len(query_pairs):,} zip×year queries")
    else:
        query_pairs = vine_zip[["zip_code", "year"]].drop_duplicates()
        log.info(f"Total zip×year queries: {len(query_pairs):,}")

    cache_path = OUTPUT_DIR / "cimis_monthly_cache.csv"
    if cache_path.exists():
        log.info(f"Resuming from cache: {cache_path}")
        cached = pd.read_csv(cache_path)
        already_done = set(zip(cached["zip_code"].astype(str), cached["year"].astype(int)))
    else:
        cached = pd.DataFrame()
        already_done: set[tuple] = set()

    new_results: list[pd.DataFrame] = []
    remaining = query_pairs[
        ~query_pairs.apply(lambda r: (str(r["zip_code"]), int(r["year"])) in already_done, axis=1)
    ]
    log.info(f"Remaining queries (not yet cached): {len(remaining):,} zip×year pairs")

    # Group remaining work by year, then batch zip codes _BATCH_SIZE at a time.
    # This reduces API calls from N to ceil(N / _BATCH_SIZE) — roughly 9× fewer.
    year_to_zips: dict[int, list[str]] = (
        remaining.groupby("year")["zip_code"]
        .apply(lambda s: s.astype(str).unique().tolist())
        .to_dict()
    )

    # Flatten to a list of (batch, year) work items for parallel execution.
    work_items: list[tuple[list[str], int]] = []
    for yr, zips in sorted(year_to_zips.items()):
        for i in range(0, len(zips), _BATCH_SIZE):
            work_items.append((zips[i : i + _BATCH_SIZE], yr))
    total_batches = len(work_items)
    log.info(
        f"Batching into {total_batches} API calls "
        f"({_BATCH_SIZE} zips/call × 2 half-years, {CONCURRENT_BATCHES} concurrent workers)"
    )

    def _run_batch(args: tuple[list[str], int]) -> pd.DataFrame:
        batch, yr = args
        result = query_year_monthly_batch(batch, yr, CIMIS_APP_KEY)
        time.sleep(API_DELAY_SECONDS)
        return result

    batch_num = 0
    with ThreadPoolExecutor(max_workers=CONCURRENT_BATCHES) as executor:
        futures = {executor.submit(_run_batch, item): item for item in work_items}
        for future in as_completed(futures):
            batch_num += 1
            batch, yr = futures[future]
            monthly = future.result()
            log.info(
                f"  [{batch_num}/{total_batches}] year={yr}, "
                f"zips {','.join(batch[:4])}{'...' if len(batch) > 4 else ''}"
            )
            if not monthly.empty:
                new_results.append(monthly)
            if batch_num % 20 == 0 and new_results:
                _save_cache(cache_path, cached, new_results)

    all_eto = _save_cache(cache_path, cached, new_results)

    if all_eto.empty:
        log.error("No ETo data retrieved. Check your API key and zip codes.")
        return

    # ── Step 4a: County-level aggregation ────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("STEP 4a — Area-weighted aggregation to county level")
    log.info("=" * 60)

    county_monthly, county_annual = aggregate_to_county(all_eto, weights)
    county_monthly.to_csv(OUTPUT_DIR / "county_vineyard_eto_monthly.csv", index=False)
    county_annual.to_csv(OUTPUT_DIR / "county_vineyard_eto_annual.csv",  index=False)
    log.info(f"county CSVs written → {OUTPUT_DIR}")

    # ── Step 4b: AVA-level aggregation ───────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("STEP 4b — Area-weighted aggregation to AVA level")
    log.info("=" * 60)

    ava_weights = compute_ava_weights(vine_ava)
    ava_monthly, ava_annual = aggregate_to_ava(all_eto, ava_weights, vine_ava)
    ava_monthly.to_csv(OUTPUT_DIR / "ava_vineyard_eto_monthly.csv", index=False)
    ava_annual.to_csv(OUTPUT_DIR / "ava_vineyard_eto_annual.csv",   index=False)
    log.info(f"AVA CSVs written → {OUTPUT_DIR}")
    log.info(f"  {ava_annual['ava_id'].nunique()} AVAs × {ava_annual['year'].nunique()} years")

    # ── Step 5: Build GeoPackage for ArcGIS Pro ───────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("STEP 5 — Building GeoPackage for ArcGIS Pro")
    log.info("=" * 60)

    ca_counties = get_ca_counties()
    gpkg_path = OUTPUT_DIR / "vineyard_water_stress.gpkg"
    build_geopackage(
        county_monthly, county_annual, ca_counties, gpkg_path,
        ava_monthly=ava_monthly, ava_annual=ava_annual, ca_avas=ca_avas,
    )

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Annual water deficit by county (ETo − precip, inches) — top 10")
    print("=" * 70)
    pivot = county_annual.pivot_table(
        index="county", columns="year", values="water_deficit_in"
    ).round(1)
    print(pivot.nlargest(10, pivot.columns[-1]).to_string())

    print("\n" + "=" * 70)
    print("Annual water deficit by AVA (ETo − precip, inches) — top 10")
    print("=" * 70)
    ava_pivot = ava_annual.pivot_table(
        index="ava_name", columns="year", values="water_deficit_in"
    ).round(1)
    print(ava_pivot.nlargest(10, ava_pivot.columns[-1]).to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
