"""
extract_vineyards.py
--------------------
Extracts vineyard polygons from the California DWR PROVISIONAL Statewide Crop
Mapping Geodatabases (2014-2022) and saves a separate feature class per year
into a single output GDB — one layer per year, intended for use in an ESRI
app with a yearly time slider.

Output layers inside the GDB:
  Vineyards_2014, Vineyards_2016, Vineyards_2018, Vineyards_2019,
  Vineyards_2020, Vineyards_2021, Vineyards_2022

Source: California Open Data - PROVISIONAL - 20XX Statewide Crop Mapping Geodatabase
Available years with data: 2014, 2016, 2018, 2019, 2020, 2021, 2022
(2015 and 2017 were never released by DWR)

HOW TO RUN IN ARCGIS PRO:
  1. Open the Python window (Analysis tab > Python)
  2. Paste this entire script and press Enter
  OR
  3. Open in a Notebook (Insert tab > New Notebook)

BEFORE RUNNING - update the two paths below:
  INPUT_FOLDER : folder that contains all your downloaded .gdb files
  OUTPUT_GDB   : path where you want the output geodatabase created
"""

import arcpy
import os

# =============================================================================
# USER SETTINGS — EDIT THESE BEFORE RUNNING
# =============================================================================

# Folder that contains all the downloaded crop mapping .gdb files
INPUT_FOLDER = r"C:\Users\brian\ABT 182 Project\Grant\DWRraw\Data\onlyGDB"

# Full path for the output geodatabase (will be created if it doesn't exist)
OUTPUT_GDB = r"C:\Users\brian\ABT 182 Project\Grant\DWRraw\Data\output_gdb"

# =============================================================================
# CONFIGURATION — no need to edit below this line
# =============================================================================

# Years the DWR dataset actually covers (2015 and 2017 were not released)
TARGET_YEARS = [2014, 2016, 2018, 2019, 2020, 2021, 2022]

# Common crop-type field names used across DWR dataset versions
CROP_FIELD_CANDIDATES = [
    "CLASS2",       # most common — Level 2 crop classification
    "CROPTYP2",
    "Crop_Type",
    "CROP_TYPE",
    "LANDUSE",
    "LABEL",
    "DWR_Stnd_C",
]

# All text values that indicate a vineyard in the crop-type field
VINEYARD_VALUES = ["Vineyard", "VINEYARD", "vineyard", "Vineyards", "VINEYARDS"]

# Common geodatabase naming patterns used by DWR / CA Open Data downloads
GDB_NAME_PATTERNS = [
    "i15_Crop_Mapping_{year}.gdb",
    "Crop{year}.gdb",
    "Crop_Mapping_{year}.gdb",
    "Statewide_Crop_Mapping_{year}.gdb",
    "{year}_Crop_Mapping.gdb",
    "CA_Crop_Mapping_{year}.gdb",
    "DWR_Crop_Mapping_{year}.gdb",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_output_gdb(gdb_path):
    """Create the output file geodatabase if it does not already exist."""
    gdb_dir = os.path.dirname(gdb_path)
    gdb_name = os.path.basename(gdb_path)
    if not arcpy.Exists(gdb_path):
        arcpy.management.CreateFileGDB(gdb_dir, gdb_name)
        print(f"  Created output GDB: {gdb_path}")
    else:
        print(f"  Output GDB already exists: {gdb_path}")


def find_gdb_for_year(folder, year):
    """
    Search the input folder for a geodatabase matching the given year.
    Tries known naming patterns first, then falls back to scanning all .gdb
    directories in the folder for any that contain the year in their name.
    """
    # Try known patterns
    for pattern in GDB_NAME_PATTERNS:
        candidate = os.path.join(folder, pattern.format(year=year))
        if arcpy.Exists(candidate):
            return candidate

    # Fallback: scan folder for any .gdb whose name contains the year
    try:
        for entry in os.listdir(folder):
            if entry.endswith(".gdb") and str(year) in entry:
                candidate = os.path.join(folder, entry)
                if arcpy.Exists(candidate):
                    return candidate
    except Exception:
        pass

    return None


def find_polygon_feature_class(gdb_path):
    """
    Return the path to the main polygon feature class inside the GDB.
    Prefers a feature class whose name contains the year or 'crop'.
    """
    arcpy.env.workspace = gdb_path
    fcs = arcpy.ListFeatureClasses(feature_type="Polygon")

    if not fcs:
        # Try without type filter in case geometry type metadata is missing
        fcs = arcpy.ListFeatureClasses()

    if not fcs:
        return None

    # Prefer a feature class with 'crop' or 'land' in the name
    for fc in fcs:
        lower = fc.lower()
        if "crop" in lower or "land" in lower or "i15" in lower:
            return os.path.join(gdb_path, fc)

    # Default to first feature class found
    return os.path.join(gdb_path, fcs[0])


def find_crop_field(fc_path):
    """
    Find the field in the feature class that holds the crop-type classification.
    Returns the field name, or None if not found.
    """
    existing_fields = {f.name for f in arcpy.ListFields(fc_path)}

    for candidate in CROP_FIELD_CANDIDATES:
        if candidate in existing_fields:
            return candidate

    # Case-insensitive fallback
    lower_map = {f.lower(): f for f in existing_fields}
    for candidate in CROP_FIELD_CANDIDATES:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def list_unique_crop_values(fc_path, field_name, limit=30):
    """Print unique values in the crop field — useful for diagnosing field values."""
    values = set()
    with arcpy.da.SearchCursor(fc_path, [field_name]) as cursor:
        for row in cursor:
            if row[0] is not None:
                values.add(row[0])
            if len(values) >= limit:
                break
    print(f"    Sample unique values in '{field_name}': {sorted(values)[:limit]}")


def build_where_clause(field_name):
    """Build a SQL WHERE clause that matches any of the vineyard values."""
    quoted = [f"'{v}'" for v in VINEYARD_VALUES]
    return f"{field_name} IN ({', '.join(quoted)})"


def extract_vineyards_from_gdb(gdb_path, year, output_gdb):
    """
    Extract vineyard polygons from a single year's geodatabase and save the
    result as a feature class named Vineyards_<year> in the output GDB.
    Returns the number of features extracted, or -1 on failure.
    """
    print(f"\n  Finding feature class in {os.path.basename(gdb_path)} ...")
    fc_path = find_polygon_feature_class(gdb_path)
    if fc_path is None:
        print("  ERROR: No polygon feature class found — skipping.")
        return -1
    print(f"  Feature class: {os.path.basename(fc_path)}")

    crop_field = find_crop_field(fc_path)
    if crop_field is None:
        print("  ERROR: Could not identify the crop-type field.")
        print("  Available fields:", [f.name for f in arcpy.ListFields(fc_path)])
        return -1
    print(f"  Crop field detected: {crop_field}")

    # Uncomment the next line to inspect what values exist in your data:
    # list_unique_crop_values(fc_path, crop_field)

    where_clause = build_where_clause(crop_field)
    output_fc = os.path.join(output_gdb, f"Vineyards_{year}")

    # Overwrite if a previous run already created this layer
    if arcpy.Exists(output_fc):
        arcpy.management.Delete(output_fc)

    arcpy.analysis.Select(fc_path, output_fc, where_clause)

    count = int(arcpy.management.GetCount(output_fc).getOutput(0))
    print(f"  Extracted {count:,} vineyard polygons -> Vineyards_{year}")
    return count


# =============================================================================
# MAIN
# =============================================================================

def main():
    arcpy.env.overwriteOutput = True

    print("=" * 60)
    print("California DWR Vineyard Extraction")
    print(f"Input folder : {INPUT_FOLDER}")
    print(f"Output GDB   : {OUTPUT_GDB}")
    print("=" * 60)

    # Create the output geodatabase
    create_output_gdb(OUTPUT_GDB)

    results = {}

    for year in TARGET_YEARS:
        print(f"\n[{year}] Searching for geodatabase ...")
        gdb_path = find_gdb_for_year(INPUT_FOLDER, year)

        if gdb_path is None:
            print(f"  WARNING: No geodatabase found for {year} in {INPUT_FOLDER}")
            print(f"  Make sure the .gdb folder for {year} is downloaded there.")
            results[year] = "NOT FOUND"
            continue

        print(f"  Found: {os.path.basename(gdb_path)}")
        count = extract_vineyards_from_gdb(gdb_path, year, OUTPUT_GDB)
        results[year] = count if count >= 0 else "ERROR"

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for year, result in results.items():
        if isinstance(result, int) and result >= 0:
            status = f"{result:,} polygons extracted"
        else:
            status = str(result)
        print(f"  {year}: {status}")

    print(f"\nOutput GDB : {OUTPUT_GDB}")
    print("Layers     : Vineyards_<year> — one per year, ready to add to ArcGIS Online")
    print("Done.")


if __name__ == "__main__":
    main()

# When running directly in the ArcGIS Python window (not as __main__), call:
main()
