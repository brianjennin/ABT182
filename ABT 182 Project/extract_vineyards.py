"""
extract_vineyards.py
--------------------
Extracts vineyard polygons from the California DWR i15 Statewide Crop
Mapping Geodatabases and saves a separate feature class per year into a
single output GDB — one layer per year, intended for use in an ESRI app
with a yearly time slider.

Output layers inside the GDB:
  Vineyards_2014, Vineyards_2016, Vineyards_2018, Vineyards_2019,
  Vineyards_2020, Vineyards_2021, Vineyards_2022, Vineyards_2023,
  Vineyards_2024

GDB naming expected in INPUT_FOLDER:
  i15_Crop_Mapping_2014.gdb  (and same pattern for 2016-2022)
  i15_Crop_Mapping_2023_Provisional_20241127.gdb
  i15_Crop_Mapping_2024_Provisional_20251208.gdb

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

# Value stored in the CLASS fields for vineyard polygons
VINEYARD_CODE = "Vineyards"

# Years and their exact GDB filenames
GDB_NAMES = {
    2014: "i15_Crop_Mapping_2014.gdb",
    2016: "i15_Crop_Mapping_2016.gdb",
    2018: "i15_Crop_Mapping_2018.gdb",
    2019: "i15_Crop_Mapping_2019.gdb",
    2020: "i15_Crop_Mapping_2020.gdb",
    2021: "i15_Crop_Mapping_2021.gdb",
    2022: "i15_Crop_Mapping_2022.gdb",
    2023: "i15_Crop_Mapping_2023_Provisional_20241127.gdb",
    2024: "i15_Crop_Mapping_2024_Provisional_20251208.gdb",
}

TARGET_YEARS = sorted(GDB_NAMES.keys())


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
    """Return the full path to the GDB for the given year, or None if not found."""
    name = GDB_NAMES[year]
    candidate = os.path.join(folder, name)
    if arcpy.Exists(candidate):
        return candidate
    return None


def find_polygon_feature_class(gdb_path):
    """
    Return the path to the main polygon feature class inside the GDB.
    Prefers a feature class whose name contains 'i15', 'crop', or 'land'.
    """
    arcpy.env.workspace = gdb_path
    fcs = arcpy.ListFeatureClasses(feature_type="Polygon")

    if not fcs:
        fcs = arcpy.ListFeatureClasses()

    if not fcs:
        return None

    for fc in fcs:
        lower = fc.lower()
        if "i15" in lower or "crop" in lower or "land" in lower:
            return os.path.join(gdb_path, fc)

    return os.path.join(gdb_path, fcs[0])


def build_where_clause():
    """
    Build a WHERE clause that selects any polygon where 'Vineyards' appears
    in any crop position (CLASS1–CLASS4).

    Single-crop vineyard fields have the value in CLASS2; multi-crop fields
    may have it in CLASS1, CLASS3, or CLASS4.
    """
    code = f"'{VINEYARD_CODE}'"
    return (
        f"CLASS1 = {code} OR CLASS2 = {code} OR "
        f"CLASS3 = {code} OR CLASS4 = {code}"
    )


def extract_vineyards_from_gdb(gdb_path, year, output_gdb):
    """
    Extract vineyard polygons from a single year's GDB and save the result
    as Vineyards_<year> in the output GDB.
    Returns the feature count, or -1 on failure.
    """
    print(f"\n  Locating feature class in {os.path.basename(gdb_path)} ...")
    fc_path = find_polygon_feature_class(gdb_path)
    if fc_path is None:
        print("  ERROR: No polygon feature class found — skipping.")
        return -1
    print(f"  Feature class: {os.path.basename(fc_path)}")

    where_clause = build_where_clause()
    output_fc = os.path.join(output_gdb, f"Vineyards_{year}")

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

    create_output_gdb(OUTPUT_GDB)

    results = {}

    for year in TARGET_YEARS:
        print(f"\n[{year}] Searching for geodatabase ...")
        gdb_path = find_gdb_for_year(INPUT_FOLDER, year)

        if gdb_path is None:
            print(f"  WARNING: {GDB_NAMES[year]} not found in {INPUT_FOLDER}")
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
