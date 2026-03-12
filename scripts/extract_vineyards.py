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
OUTPUT_GDB = r"C:\Users\brian\ABT 182 Project\Grant\DWRraw\Data\Vineyards_By_Year.gdb"

# =============================================================================
# CONFIGURATION — no need to edit below this line
# =============================================================================

# Exact GDB filename and feature class name for each year.
# Note: 2022 GDB has a lowercase 'c' in its filename.
# Note: 2014 uses a different field (DWR_Standard_Legend) than all other years.
YEAR_CONFIG = {
    2014: {
        "gdb":   "i15_Crop_Mapping_2014.gdb",
        "fc":    "i15_Crop_Mapping_2014",
        "where": "DWR_Standard_Legend = 'V | VINEYARD'",
    },
    2016: {
        "gdb":   "i15_Crop_Mapping_2016.gdb",
        "fc":    "i15_Crop_Mapping_2016",
        "where": "CROPTYP2 LIKE 'V%'",
    },
    2018: {
        "gdb":   "i15_Crop_Mapping_2018.gdb",
        "fc":    "i15_Crop_Mapping_2018",
        "where": "CROPTYP2 LIKE 'V%'",
    },
    2019: {
        "gdb":   "i15_Crop_Mapping_2019.gdb",
        "fc":    "i15_Crop_Mapping_2019",
        "where": "CROPTYP2 LIKE 'V%'",
    },
    2020: {
        "gdb":   "i15_Crop_Mapping_2020.gdb",
        "fc":    "i15_Crop_Mapping_2020",
        "where": "CROPTYP2 LIKE 'V%'",
    },
    2021: {
        "gdb":   "i15_Crop_Mapping_2021.gdb",
        "fc":    "i15_Crop_Mapping_2021",
        "where": "CROPTYP2 LIKE 'V%'",
    },
    2022: {
        "gdb":   "i15_crop_mapping_2022.gdb",
        "fc":    "i15_Crop_Mapping_2022",
        "where": "CROPTYP2 LIKE 'V%'",
    },
    2023: {
        "gdb":   "i15_Crop_Mapping_2023_Provisional_20241127.gdb",
        "fc":    "i15_Crop_Mapping_2023_Provisional",
        "where": "CROPTYP2 LIKE 'V%'",
    },
    2024: {
        "gdb":   "i15_Crop_Mapping_2024_Provisional_20251208.gdb",
        "fc":    "i15_Crop_Mapping_2024_Provisional",
        "where": "CROPTYP2 LIKE 'V%'",
    },
}

TARGET_YEARS = sorted(YEAR_CONFIG.keys())


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


def extract_vineyards_from_gdb(year, output_gdb):
    """
    Extract vineyard polygons for the given year and save the result
    as Vineyards_<year> in the output GDB.
    Returns the feature count, or -1 on failure.
    """
    cfg = YEAR_CONFIG[year]
    gdb_path = os.path.join(INPUT_FOLDER, cfg["gdb"])
    fc_path  = os.path.join(gdb_path, cfg["fc"])
    where    = cfg["where"]

    if not arcpy.Exists(gdb_path):
        print(f"  WARNING: GDB not found: {cfg['gdb']}")
        return -1

    if not arcpy.Exists(fc_path):
        print(f"  WARNING: Feature class not found: {cfg['fc']}")
        return -1

    print(f"  Feature class : {cfg['fc']}")
    print(f"  WHERE clause  : {where}")

    output_fc = os.path.join(output_gdb, f"Vineyards_{year}")
    if arcpy.Exists(output_fc):
        arcpy.management.Delete(output_fc)

    arcpy.analysis.Select(fc_path, output_fc, where)

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
        print(f"\n[{year}]")
        count = extract_vineyards_from_gdb(year, OUTPUT_GDB)
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
