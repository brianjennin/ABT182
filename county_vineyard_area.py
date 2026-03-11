"""
Aggregate vineyard area (acres) per county per year from vineyard_zip_area.csv.
Output: county_vineyard_area.csv  (year, county, vineyard_acres)
"""

import pandas as pd
from pathlib import Path

M2_PER_ACRE = 4_046.856_422

src = Path(__file__).parent / "vineyard_zip_area.csv"
out = Path(__file__).parent / "county_vineyard_area.csv"

df = pd.read_csv(src, dtype={"zip_code": str})

result = (
    df.groupby(["year", "county"], sort=True)["area_m2"]
    .sum()
    .div(M2_PER_ACRE)
    .round(2)
    .rename("vineyard_acres")
    .reset_index()
)

result.to_csv(out, index=False)
print(f"Wrote {len(result)} rows → {out}")
print(result.head(10).to_string(index=False))
