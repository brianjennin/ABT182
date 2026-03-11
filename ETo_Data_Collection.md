# ETo Data Collection

## Overview

Reference evapotranspiration (ETo) data for California vineyards was collected using the **CIMIS Web API** (California Irrigation Management Information System, https://cimis.water.ca.gov). The dataset uses ASCE-standardized reference ETo, an alfalfa-based measure reported in inches per day, and covers the years 2014, 2016, and 2018--2024 -- matching the release schedule of the Department of Water Resources (DWR) i15 Crop Mapping geodatabases.

## Data Collection Steps

1. **Vineyard identification** -- Vineyard polygons were extracted from DWR i15 Crop Mapping geodatabases for each target year. Polygons were filtered by crop-type codes (e.g., `CROPTYP2` starting with "V" for 2016 and later; `DWR_Standard_Legend = "V | VINEYARD"` for 2014).

2. **Zip-code assignment** -- Each vineyard polygon centroid was spatially joined to U.S. Census TIGER ZCTA (Zip Code Tabulation Area) boundaries for California to determine its zip code.

3. **CIMIS API queries** -- Daily ETo and precipitation values were retrieved from the CIMIS Web API by zip code, queried in batches of up to nine zip codes per request. Each year was split into two half-year windows (January--June and July--December) to stay within the API's 1,750-record limit per request. Daily values were summed to produce monthly totals.

4. **Area-weighted aggregation** -- Vineyard area within each zip code was used as a weight to aggregate monthly ETo and precipitation from the zip-code level up to county and American Viticultural Area (AVA) levels. For AVAs that overlap (e.g., nested appellations), vineyard centroids contribute independently to each enclosing AVA.

5. **Annual summaries** -- Monthly values were summed to annual totals. A water-deficit metric was calculated as the difference between annual ETo and annual precipitation.

## Outputs

| File | Description |
|------|-------------|
| `county_vineyard_eto_monthly.csv` | Monthly ETo and precipitation by county |
| `county_vineyard_eto_annual.csv` | Annual ETo, precipitation, and water deficit by county |
| `ava_vineyard_eto_monthly.csv` | Monthly ETo and precipitation by AVA |
| `ava_vineyard_eto_annual.csv` | Annual ETo, precipitation, and water deficit by AVA |
| `vineyard_water_stress.gpkg` | GeoPackage containing the above four tables joined to county/AVA geometries |

## Key Details

- **Units**: inches (daily values summed to monthly and annual totals)
- **ETo type**: ASCE standardized reference (alfalfa-based)
- **Precipitation source**: CIMIS daily precipitation (`day-precip`), queried alongside ETo
- **Geographic scope**: All California counties and AVAs containing mapped vineyards
- **Processing script**: `cimis_vineyard_eto.py`
