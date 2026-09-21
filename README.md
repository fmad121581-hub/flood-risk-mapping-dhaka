# GIS-Based Flood Risk and Vulnerability Mapping
## Dhaka Metropolitan Region, Bangladesh

![Map 5 — Ward-Level Flood Risk](outputs/maps/map5_ward_risk.png)

---

## Overview

This project develops a comprehensive flood risk and vulnerability assessment for the **Dhaka Metropolitan Region** (960 km², ~12.5 million people) using the **UNDRR Risk = Hazard × Exposure × Vulnerability** framework.

Dhaka is one of the world's most flood-exposed megacities, situated on the Bengal Delta floodplain of the Buriganga, Turag, Balu, and Shitalakshya rivers. This project provides a spatially explicit, ward-level evidence base for flood risk reduction and urban planning.

---

## Key Findings

| Finding | Value |
|---|---|
| Mean annual rainfall (2004–2023) | 2,263 mm (CV = 31.8%) |
| Monsoon (JJAS) share of annual rainfall | 63% |
| May rainfall trend (significant) | +14.58 mm/yr (p = 0.008) |
| 10-year return period event | 705 mm peak monthly |
| Study area in High/Very High hazard | 67% |
| Population in Very High risk class | 84.9% (10.6 million people) |
| Highest risk ward | Ward No-63 (score 0.940) |
| Largest population at risk (single ward) | Dhania — 114,844 people |

---

## Project Structure

```
flood-risk-mapping-dhaka/
├── scripts/
│   ├── phase1_rainfall_analysis.py      # Rainfall statistics, Mann-Kendall, Gumbel
│   ├── phase2_flood_hazard.py           # FHI from DEM, TWI, slope, CN
│   ├── phase3_exposure_vulnerability.py # Risk index, ward-level zonal stats
│   ├── phase4_maps.py                   # Publication-quality cartographic maps
│   └── phase4_stats.py                  # Summary tables and report figures
├── outputs/
│   ├── maps/                            # 5 cartographic maps (PNG)
│   ├── figures/                         # 5 analytical figures (PNG)
│   └── tables/                          # 6 summary tables (CSV)
└── report/
    └── Flood_Risk_Report_Dhaka_FINAL.pdf  # Full written report
```

---

## Methodology

The analysis is structured across three phases following the UNDRR framework:

### Phase 1 — Rainfall Analysis
- **Data:** NASA POWER MERRA-2 monthly rainfall (lat 23.81°, lon 90.41°), 2004–2023
- **Methods:** Descriptive statistics, Mann-Kendall trend test, Sen's slope, Gumbel extreme value distribution
- **Output:** Return period estimates for 10-, 25-, and 50-year design events

### Phase 2 — Flood Hazard Mapping
- **Data:** SRTM 30m DEM, ESA WorldCover 2021
- **Methods:** TWI (GRASS GIS r.watershed), slope, SCS Curve Number, composite Flood Hazard Index
- **Weights:** TWI 35% | Elevation 30% | Slope 20% | CN 15%
- **Output:** 4-class flood hazard raster (Low / Medium / High / Very High)

### Phase 3 — Exposure, Vulnerability and Risk
- **Data:** WorldPop 2020, ESA WorldCover 2021, SRTM DEM, GADM v4.1 ward boundaries
- **Exposure:** Normalised population density (permanent water excluded)
- **Vulnerability:** Population density (40%) + built-up intensity (35%) + low elevation ≤4m (25%)
- **Risk:** Hazard × Exposure × Vulnerability, normalised 0–1
- **Classification:** Two-tier quantile (zero pixels = No Risk; non-zero at 25/50/75th percentile)
- **Output:** Ward-level risk ranking for 225 of 242 GADM Level 4 wards

---

## Maps and Figures

| | |
|---|---|
| ![Map 1](outputs/maps/map1_study_area.png) | ![Map 3](outputs/maps/map3_flood_hazard_class.png) |
| **Map 1** — Study area and ward boundaries | **Map 3** — Flood hazard classification |
| ![Map 4](outputs/maps/map4_risk_index.png) | ![Map 5](outputs/maps/map5_ward_risk.png) |
| **Map 4** — Composite risk index (continuous) | **Map 5** — Ward-level risk classification |

---

## Data Sources

| Dataset | Source | Resolution |
|---|---|---|
| SRTM 30m DEM | NASA/USGS | 30m spatial |
| ESA WorldCover 2021 | ESA Copernicus | 10m spatial |
| WorldPop 2020 | University of Southampton | ~92m |
| NASA POWER MERRA-2 | NASA Langley | 0.5°×0.625° |
| GADM v4.1 | GADM.org | Ward polygons |

---

## Tools and Libraries

| Category | Tools |
|---|---|
| GIS | QGIS 3.x, GRASS GIS (r.watershed, r.fill.dir) |
| Python | pandas, geopandas, rasterio, numpy, matplotlib, scipy |
| Remote sensing | ESA WorldCover 2021, SRTM, WorldPop |
| CRS | Storage: EPSG:4326 · Analysis: EPSG:32646 (UTM Zone 46N) |

---

## How to Run

### Requirements
```bash
pip install pandas geopandas rasterio numpy matplotlib scipy contextily matplotlib-scalebar
```
Before running any script, update the ROOT or BASE path at the top of each file to match your local project folder.
### Run order
```bash
python scripts/phase1_rainfall_analysis.py
python scripts/phase2_flood_hazard.py
python scripts/phase3_exposure_vulnerability.py
python scripts/phase4_maps.py
python scripts/phase4_stats.py
```

### Notes
- Set `ROOT` path at the top of each script to your local project folder
- Phase 2 DEM preprocessing (sink filling, TWI) must be done in QGIS/GRASS before running the Python script
- All rasters must be aligned to `data/output/phase2/dem_clipped.tif` reference grid
- DEM nodata value = −32767

---

## Author

**Fahim Ahmed**
Undergraduate, Department of Urban and Regional Planning
Bangladesh University of Engineering and Technology (BUET)

---

## References

- UNDRR (2015). Sendai Framework for Disaster Risk Reduction 2015–2030.
- Beven, K. J., & Kirkby, M. J. (1979). A physically based variable contributing area model. *Hydrological Sciences Bulletin*, 24(1), 43–69.
- ESA (2021). ESA WorldCover 10m 2021 v200.
- WorldPop (2020). Global High Resolution Population Denominators Project. University of Southampton.
- NASA POWER (2023). MERRA-2 Reanalysis. NASA Langley Research Center.
- USDA (1986). Technical Release 55: Urban Hydrology for Small Watersheds.

## Sensitivity Analysis & Validation (added)

Two additional analyses strengthen the FHI weighting and the ward risk ranking above:

- **`phase2b_sensitivity_analysis.py`** — tests the Flood Hazard Index against 7 alternative weighting scenarios plus a 120-draw Monte Carlo sweep. The ward ranking is stable under most re-weightings (Spearman ρ ≥ 0.93 vs baseline) but noticeably more sensitive to the Curve Number (land cover) weight specifically (ρ = 0.69, only 5/10 top wards retained). See `table6_sensitivity_scenarios.csv`, `table7_sensitivity_ward_rank.csv`, `table8_sensitivity_montecarlo_summary.csv`, `fig6_sensitivity_hazard_class.png`, `fig7_sensitivity_montecarlo.png`.

- **`phase3b_validation.py`** — cross-checks the ward risk ranking against an independent, peer-reviewed source: Alam, R., Quayyum, Z., Moulds, S., Radia, M.A., Sara, H.H., Hasan, M.T., & Butler, A. (2023). "Dhaka city water logging hazards: area identification and vulnerability assessment through GIS-remote sensing techniques." *Environmental Monitoring and Assessment*, 195(5), 543. https://doi.org/10.1007/s10661-023-11106-y. Wards that paper classifies as high vulnerability land at the 84th percentile of this model's own risk ranking on average; wards it classifies as very-low vulnerability land at the 36th — independent confirmation the ranking tracks something real. See `table9_validation_matches.csv`, `table10_validation_summary.csv`, `fig8_validation_percentiles.png`.

Full write-up, including a code audit of Phases 1–4 (two real bugs found and fixed — a stale hardcoded path and a NaN-handling bug that had been silently blanking the DEM/slope summary statistics) and a discussion of what "risk" vs "hazard" means for interpreting the ward ranking (see the Demra case), is in [`docs/methodology_addendum.md`](docs/methodology_addendum.md).
