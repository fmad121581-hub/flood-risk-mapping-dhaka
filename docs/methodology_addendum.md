# Methodology Addendum: Code Audit, Sensitivity Analysis & Validation

**Project:** GIS-Based Flood Risk & Vulnerability Mapping, Dhaka
**Scope of this addendum:** Phase 2 (Flood Hazard), Phase 2b (new — weight sensitivity), Phase 3 (Exposure/Vulnerability/Risk), Phase 3b (new — external validation)

This addendum documents three things added to strengthen the analysis: a correctness audit of the existing pipeline, a sensitivity analysis on the Flood Hazard Index (FHI) weights, and an external validation of the ward risk ranking against independent published research. All scripts referenced below are in `scripts/` and all outputs in `data/output/`.

---

## 1. Code Audit

Before adding new analysis, the existing Phase 1–4 scripts were reviewed line by line and run end to end to confirm every reported number is actually correct. Two real defects were found and fixed.

### 1.1 Hardcoded path pointed at a folder that no longer exists
`phase1_rainfall_analysis.py`, `phase3_exposure_vulnerability.py`, and `phase4_maps.py` all hardcoded `ROOT`/`BASE` as `C:/Users/user/OneDrive/Must`, which doesn't exist on this machine (the project lives under `OneDrive/Projects/Flood_risk`). This didn't corrupt any results — outputs already on disk were presumably produced by locally editing the path before each run — but it meant the scripts as saved would fail for anyone (including a future you) trying to reproduce the pipeline from scratch. All three are now fixed to the correct path; `phase2_flood_hazard.py` and `phase2b`/`phase3b` (new) instead derive `ROOT` dynamically from the script's own location, which is portable across machines and doesn't need editing.

### 1.2 Corrupted DEM elevation and slope statistics in `phase2_summary_stats.csv`
This one did corrupt output. `phase2_flood_hazard.py` converts DEM/slope nodata cells to `NaN`, but then filtered "valid" pixels by comparing against the *old* sentinel value (`-32767`) instead of checking for `NaN`. Since `NaN != -32767` evaluates to `True`, every nodata cell was incorrectly counted as valid, and because `NaN` propagates through `.mean()`/`.min()`/`.max()`, the result was silently blank output: `phase2_summary_stats.csv` reported empty values for DEM mean/min/max elevation and mean slope. Verified by re-running before and after the fix — before: blank; after: `DEM mean elevation = 8.98 m`, `min = -37.0 m`, `max = 49.0 m`, `Mean slope = 2.24°`, all consistent with the DEM read-in log.

A second, related defect was found in the same area: `fill_sinks_simple()`, the function meant to produce a hydrologically-corrected `dem_filled.tif`, never actually fills anything. It initializes the whole surface to the maximum elevation with no boundary condition for water to drain toward, so the relaxation loop terminates after one pass having changed nothing (confirmed: DEM elevation stats came out as `mean == min == max`, a flat plane). A boundary-seeding fix was attempted (seeding the domain edge with real elevation, per the standard Planchon–Darboux approach), which improved convergence from 1 to 131 iterations, but the relaxation still collapses toward a near-uniform surface on this irregularly-clipped domain, because `numpy.roll`'s wraparound behaviour links unrelated far edges of the array as if they were neighbours. Properly fixing this needs a priority-queue-based fill (e.g. Barnes et al. 2014), which is a larger rewrite than this pass covers.

**Practical impact:** none on the actual hazard/risk results. `dem_filled` is saved as `dem_filled.tif` but is not used anywhere else in the pipeline — slope, TWI, and elevation classification all correctly use the raw DEM (`dem_raw`), not the filled one. The fix applied: elevation/slope summary statistics and the Fig 6 elevation-vs-TWI scatter plot now read from `dem_raw`, and the script prints an explicit warning if `dem_filled.tif` looks degenerate, so nobody downstream mistakes it for validated output.

### 1.3 What was *not* a bug
The hazard/risk classification, the composite FHI formula, the two-tier risk classification in Phase 3, and the ward zonal statistics were all checked and are computing correctly as written — the `elev_score` invert-logic comment that looks alarming in `phase2_flood_hazard.py` (lines ~599–605) is a leftover debugging comment; the code beneath it already applies the correction. Phase 1 and Phase 3 had no output-corrupting bugs, only the stale path noted above.

---

## 2. Sensitivity Analysis on FHI Weights (`phase2b_sensitivity_analysis.py`)

**Why:** The FHI weights (TWI 0.35 / Elevation 0.30 / Slope 0.20 / CN 0.15) come from literature (Tran et al. 2008; Khosravi et al. 2016; Bhuiyan & Dutta 2012), not from fitting or calibrating against Dhaka-specific data. A reviewer's first question is reasonably "how much does the answer change if those weights were slightly different?"

**Method:**
- **7 named scenarios**: the baseline plus TWI-heavy, elevation-heavy, slope-heavy, CN/urban-heavy, equal weights, and the no-LULC fallback weighting already defined in `phase2_flood_hazard.py` itself.
- **120-draw Monte Carlo**: random weight combinations from a Dirichlet distribution centred on the baseline weights, so draws are "plausible" alternatives rather than arbitrary extremes.
- For each scenario/draw: recomputed the hazard class map, its class-area percentages, and (via zonal statistics) the ward-level mean FHI, then compared to baseline with Spearman rank correlation and top-10-ward overlap.

**Results** (full tables in `data/output/phase2b_sensitivity/`):

| Scenario | % Very High hazard area | Spearman ρ vs baseline (ward ranking) | Top-10 wards retained |
|---|---|---|---|
| Baseline | 27.2% | 1.00 | 10/10 |
| TWI-heavy | 30.5% | 0.99 | 9/10 |
| Elevation-heavy | 23.4% | 0.98 | 9/10 |
| Slope-heavy | 30.0% | 0.99 | 9/10 |
| Equal weights | 31.3% | 0.93 | 8/10 |
| No-LULC fallback | 30.1% | 0.93 | 7/10 |
| **CN/urban-heavy** | 28.7% | **0.69** | **5/10** |

Monte Carlo (120 draws): "Very High" hazard area ranges 21.4%–39.7% (mean 27.9%, std 3.6 pp) and ward-ranking correlation vs baseline averages 0.89 (worst single draw: 0.33).

**Interpretation:** the hazard classification and ward ranking are reasonably robust to plausible re-weighting of TWI, elevation, and slope — the ward identified as highest-risk stays highest-risk under almost every scenario. The ranking is noticeably more sensitive to how much weight the Curve Number (land cover / imperviousness) term gets: the CN-heavy scenario only retains half of the baseline's top-10 wards. This makes physical sense — CN captures a fundamentally different driver (surface imperviousness) than the three terrain-based factors, which are more correlated with each other — but it means the CN weight (currently 0.15, literature-derived) is the single most consequential modelling choice and deserves the most scrutiny or local calibration if this work continues.

---

## 3. External Validation (`phase3b_validation.py`)

**Why:** Nothing in the original pipeline checked the Phase 3 ward risk ranking against any real-world flood record. `phase2_flood_hazard.py` did carry an informal QGIS checklist comment suggesting a visual comparison against "Demra, Amin Bazar, Mirpur, Khilkhet, Rayer Bazar" as known flood-prone areas, but that list wasn't cited and was never actually checked in code.

**Method:** Cross-referenced `ward_risk_ranked.csv` against an independent, peer-reviewed source:

> Alam, R., Quayyum, Z., Moulds, S., Radia, M.A., Sara, H.H., Hasan, M.T., & Butler, A. (2023). "Dhaka city water logging hazards: area identification and vulnerability assessment through GIS-remote sensing techniques." *Environmental Monitoring and Assessment*, 195(5), 543. https://doi.org/10.1007/s10661-023-11106-y

That paper independently classifies specific DNCC/DSCC wards and peripheral unions by water-logging vulnerability, using its own GIS/remote-sensing method — not this project's data, weights, or code. For every ward/union it names, this project's risk percentile (100 = highest modelled risk of 226 wards) was looked up.

**Results** (`data/output/phase3b_validation/`):

| Reference group (Alam et al. 2023) | n matched | Mean risk percentile (this model) |
|---|---|---|
| High vulnerability (Wards 2, 5, 7, 14) | 5 | **84.4** |
| Low vulnerability (Wards 8, 9, 33) | 3 | 53.9 |
| Low vulnerability (peripheral unions) | 14 | 51.5 |
| Very low vulnerability (peripheral unions) | 5 | **36.4** |

**Interpretation:** the model correctly separates the paper's "high vulnerability" wards (mean 84th percentile — near the top of the modelled risk ranking) from its "very low vulnerability" unions (mean 36th percentile). This is a genuine, independent signal that the risk ranking is picking up something real, not an artifact of the modelling choices. The separation between the two "low" and "very low" groups is weaker than ideal (51.5 vs 36.4 rather than both near 0), which is worth stating plainly rather than smoothing over — it suggests the model's mid-to-low risk tier is less discriminating than its top tier.

**A specific, important nuance — Demra:** `phase2_flood_hazard.py`'s own QGIS checklist named Demra as an area to expect high hazard, and this project's Phase 2 hazard layer does classify it High (mean hazard 3.17/4). But Demra ranks near the *bottom* of the final Phase 3 *risk* ranking (36th of 226, i.e. the 84th-lowest). This is not a contradiction once the model's structure is understood: Risk = Hazard × Exposure × Vulnerability is deliberately a *current, population-weighted* measure (see the two-tier classification rationale already documented in `phase3_exposure_vulnerability.py`), and Demra's measured population density (WorldPop 2020) is comparatively low. Alam et al. (2023) independently corroborates this — they also classify Demra as "very low vulnerability" overall, despite the physical flood hazard there being real. **This distinction — hazard is a property of the land, risk is hazard combined with who is currently exposed to it — should be stated explicitly wherever the ward ranking is presented (report, poster, or otherwise), since "low risk rank" is easy to misread as "not flood-prone."**

---

## 4. Limitations (for the report's limitations section)

- **FHI weights are literature-derived, not locally calibrated.** Section 2 shows the ward ranking is moderately sensitive to the CN weight specifically; a local calibration (e.g. against Section 3's validation source, or historical waterlogging complaint records if available from DNCC/DSCC) would strengthen this further.
- **TWI is not computed by this pipeline** — `phase2_flood_hazard.py` reads a pre-computed `twi_aligned.tif` (produced externally, presumably in QGIS/SAGA) rather than deriving it from the DEM in-script. The `compute_twi()` function in the script is dead code. This should be noted so a reader doesn't assume the whole pipeline is self-contained Python.
- **`dem_filled.tif` is not reliable** (Section 1.2) and should not be used or cited as a sink-filled product until replaced with GRASS `r.fill.dir` or SAGA "Fill Sinks (Wang & Liu)" output, as the script's own comments already recommended.
- **Validation is qualitative/rank-based, not pixel-level.** Alam et al. (2023) publishes ward-level vulnerability classes, not a raw raster, so this validation checks *relative ordering* rather than absolute agreement at the pixel level. A stronger validation would use actual flood-extent records (satellite-derived inundation maps, DNCC/DSCC waterlogging complaint logs, or news-reported 2023–2024 monsoon flooding locations) if these become available.
- **Exposure is based on WorldPop 2020**, which will understate current population in any part of Dhaka that has densified since 2020.

---

*Prepared as part of a code audit and analytical deepening pass. All referenced scripts, output CSVs, and figures are in the project's `scripts/` and `data/output/` folders.*
