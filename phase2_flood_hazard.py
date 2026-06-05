"""
================================================================================
Phase 2: Flood Hazard Mapping — Dhaka, Bangladesh
================================================================================
Script:  scripts/phase2_flood_hazard.py
Project: GIS-based Flood Risk & Vulnerability Mapping, Dhaka
Author:  Urban & Regional Planning, BUET

Methodology:
  - DEM preprocessing (fill sinks, clip to boundary)
  - Slope classification
  - Topographic Wetness Index (TWI) — proxy for inundation potential
  - LULC → SCS Curve Number (CN) for runoff estimation
  - Composite Flood Hazard Index (FHI) from TWI + Elevation + Slope + CN
  - Hazard classification: Low / Medium / High / Very High
  - Return period rainfall thresholds from Phase 1 applied as scenario inputs

Framework: UNDRR — Risk = Hazard × Exposure × Vulnerability
CRS:  Storage in EPSG:4326; analysis in EPSG:32646 (UTM Zone 46N, metric)

Phase 1 Return Period Inputs (Gumbel):
  10-year: 705 mm   |  25-year: 815 mm   |  50-year: 897 mm

Outputs → data/output/phase2/
  dem_clipped.tif              — DEM clipped to Dhaka boundary
  dem_filled.tif               — Sink-filled DEM
  slope_degrees.tif            — Slope in degrees
  slope_classified.tif         — Slope class raster (1–4)
  twi.tif                      — Topographic Wetness Index
  twi_classified.tif           — TWI class raster (1–4)
  cn_raster.tif                — SCS Curve Number from LULC
  cn_classified.tif            — CN class raster (1–4)
  elev_classified.tif          — Elevation class raster (1–4)
  flood_hazard_index.tif       — Composite FHI (continuous 1–4)
  flood_hazard_class.tif       — Final 4-class hazard map
  fig1_dem_and_slope.png
  fig2_twi_map.png
  fig3_cn_map.png
  fig4_hazard_map.png
  fig5_hazard_by_scenario.png
  fig6_hazard_stats.png
  phase2_summary_stats.csv
================================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
from scipy.ndimage import generic_filter, label
import json

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0. PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE     = os.path.dirname(os.path.abspath(__file__))  # scripts/
ROOT     = os.path.dirname(BASE)                        # flood_risk_dhaka/
RAW      = os.path.join(ROOT, "data", "raw")
PROC     = os.path.join(ROOT, "data", "processed")
OUT      = os.path.join(ROOT, "data", "output", "phase2")

os.makedirs(PROC, exist_ok=True)
os.makedirs(OUT,  exist_ok=True)

DEM_RAW       = os.path.join(RAW, "dem",        "srtm_dhaka_30m.tif")
BOUNDARY_SHP  = os.path.join(ROOT,              "dhaka_boundary.shp")
LULC_MOSAIC   = os.path.join(PROC,              "lulc_dhaka_clipped.tif")  # from Phase 2 QGIS prep (see notes)
RAINFALL_CSV  = os.path.join(RAW, "rainfall",   "rainfall_dhaka_monthly_2004_2023.csv")

# Phase 1 return period values (mm, monthly peak — Gumbel fit)
RETURN_PERIODS = {10: 705, 25: 815, 50: 897}

# Target analysis CRS: UTM Zone 46N
CRS_METRIC = "EPSG:32646"
CRS_GEO    = "EPSG:4326"

print("="*72)
print("Phase 2: Flood Hazard Mapping — Dhaka, Bangladesh")
print("="*72)

# ─────────────────────────────────────────────────────────────────────────────
# 1. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def reproject_raster(src_path, dst_path, dst_crs, resampling=Resampling.bilinear):
    """Reproject a raster to a new CRS and save."""
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        meta = src.meta.copy()
        meta.update({"crs": dst_crs, "transform": transform,
                     "width": width, "height": height, "nodata": src.nodata})
        with rasterio.open(dst_path, "w", **meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                )
    print(f"  → Reprojected: {os.path.basename(dst_path)}")


def clip_raster_to_boundary(raster_path, boundary_shp, out_path, target_crs=None):
    """
    Clip a raster to a vector boundary polygon.
    Optionally reproject boundary to match raster CRS first.
    """
    gdf = gpd.read_file(boundary_shp)
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        if gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)
        shapes = [geom.__geo_interface__ for geom in gdf.geometry]
        out_image, out_transform = mask(src, shapes, crop=True, nodata=-9999)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width":  out_image.shape[2],
            "transform": out_transform,
            "nodata": -9999,
        })
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(out_image)
    print(f"  → Clipped: {os.path.basename(out_path)}")


def fill_sinks_simple(dem_array, nodata=-9999):
    """
    Simple iterative sink filling (Planchon-Darboux approximation).
    For production use, replace with SAGA 'Fill Sinks (Wang & Liu)' in QGIS.
    Sets each cell ≥ its lowest neighbor, iterating until stable.
    """
    valid = dem_array != nodata
    filled = dem_array.copy().astype(float)
    filled[~valid] = np.nan
    # Initialise filled surface as max observed elevation
    surface = np.full_like(filled, np.nanmax(filled))
    surface[~valid] = np.nan

    max_iter = 500
    for iteration in range(max_iter):
        changed = False
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            shifted = np.roll(np.roll(surface, dy, axis=0), dx, axis=1)
            candidate = np.minimum(surface, shifted + 1e-5)
            candidate = np.maximum(candidate, filled)  # never below raw DEM
            candidate[~valid] = np.nan
            if np.nanmax(np.abs(candidate - surface)) > 1e-6:
                surface = candidate
                changed = True
        if not changed:
            print(f"  Sink filling converged after {iteration+1} iterations.")
            break

    result = filled.copy()
    result[valid] = surface[valid]
    return result


def compute_slope_degrees(dem_array, cellsize_m, nodata=-9999):
    """
    Compute slope in degrees using Horn's method (same as QGIS/GDAL).
    cellsize_m: pixel size in metres.
    """
    valid_mask = dem_array != nodata
    z = dem_array.astype(float)
    z[~valid_mask] = np.nan

    dz_dx = np.full_like(z, np.nan)
    dz_dy = np.full_like(z, np.nan)

    # Central differences (Horn 1981)
    dz_dx[:, 1:-1] = (z[:, 2:] - z[:, :-2]) / (2 * cellsize_m)
    dz_dy[1:-1, :] = (z[:-2, :] - z[2:, :]) / (2 * cellsize_m)

    # Edge rows/cols — forward/backward differences
    dz_dx[:, 0]  = (z[:, 1]  - z[:, 0])  / cellsize_m
    dz_dx[:, -1] = (z[:, -1] - z[:, -2]) / cellsize_m
    dz_dy[0, :]  = (z[0, :]  - z[1, :])  / cellsize_m
    dz_dy[-1, :] = (z[-2, :] - z[-1, :]) / cellsize_m

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)
    slope_deg[~valid_mask] = nodata
    return slope_deg


def compute_twi(dem_filled, cellsize_m, nodata=-9999):
    """
    Compute Topographic Wetness Index: TWI = ln(As / tan(β))
    where:
      As  = specific contributing area (flow accumulation × cell area) in m²/m
      β   = local slope in radians

    For full accuracy, use SAGA TWI in QGIS (r.watershed or SAGA > TWI).
    This function provides a reasonable Python-only approximation using
    a D8 flow accumulation.
    """
    valid = (dem_filled != nodata) & (~np.isnan(dem_filled))
    z = dem_filled.copy().astype(float)
    z[~valid] = np.nan

    rows, cols = z.shape

    # ── D8 flow direction (8 neighbours, steepest descent) ──
    # Encode directions: 1=E,2=SE,4=S,8=SW,16=W,32=NW,64=N,128=NE
    dy = [0,  1, 1,  1, 0, -1, -1, -1]
    dx = [1,  1, 0, -1,-1, -1,  0,  1]
    diag_dist = np.array([1,np.sqrt(2),1,np.sqrt(2),1,np.sqrt(2),1,np.sqrt(2)])

    flow_dir = np.full((rows, cols), -1, dtype=int)

    for r in range(rows):
        for c in range(cols):
            if not valid[r, c]:
                continue
            max_drop = 0
            best_d = -1
            for d in range(8):
                nr, nc = r + dy[d], c + dx[d]
                if 0 <= nr < rows and 0 <= nc < cols and valid[nr, nc]:
                    drop = (z[r,c] - z[nr,nc]) / (diag_dist[d] * cellsize_m)
                    if drop > max_drop:
                        max_drop = drop
                        best_d = d
            flow_dir[r, c] = best_d

    # ── Flow accumulation (D8 sequential) ──
    # Use topological sort (sort pixels by elevation descending)
    flat_idx = np.argsort(-z.ravel())
    accum = np.ones((rows, cols), dtype=float)

    for idx in flat_idx:
        r, c = divmod(idx, cols)
        if not valid[r, c] or flow_dir[r, c] < 0:
            continue
        d = flow_dir[r, c]
        nr, nc = r + dy[d], c + dx[d]
        if 0 <= nr < rows and 0 <= nc < cols and valid[nr, nc]:
            accum[nr, nc] += accum[r, c]

    # ── Specific contributing area As (m²/m) ──
    As = accum * cellsize_m   # unitless count × cell width

    # ── Slope in radians (minimum 0.001 to avoid ln(inf)) ──
    slope_rad = np.arctan(np.gradient(z, cellsize_m)[0]**2 +
                          np.gradient(z, cellsize_m)[1]**2) ** 0.5
    # Floor slope to avoid division by zero
    slope_rad = np.where(slope_rad < 0.001, 0.001, slope_rad)

    # ── TWI ──
    twi = np.log(As / slope_rad)
    twi[~valid] = nodata
    return twi, accum


def save_raster_like(array, reference_path, out_path, dtype="float32", nodata=-9999):
    """Save array to GeoTIFF with same metadata as reference raster."""
    with rasterio.open(reference_path) as ref:
        meta = ref.meta.copy()
    meta.update({"dtype": dtype, "count": 1, "nodata": nodata})
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(array.astype(dtype), 1)
    print(f"  → Saved: {os.path.basename(out_path)}")


def classify_raster(array, breaks, labels=None, nodata=-9999):
    """
    Classify continuous raster into integer classes.
    breaks: [low, mid1, mid2, high] inclusive upper bounds
    Returns integer array with classes 1..len(breaks), nodata→0
    """
    out = np.zeros_like(array, dtype=np.int16)
    valid = ~np.isnan(array) if np.isnan(nodata) else array != nodata
    for i, thresh in enumerate(breaks):
        if i == 0:
            mask_i = valid & (array <= thresh)
        else:
            mask_i = valid & (array > breaks[i-1]) & (array <= thresh)
        out[mask_i] = i + 1
    # Values above the last break
    out[valid & (array > breaks[-1])] = len(breaks)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. ESA WORLDCOVER → SCS CURVE NUMBER LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

# ESA WorldCover 2021 class values and their CN mapping.
# CN values follow SCS-CN method (AMC-II, HSG C — typical for Dhaka alluvial
# deposits with heavy urbanisation). Adjust HSG per local soil survey.
#
# ESA class → (description, CN)
ESA_CN_LOOKUP = {
    10:  ("Tree cover",             55),   # Forested areas — high infiltration
    20:  ("Shrubland",              65),
    30:  ("Grassland",              74),
    40:  ("Cropland",               78),   # Agricultural fields, seasonal
    50:  ("Built-up",               90),   # Impervious — Dhaka urban fabric
    60:  ("Bare/sparse vegetation", 77),
    70:  ("Snow and ice",           -1),   # Not applicable for Dhaka
    80:  ("Permanent water bodies", 99),   # Rivers, ponds
    90:  ("Herbaceous wetland",     85),   # Low-lying wetlands / beels
    95:  ("Mangroves",              60),
    100: ("Moss and lichen",        45),
}

# CN classification thresholds (NRCS, adapted for Dhaka)
# Class 1 (Low runoff): CN ≤ 60
# Class 2 (Moderate):   CN 60–75
# Class 3 (High):       CN 75–85
# Class 4 (Very High):  CN > 85
CN_BREAKS = [60, 75, 85, 100]


def lulc_to_cn(lulc_array, lookup=ESA_CN_LOOKUP, nodata=-9999):
    """Map ESA WorldCover integer class → SCS Curve Number array."""
    cn_array = np.full_like(lulc_array, nodata, dtype=np.float32)
    for esa_class, (_, cn) in lookup.items():
        if cn > 0:
            cn_array[lulc_array == esa_class] = cn
    return cn_array


# ─────────────────────────────────────────────────────────────────────────────
# 3. DEM PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

print("\n[Step 1] DEM Preprocessing")
print("-"*50)

# 3a. Reproject DEM → UTM 46N for metric analysis
dem_reproj_path = os.path.join(PROC, "dem_utm46n.tif")
if not os.path.exists(dem_reproj_path):
    print("  Reprojecting DEM to EPSG:32646 ...")
    reproject_raster(DEM_RAW, dem_reproj_path, CRS_METRIC)
else:
    print("  DEM (UTM 46N) already exists, skipping reproject.")

# 3b. Clip to Dhaka boundary
dem_clipped_path = os.path.join(OUT, "dem_clipped.tif")
if not os.path.exists(dem_clipped_path):
    print("  Clipping DEM to Dhaka boundary ...")
    clip_raster_to_boundary(dem_reproj_path, BOUNDARY_SHP, dem_clipped_path)
else:
    print("  Using existing dem_clipped.tif (QGIS output)")

# 3c. Read clipped DEM
with rasterio.open(dem_clipped_path) as src:
    dem_raw   = src.read(1).astype(float)
    dem_raw[dem_raw == -32767] = np.nan
    dem_raw[dem_raw <= -32000] = np.nan
    dem_nodata = -32767.0
    dem_meta  = src.meta.copy()
    cellsize  = src.transform.a   # pixel width in metres (UTM)
    dem_crs   = src.crs

print(f"  DEM shape : {dem_raw.shape}")
print(f"  Cell size : {cellsize:.1f} m")
dem_raw = dem_raw.astype(float)
dem_raw[dem_raw == -9999] = np.nan
valid_pixels = dem_raw[~np.isnan(dem_raw)]
print(f"  Elev range: {valid_pixels.min():.1f} – {valid_pixels.max():.1f} m")
print(f"  Mean elev : {valid_pixels.mean():.2f} m  (Dhaka is very flat!)")

# 3d. Fill sinks
print("  Filling sinks (Python approximation — see QGIS note below) ...")
print("  NOTE: For production, run SAGA 'Fill Sinks (Wang & Liu)' in QGIS")
print("        on dem_clipped.tif and save as dem_filled.tif, then continue.")
dem_filled = dem_raw.copy()
dem_filled[np.isnan(dem_filled)] = -9999
dem_filled = fill_sinks_simple(dem_filled, nodata=-9999)
dem_filled = dem_filled.astype(float)
dem_filled[dem_filled == -9999] = np.nan

dem_filled_path = os.path.join(OUT, "dem_filled.tif")
if not os.path.exists(dem_filled_path):
    save_raster_like(dem_filled.astype(np.float32), dem_clipped_path,
                     dem_filled_path, dtype="float32", nodata=-9999)
    print("  Saved Python sink-filled DEM (use GRASS r.fill.dir for better results)")
else:
    print("  Using existing dem_filled.tif (GRASS r.fill.dir output)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. SLOPE CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n[Step 2] Slope Calculation")
print("-"*50)

slope_deg = compute_slope_degrees(dem_raw, cellsize_m=cellsize,
                                  nodata=np.nan)

# Classification for Dhaka (very flat deltaic plain):
# Class 1 (Low):       0–1°   — floodplain floor, highest inundation risk
# Class 2 (Moderate):  1–3°
# Class 3 (High):      3–6°
# Class 4 (Very High): >6°    — elevated ridges, low flood risk
SLOPE_BREAKS = [1, 3, 6, 90]

slope_classified = classify_raster(slope_deg, SLOPE_BREAKS, nodata=dem_nodata)

valid_slope = slope_deg[slope_deg != dem_nodata]
print(f"  Slope range: {valid_slope.min():.2f}° – {valid_slope.max():.2f}°")
print(f"  Mean slope : {valid_slope.mean():.3f}° (confirms deltaic flatness)")
print(f"  % pixels <1°: {(valid_slope < 1).mean()*100:.1f}%")

slope_path  = os.path.join(OUT, "slope_degrees.tif")
slopec_path = os.path.join(OUT, "slope_classified.tif")
save_raster_like(slope_deg.astype(np.float32),  dem_clipped_path, slope_path)
save_raster_like(slope_classified.astype(np.int16), dem_clipped_path,
                 slopec_path, dtype="int16", nodata=0)

# ─────────────────────────────────────────────────────────────────────────────
# 5. TOPOGRAPHIC WETNESS INDEX (TWI)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[Step 3] Topographic Wetness Index (TWI)")
print("-"*50)
print("  Computing D8 flow accumulation and TWI ...")
print("  NOTE: For production, use SAGA > Terrain Analysis > TWI in QGIS")
print("        (r.watershed module) on dem_filled.tif → twi.tif")

twi_path_saga = os.path.join(OUT, "twi_aligned.tif")
with rasterio.open(twi_path_saga) as src:
    twi = src.read(
        1,
        out_shape=(dem_filled.shape[0], dem_filled.shape[1]),
        resampling=Resampling.bilinear
    ).astype(float)
    nodata_val = src.nodata
    if nodata_val is not None:
        twi[twi == nodata_val] = np.nan
flow_accum = np.ones_like(twi)

# TWI classification — Dhaka-specific thresholds
# High TWI = convergent, low-lying terrain = higher flood potential
# Class 1 (Low):       TWI ≤ 6   — well-drained upland
# Class 2 (Moderate):  TWI 6–8
# Class 3 (High):      TWI 8–10
# Class 4 (Very High): TWI > 10  — valley bottoms, channels, wetlands
TWI_BREAKS = [6, 8, 10, 99]

valid_twi = twi[twi != dem_nodata]
print(f"  TWI range : {np.nanmin(valid_twi):.2f} – {np.nanmax(valid_twi):.2f}")
print(f"  Mean TWI  : {np.nanmean(valid_twi):.2f}")

twi_classified = classify_raster(twi, TWI_BREAKS, nodata=dem_nodata)

twi_path  = os.path.join(OUT, "twi.tif")
twic_path = os.path.join(OUT, "twi_classified.tif")
save_raster_like(twi.astype(np.float32), dem_clipped_path, twi_path)
save_raster_like(twi_classified.astype(np.int16), dem_clipped_path,
                 twic_path, dtype="int16", nodata=0)

# ─────────────────────────────────────────────────────────────────────────────
# 6. ELEVATION CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n[Step 4] Elevation Classification")
print("-"*50)

# Dhaka elevation thresholds (SRTM values, metres above WGS84 geoid)
# Class 1 (Very High risk): elev ≤ 4 m   — tidal/floodplain
# Class 2 (High risk):      4–7 m
# Class 3 (Moderate risk):  7–12 m
# Class 4 (Low risk):       > 12 m       — Pleistocene terrace (Madhupur Tract)
ELEV_BREAKS = [4, 7, 12, 9999]

# NOTE: For Dhaka the Madhupur elevated terrace sits around 8–15 m;
# most of the city is below 8 m. Adjust these after inspecting your DEM
# histogram in QGIS (Layer Properties → Histogram).

dem_clipped_for_elev = dem_raw.copy()
elev_classified = classify_raster(dem_clipped_for_elev, ELEV_BREAKS, nodata=np.nan)

print(f"  Class 1 (≤4 m) :  {(elev_classified==1).sum():,} pixels")
print(f"  Class 2 (4–7 m):  {(elev_classified==2).sum():,} pixels")
print(f"  Class 3 (7–12 m): {(elev_classified==3).sum():,} pixels")
print(f"  Class 4 (>12 m):  {(elev_classified==4).sum():,} pixels")

elevc_path = os.path.join(OUT, "elev_classified.tif")
save_raster_like(elev_classified.astype(np.int16), dem_clipped_path,
                 elevc_path, dtype="int16", nodata=0)

# ─────────────────────────────────────────────────────────────────────────────
# 7. LULC → CURVE NUMBER (CN)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[Step 5] LULC → SCS Curve Number")
print("-"*50)

# Check if LULC mosaic exists (should be prepared in QGIS first)
if os.path.exists(LULC_MOSAIC):
    print("  Reading LULC mosaic ...")
    with rasterio.open(LULC_MOSAIC) as src:
        # Resample LULC (10 m) to DEM grid (30 m) using nearest-neighbour
        from rasterio.enums import Resampling as RS
        lulc_resampled = src.read(
            1,
            out_shape=(dem_raw.shape[0], dem_raw.shape[1]),
            resampling=RS.nearest
        )
    cn_array = lulc_to_cn(lulc_resampled)
    cn_classified = classify_raster(cn_array, CN_BREAKS, nodata=-9999)

    print("  ESA class distribution:")
    for esa_class, (desc, cn) in ESA_CN_LOOKUP.items():
        count = (lulc_resampled == esa_class).sum()
        pct   = count / lulc_resampled.size * 100
        if count > 0:
            print(f"    {esa_class:3d} {desc:<25s} CN={cn:3d}  {pct:5.1f}%")

    cn_path  = os.path.join(OUT, "cn_raster.tif")
    cnc_path = os.path.join(OUT, "cn_classified.tif")
    save_raster_like(cn_array, dem_clipped_path, cn_path)
    save_raster_like(cn_classified.astype(np.int16), dem_clipped_path,
                     cnc_path, dtype="int16", nodata=0)
    HAVE_LULC = True

else:
    print("  ⚠  LULC mosaic not found at:", LULC_MOSAIC)
    print("  ⚠  Proceeding WITHOUT CN layer.")
    print("  →  In QGIS: merge ESA WorldCover tiles, clip to Dhaka,")
    print("     reproject to EPSG:32646, save as data/processed/lulc_dhaka_clipped.tif")
    print("  →  Then re-run this script; CN will be included automatically.")
    cn_classified = np.zeros_like(dem_filled, dtype=np.int16)
    HAVE_LULC = False

# ─────────────────────────────────────────────────────────────────────────────
# 8. COMPOSITE FLOOD HAZARD INDEX (FHI)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[Step 6] Composite Flood Hazard Index")
print("-"*50)

# ── Weights (must sum to 1.0) ──
# Based on literature for deltaic urban cities (Tran et al. 2008,
# Khosravi et al. 2016, Bhuiyan & Dutta 2012 for Bangladesh):
#
#   TWI        0.35  — primary driver; captures flow convergence
#   Elevation  0.30  — low elevation = high flood depth potential
#   Slope      0.20  — low slope = slow drainage = prolonged inundation
#   CN         0.15  — urban impervious surfaces amplify runoff
#
# NOTE: If LULC is unavailable (HAVE_LULC=False), redistribute
#       CN weight to TWI (+0.10) and Elevation (+0.05).

if HAVE_LULC:
    W_TWI  = 0.35
    W_ELEV = 0.30
    W_SLOP = 0.20
    W_CN   = 0.15
else:
    W_TWI  = 0.45
    W_ELEV = 0.35
    W_SLOP = 0.20
    W_CN   = 0.00

print(f"  Weights — TWI:{W_TWI}  Elev:{W_ELEV}  Slope:{W_SLOP}  CN:{W_CN}")
print(f"  LULC/CN included: {HAVE_LULC}")

# All classified layers are 1–4; scale to 0–1 by dividing by 4.
# For hazard-positive factors (TWI, CN, low-slope, low-elev):
#   class 4 = highest hazard
# For hazard-negative factors (elevation, slope):
#   class 4 = lowest hazard → invert: score = (5 - class) / 4

def norm_hazard(classified_array, invert=False):
    """Normalise classified integer array (1–4) to 0–1 hazard score."""
    arr = classified_array.astype(float)
    arr[arr == 0] = np.nan  # nodata
    if invert:
        arr = 5.0 - arr   # 4→1, 1→4
    return arr / 4.0


twi_score  = norm_hazard(twi_classified,   invert=False)  # high TWI = high hazard
elev_score = norm_hazard(elev_classified,  invert=False)  # low elev = high hazard
            # NOTE: elev classified 1=≤4m (highest hazard) → class 1 = worst
            # So class 1 should score highest: invert=False keeps this
            # (class 1 = score 0.25, class 4 = score 1.0) — WRONG, must invert.
            # Correction: elev class 1 = worst flood risk → should map to 1.0
elev_score = norm_hazard(elev_classified,  invert=True)   # fixed: low elev → high score
slope_score= norm_hazard(slope_classified, invert=True)   # low slope → high hazard

if HAVE_LULC:
    cn_score = norm_hazard(cn_classified, invert=False)  # high CN = high runoff
else:
    cn_score = np.zeros_like(twi_score)

# Composite FHI
fhi = (W_TWI  * twi_score +
       W_ELEV * elev_score +
       W_SLOP * slope_score +
       W_CN   * cn_score)

fhi_nodata = np.isnan(twi_score) | np.isnan(elev_score) | np.isnan(slope_score)
fhi_raw = fhi.copy()
fhi_raw[fhi_nodata] = -9999

print(f"  FHI range : {np.nanmin(fhi):.4f} – {np.nanmax(fhi):.4f}")
print(f"  FHI mean  : {np.nanmean(fhi):.4f}")

# ── Final hazard classification (Jenks / quantile cut on valid values) ──
# Using fixed breaks on 0–1 scale for reproducibility and interpretability:
# Low:       FHI 0.00–0.35
# Medium:    FHI 0.35–0.55
# High:      FHI 0.55–0.75
# Very High: FHI > 0.75

FHI_BREAKS = [0.35, 0.55, 0.75, 1.0]

hazard_class = np.zeros_like(fhi_raw, dtype=np.int16)
valid_fhi = fhi_raw != -9999

hazard_labels = ["Low", "Medium", "High", "Very High"]
for i, thresh in enumerate(FHI_BREAKS):
    lower = FHI_BREAKS[i-1] if i > 0 else 0.0
    mask_i = valid_fhi & (fhi_raw >= lower) & (fhi_raw <= thresh)
    hazard_class[mask_i] = i + 1

# Count pixels per class
total_valid = valid_fhi.sum()
print("\n  Hazard class distribution:")
for i, label in enumerate(hazard_labels):
    count = (hazard_class == i+1).sum()
    pct   = count / total_valid * 100
    print(f"    Class {i+1} {label:<12s}: {count:>8,} pixels  ({pct:5.1f}%)")

# Save FHI rasters
fhi_path     = os.path.join(OUT, "flood_hazard_index.tif")
hazard_path  = os.path.join(OUT, "flood_hazard_class.tif")
save_raster_like(fhi_raw.astype(np.float32),    dem_clipped_path, fhi_path)
save_raster_like(hazard_class.astype(np.int16), dem_clipped_path,
                 hazard_path, dtype="int16", nodata=0)

# ─────────────────────────────────────────────────────────────────────────────
# 9. RETURN PERIOD SCENARIO ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[Step 7] Return Period Scenario Analysis")
print("-"*50)
print("  Applying Phase 1 return period thresholds to estimate")
print("  runoff depth per LULC class using SCS-CN method.")

# SCS-CN: Runoff depth Q = (P - Ia)² / (P - Ia + S)
# where  S = (1000/CN) - 10  (inches), Ia = 0.2S (initial abstraction)
# Convert to mm:  S_mm = 25.4 * S,  P in mm

def scs_runoff_mm(rainfall_mm, cn):
    """SCS Curve Number method. Returns runoff Q in mm."""
    S_in  = (1000.0 / cn) - 10.0   # retention parameter (inches)
    S_mm  = 25.4 * S_in
    Ia    = 0.2 * S_mm              # initial abstraction
    P     = rainfall_mm
    if P <= Ia:
        return 0.0
    Q = (P - Ia)**2 / (P - Ia + S_mm)
    return Q

print("\n  SCS-CN Runoff by LULC Class and Return Period:")
print(f"  {'LULC':<25s}  {'CN':>4s}  " +
      "  ".join([f"Q{rp}yr(mm)" for rp in RETURN_PERIODS]))

for esa_class, (desc, cn) in ESA_CN_LOOKUP.items():
    if cn <= 0:
        continue
    runoffs = {rp: scs_runoff_mm(p, cn) for rp, p in RETURN_PERIODS.items()}
    row = f"  {desc:<25s}  CN={cn:2d}  "
    row += "  ".join([f"{q:>10.1f}" for q in runoffs.values()])
    print(row)

# Build scenario summary table
scenario_rows = []
for esa_class, (desc, cn) in ESA_CN_LOOKUP.items():
    if cn <= 0:
        continue
    for rp, p in RETURN_PERIODS.items():
        q = scs_runoff_mm(p, cn)
        scenario_rows.append({
            "ESA_Class": esa_class,
            "LULC_Desc": desc,
            "CN": cn,
            "Return_Period_yr": rp,
            "Rainfall_mm": p,
            "Runoff_mm": round(q, 2),
            "Runoff_Ratio": round(q / p, 3),
        })

df_scenarios = pd.DataFrame(scenario_rows)

# ─────────────────────────────────────────────────────────────────────────────
# 10. SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

valid_dem   = dem_filled[dem_filled != dem_nodata]
valid_slope2 = slope_deg[slope_deg != dem_nodata]
valid_twi2   = twi[twi != dem_nodata]

summary = {
    "Parameter": [
        "DEM mean elevation (m)",
        "DEM min elevation (m)",
        "DEM max elevation (m)",
        "Mean slope (degrees)",
        "% area with slope < 1°",
        "Mean TWI",
        "Max TWI",
        "Hazard Class 1 - Low (%)",
        "Hazard Class 2 - Medium (%)",
        "Hazard Class 3 - High (%)",
        "Hazard Class 4 - Very High (%)",
        "FHI mean",
        "FHI std",
        "LULC/CN included",
        "Return Period 10yr rainfall (mm)",
        "Return Period 25yr rainfall (mm)",
        "Return Period 50yr rainfall (mm)",
    ],
    "Value": [
        round(float(valid_dem.mean()), 2),
        round(float(valid_dem.min()), 2),
        round(float(valid_dem.max()), 2),
        round(float(valid_slope2.mean()), 4),
        round(float((valid_slope2 < 1).mean() * 100), 1),
        round(float(np.nanmean(valid_twi2)), 3),
        round(float(np.nanmax(valid_twi2)), 3),
        round(float((hazard_class == 1).sum() / total_valid * 100), 1),
        round(float((hazard_class == 2).sum() / total_valid * 100), 1),
        round(float((hazard_class == 3).sum() / total_valid * 100), 1),
        round(float((hazard_class == 4).sum() / total_valid * 100), 1),
        round(float(np.nanmean(fhi)), 4),
        round(float(np.nanstd(fhi)), 4),
        str(HAVE_LULC),
        RETURN_PERIODS[10],
        RETURN_PERIODS[25],
        RETURN_PERIODS[50],
    ]
}

df_summary = pd.DataFrame(summary)
df_summary.to_csv(os.path.join(OUT, "phase2_summary_stats.csv"), index=False)
df_scenarios.to_csv(os.path.join(OUT, "return_period_runoff_scenarios.csv"), index=False)
print("\n  Summary stats saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 11. VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

# Colour scheme consistent across all maps
HAZARD_CMAP   = mcolors.ListedColormap(["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"])
HAZARD_BOUNDS = [0.5, 1.5, 2.5, 3.5, 4.5]
HAZARD_NORM   = mcolors.BoundaryNorm(HAZARD_BOUNDS, HAZARD_CMAP.N)
HAZARD_NAMES  = ["Low", "Medium", "High", "Very High"]
HAZARD_COLORS = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]


def hazard_legend_patches():
    return [mpatches.Patch(color=c, label=l)
            for c, l in zip(HAZARD_COLORS, HAZARD_NAMES)]


print("\n[Step 8] Generating figures ...")

# ── Fig 1: DEM + Slope ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                         facecolor="#1a1a2e")
fig.suptitle("Phase 2: DEM Preprocessing — Dhaka, Bangladesh",
             color="white", fontsize=14, fontweight="bold", y=1.01)

ax1, ax2 = axes

rows = np.where(np.any(hazard_class > 0, axis=1))[0]
cols = np.where(np.any(hazard_class > 0, axis=0))[0]
hazard_cropped = hazard_class[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1].astype(float)
hazard_cropped[hazard_cropped == 0] = np.nan

fig, ax = plt.subplots(figsize=(10, 10), facecolor="#1a1a2e")
im = ax.imshow(hazard_cropped, cmap=HAZARD_CMAP, norm=HAZARD_NORM,
               interpolation="nearest")
ax.set_title("Flood Hazard Classification\nDhaka, Bangladesh",
             color="white", fontsize=14, fontweight="bold")
ax.axis("off")
ax.set_facecolor("#0a0a1a")
fig.patch.set_facecolor("#1a1a2e")
legend = ax.legend(handles=hazard_legend_patches(), loc="lower left",
                   framealpha=0.85, facecolor="#0f3460",
                   labelcolor="white", fontsize=10,
                   title="Hazard Level", title_fontsize=10)
legend.get_title().set_color("white")
counts = [(hazard_class == i).sum() for i in range(1, 5)]
annot = "\n".join([f"{n}: {c/total_valid*100:.1f}%"
                   for n, c in zip(HAZARD_NAMES, counts)])
ax.text(0.98, 0.02, annot, transform=ax.transAxes,
        ha="right", va="bottom", color="white", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="#0f3460", alpha=0.9))
weight_txt = (f"Weights: TWI={W_TWI} | Elev={W_ELEV} | "
              f"Slope={W_SLOP} | CN={W_CN}")
ax.text(0.02, 0.98, weight_txt, transform=ax.transAxes,
        ha="left", va="top", color="#aaaaaa", fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_hazard_map.png"),
            dpi=200, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
print("  → fig4_hazard_map.png")

hazard_plot = hazard_class.copy().astype(float)
hazard_plot[hazard_plot == 0] = np.nan

dem_plot = dem_raw.copy()
valid_rows = np.any(~np.isnan(dem_plot), axis=1)
valid_cols = np.any(~np.isnan(dem_plot), axis=0)
dem_plot = dem_plot[valid_rows][:, valid_cols]
im1 = ax1.imshow(dem_plot, cmap="terrain", interpolation="bilinear")
ax1.set_title("Filled DEM (metres)", color="white", fontsize=11)
ax1.axis("off")
cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cb1.ax.yaxis.set_tick_params(color="white")
cb1.set_label("Elevation (m)", color="white")
plt.setp(plt.getp(cb1.ax.axes, "yticklabels"), color="white")

slope_plot = slope_deg.copy().astype(float)
slope_plot[slope_plot == dem_nodata] = np.nan
im2 = ax2.imshow(slope_plot, cmap="YlOrRd", vmin=0, vmax=10,
                 interpolation="bilinear")
ax2.set_title("Slope (degrees)", color="white", fontsize=11)
ax2.axis("off")
cb2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cb2.ax.yaxis.set_tick_params(color="white")
cb2.set_label("Slope (°)", color="white")
plt.setp(plt.getp(cb2.ax.axes, "yticklabels"), color="white")

for ax in axes:
    ax.set_facecolor("#0f3460")
fig.patch.set_facecolor("#1a1a2e")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_dem_and_slope.png"),
            dpi=180, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
print("  → fig1_dem_and_slope.png")

# ── Fig 2: TWI Map ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8), facecolor="#1a1a2e")
twi_plot = twi.copy().astype(float)
twi_plot[twi_plot == dem_nodata] = np.nan
im = ax.imshow(twi_plot, cmap="Blues", vmin=4, vmax=14)
ax.set_title("Topographic Wetness Index (TWI)\nDhaka, Bangladesh",
             color="white", fontsize=13, fontweight="bold")
ax.axis("off")
ax.set_facecolor("#0f3460")
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("TWI Value", color="white")
cb.ax.yaxis.set_tick_params(color="white")
plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")

# Annotate break thresholds
for thresh, label_txt in zip(TWI_BREAKS[:-1], ["Low|Mod", "Mod|High", "High|VHigh"]):
    cb.ax.axhline(thresh, color="yellow", linewidth=1.2, linestyle="--")
    cb.ax.text(2.2, thresh, label_txt, color="yellow", fontsize=7, va="center")

fig.patch.set_facecolor("#1a1a2e")
ax.text(0.02, 0.02, f"TWI Mean: {np.nanmean(twi_plot):.2f}  |  D8 flow accumulation",
        transform=ax.transAxes, color="#aaaaaa", fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_twi_map.png"),
            dpi=180, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
print("  → fig2_twi_map.png")

# ── Fig 3: CN Map (if LULC available) or placeholder ───────────────────────
fig, ax = plt.subplots(figsize=(8, 8), facecolor="#1a1a2e")
if HAVE_LULC:
    cn_plot = cn_array.copy()
    cn_plot[cn_plot == -9999] = np.nan
    im = ax.imshow(cn_plot, cmap="RdYlGn_r", vmin=40, vmax=100)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Curve Number (CN)", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")
    title = "SCS Curve Number from ESA WorldCover\nDhaka, Bangladesh"
else:
    ax.text(0.5, 0.5, "LULC mosaic not yet prepared.\n\nRun QGIS step first:\n"
            "Merge ESA WorldCover tiles → clip\n"
            "Reproject → EPSG:32646\nSave as data/processed/lulc_dhaka_clipped.tif",
            ha="center", va="center", color="white", fontsize=12,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="#e74c3c", alpha=0.8))
    title = "Curve Number Map (Pending LULC)"

ax.set_title(title, color="white", fontsize=13, fontweight="bold")
ax.axis("off")
ax.set_facecolor("#0f3460")
fig.patch.set_facecolor("#1a1a2e")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_cn_map.png"),
            dpi=180, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
print("  → fig3_cn_map.png")

# ── Fig 4: Final Flood Hazard Map ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 10), facecolor="#1a1a2e")
ax.set_aspect('equal')

hazard_plot = hazard_class.copy().astype(float)
hazard_plot[hazard_plot == 0] = np.nan

im = ax.imshow(hazard_plot, cmap=HAZARD_CMAP, norm=HAZARD_NORM,
               interpolation="nearest", extent=[0, hazard_plot.shape[1],
               hazard_plot.shape[0], 0])
ax.set_xlim(0, hazard_plot.shape[1])
ax.set_ylim(hazard_plot.shape[0], 0)
ax.set_title("Flood Hazard Classification\nDhaka, Bangladesh",
             color="white", fontsize=14, fontweight="bold")
ax.axis("off")
ax.set_facecolor("#0a0a1a")
fig.patch.set_facecolor("#1a1a2e")

legend = ax.legend(handles=hazard_legend_patches(), loc="lower left",
                   framealpha=0.85, facecolor="#0f3460",
                   labelcolor="white", fontsize=10,
                   title="Hazard Level", title_fontsize=10)
legend.get_title().set_color("white")

# Pixel count annotations
counts = [(hazard_class == i).sum() for i in range(1, 5)]
annot = "\n".join([f"{n}: {c/total_valid*100:.1f}%"
                   for n, c in zip(HAZARD_NAMES, counts)])
ax.text(0.98, 0.02, annot, transform=ax.transAxes,
        ha="right", va="bottom", color="white", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="#0f3460", alpha=0.9))

weight_txt = (f"Weights: TWI={W_TWI} | Elev={W_ELEV} | "
              f"Slope={W_SLOP} | CN={W_CN}")
ax.text(0.02, 0.98, weight_txt, transform=ax.transAxes,
        ha="left", va="top", color="#aaaaaa", fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_hazard_map.png"),
            dpi=200, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
print("  → fig4_hazard_map.png")

# ── Fig 5: Return Period Runoff Scenarios ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#1a1a2e",
                         sharey=True)

lulc_labels  = [r["LULC_Desc"] for r in scenario_rows
                if r["Return_Period_yr"] == 10]
cn_vals      = [r["CN"] for r in scenario_rows
                if r["Return_Period_yr"] == 10]
colors_bar   = ["#e74c3c" if cn >= 85 else "#f39c12" if cn >= 75
                else "#2ecc71" for cn in cn_vals]

for i, (rp, p) in enumerate(RETURN_PERIODS.items()):
    ax  = axes[i]
    qs  = [r["Runoff_mm"] for r in scenario_rows if r["Return_Period_yr"] == rp]
    y   = range(len(qs))
    ax.barh(y, qs, color=colors_bar, edgecolor="none", height=0.7)
    ax.set_yticks(list(y))
    ax.set_yticklabels(lulc_labels if i == 0 else [], color="white", fontsize=9)
    ax.set_xlabel("Runoff Depth (mm)", color="white", fontsize=10)
    ax.set_title(f"{rp}-Year Return Period\n(P = {p} mm)",
                 color="white", fontsize=11, fontweight="bold")
    ax.set_facecolor("#0f3460")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333366")
    for j, (q, cn) in enumerate(zip(qs, cn_vals)):
        ax.text(q + 2, j, f"{q:.0f} mm", va="center",
                color="white", fontsize=8)

fig.suptitle("SCS-CN Runoff Depth by LULC and Return Period — Dhaka",
             color="white", fontsize=13, fontweight="bold")
fig.patch.set_facecolor("#1a1a2e")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_hazard_by_scenario.png"),
            dpi=180, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
print("  → fig5_hazard_by_scenario.png")

# ── Fig 6: Hazard Statistics Summary ────────────────────────────────────────
fig = plt.figure(figsize=(14, 6), facecolor="#1a1a2e")
gs  = GridSpec(1, 3, figure=fig, wspace=0.3)

# 6a: Hazard class pie
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("#0f3460")
pct_vals = [float((hazard_class == i+1).sum()) / total_valid * 100
            for i in range(4)]
wedges, texts, autotexts = ax1.pie(
    pct_vals, labels=HAZARD_NAMES, colors=HAZARD_COLORS,
    autopct="%1.1f%%", startangle=90,
    textprops={"color": "white", "fontsize": 9},
    wedgeprops={"edgecolor": "#1a1a2e", "linewidth": 2}
)
for at in autotexts:
    at.set_color("white")
ax1.set_title("Hazard Class Distribution", color="white", fontsize=10)

# 6b: FHI histogram
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("#0f3460")
fhi_valid = fhi_raw[fhi_raw != -9999]
ax2.hist(fhi_valid, bins=50, color="#3498db", edgecolor="none", alpha=0.85)
for thresh, col in zip(FHI_BREAKS[:-1], HAZARD_COLORS[1:]):
    ax2.axvline(thresh, color=col, linewidth=1.5, linestyle="--", label=f"{thresh}")
ax2.set_xlabel("FHI Value", color="white")
ax2.set_ylabel("Pixel Count", color="white")
ax2.set_title("Flood Hazard Index Distribution", color="white", fontsize=10)
ax2.tick_params(colors="white")
ax2.spines[:].set_color("#333366")
ax2.legend(fontsize=8, labelcolor="white", facecolor="#0a0a1a",
           title="Class breaks", title_fontsize=8)

# 6c: TWI vs Elevation scatter (sampled)
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor("#0f3460")
valid_mask2 = (dem_filled != dem_nodata) & (~np.isnan(twi))
dem_flat  = dem_filled[valid_mask2].ravel()
twi_flat  = twi[valid_mask2].ravel()
hc_flat   = hazard_class[valid_mask2].ravel()

# Sample 5000 pts for scatter
rng = np.random.default_rng(42)
idx_s = rng.choice(len(dem_flat), min(5000, len(dem_flat)), replace=False)
sc = ax3.scatter(dem_flat[idx_s], twi_flat[idx_s],
                 c=hc_flat[idx_s], cmap=HAZARD_CMAP, norm=HAZARD_NORM,
                 alpha=0.4, s=5)
ax3.set_xlabel("Elevation (m)", color="white")
ax3.set_ylabel("TWI", color="white")
ax3.set_title("Elevation vs TWI\n(coloured by hazard class)", color="white", fontsize=10)
ax3.tick_params(colors="white")
ax3.spines[:].set_color("#333366")

fig.suptitle("Phase 2: Flood Hazard Statistics — Dhaka",
             color="white", fontsize=13, fontweight="bold")
fig.patch.set_facecolor("#1a1a2e")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_hazard_stats.png"),
            dpi=180, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
print("  → fig6_hazard_stats.png")

# ─────────────────────────────────────────────────────────────────────────────
# 12. DONE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*72)
print("Phase 2 complete.")
print(f"Outputs saved to: {OUT}")
print("="*72)
print("""
─────────────────────────────────────────────────────────────────
QGIS VALIDATION CHECKLIST (do these before Phase 3):
─────────────────────────────────────────────────────────────────
□ 1. Load flood_hazard_class.tif in QGIS; apply HAZARD_COLORS palette
□ 2. Overlay dhaka_boundary.shp to confirm spatial alignment
□ 3. Compare hazard class map against known flood-prone wards:
     (Demra, Amin Bazar, Mirpur, Khilkhet, Rayer Bazar)
□ 4. If results look off → adjust FHI_BREAKS or TWI_BREAKS constants
□ 5. Run SAGA TWI in QGIS → compare to Python TWI output
□ 6. Prepare LULC mosaic (ESA tiles → Merge → Clip → Reproject)
     and rerun with HAVE_LULC=True for more accurate CN weighting
□ 7. Export flood_hazard_class.tif as PNG/PDF map layout in QGIS
─────────────────────────────────────────────────────────────────
NEXT: Phase 3 — Exposure & Vulnerability Mapping
  - Population exposure by hazard zone (WorldPop raster)
  - Admin-unit level aggregation (GADM Level 4)
  - Vulnerability index: population density, poverty, building density
  - Final risk = Hazard × Exposure × Vulnerability raster
─────────────────────────────────────────────────────────────────
""")
