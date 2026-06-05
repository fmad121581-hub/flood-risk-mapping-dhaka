# %% ============================================================
# Phase 4 — Cartographic Maps
# Project: GIS-based Flood Risk & Vulnerability Mapping, Dhaka
# Output:  data/output/phase4/maps/  (PNG 300dpi + PDF)
# CRS:     All analysis in EPSG:32646 (UTM 46N)
#          Display labels/titles use geographic context only
# ============================================================

# %% --- 0. Imports and paths ---

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from matplotlib_scalebar.scalebar import ScaleBar
import geopandas as gpd
import rasterio
from rasterio.plot import show as rshow
from rasterio.mask import mask as rmask
import contextily as ctx

# ── Project root (adjust if needed) ──────────────────────────
ROOT = r"C:/Users/user/OneDrive/Must"

# ── Input paths ──────────────────────────────────────────────
BOUNDARY_UTM   = os.path.join(ROOT, "dhaka_boundary_utm.shp")
WARDS          = os.path.join(ROOT, "data/processed/gadm_dhaka_l4.shp")
DEM            = os.path.join(ROOT, "data/output/phase2/dem_clipped.tif")
HAZARD_CLASS   = os.path.join(ROOT, "data/output/phase2/flood_hazard_class.tif")
RISK_INDEX     = os.path.join(ROOT, "data/output/phase3/risk_index.tif")
RISK_CLASS     = os.path.join(ROOT, "data/output/phase3/risk_classified.tif")
WARD_RISK_SHP  = os.path.join(ROOT, "data/output/phase3/ward_risk.shp")
RAINFALL_CSV   = os.path.join(ROOT, "data/raw/rainfall/rainfall_dhaka_monthly_2004_2023.csv")

# ── Output folder ────────────────────────────────────────────
OUT = os.path.join(ROOT, "data/output/phase4/maps")
os.makedirs(OUT, exist_ok=True)

# ── Shared style ─────────────────────────────────────────────
TITLE_FONT  = {"fontsize": 14, "fontweight": "bold", "color": "#1a1a2e"}
LABEL_FONT  = {"fontsize": 9,  "color": "#444444"}
CAPTION_FONT= {"fontsize": 7.5,"color": "#666666", "style": "italic"}
DPI         = 300
FIG_W, FIG_H = 10, 9        # inches — A4-friendly landscape section

# ── Helper: add north arrow (pure matplotlib, no basemap needed) ──
def add_north_arrow(ax, x=0.96, y=0.96, size=0.05):
    """Place a north arrow at axes-fraction coordinates."""
    ax.annotate(
        "N", xy=(x, y), xytext=(x, y - size),
        xycoords="axes fraction", textcoords="axes fraction",
        ha="center", va="top",
        fontsize=11, fontweight="bold", color="#1a1a2e",
        arrowprops=dict(arrowstyle="-|>", color="#1a1a2e", lw=1.5),
    )

# ── Helper: add scale bar (requires matplotlib-scalebar) ─────
def add_scalebar(ax):
    sb = ScaleBar(
        1, "m", length_fraction=0.25,
        location="lower left", pad=0.5,
        color="#1a1a2e", box_alpha=0.7,
        font_properties={"size": 8},
    )
    ax.add_artist(sb)

# ── Helper: add data-source caption ──────────────────────────
def add_caption(fig, text):
    fig.text(
        0.01, 0.01, text,
        **CAPTION_FONT, transform=fig.transFigure,
    )

# ── Helper: read raster as masked array ──────────────────────
def read_raster(path, nodata=None):
    with rasterio.open(path) as src:
        data = src.read(1).astype(float)
        nd = nodata if nodata is not None else src.nodata
        if nd is not None:
            data = np.ma.masked_equal(data, nd)
        extent = [src.bounds.left, src.bounds.right,
                  src.bounds.bottom, src.bounds.top]
        crs = src.crs
    return data, extent, crs


# %% ============================================================
# MAP 1 — Study Area Overview
# Shows wards + boundary + basemap + locator inset
# ============================================================
print("Generating Map 1 — Study area overview ...")

boundary = gpd.read_file(BOUNDARY_UTM)   # EPSG:32646
wards    = gpd.read_file(WARDS)

# Reproject to Web Mercator for contextily basemap
boundary_wm = boundary.to_crs(epsg=3857)
wards_wm    = wards.to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
wards_wm.plot(
    ax=ax,
    facecolor="none",
    edgecolor="#555555",
    linewidth=0.4,
    zorder=2,
)
boundary_wm.plot(
    ax=ax,
    facecolor="none",
    edgecolor="#c0392b",
    linewidth=1.8,
    zorder=3,
)
try:
    ctx.add_basemap(
        ax, source=ctx.providers.CartoDB.Positron, zoom=12
    )
except Exception:
    ax.set_facecolor("#d0e8f5")   # fallback if no internet

# Legend
red_patch  = mpatches.Patch(edgecolor="#c0392b", facecolor="none",
                              linewidth=2, label="Dhaka study area boundary")
gray_patch = mpatches.Patch(edgecolor="#555555", facecolor="none",
                              linewidth=1, label="GADM L4 ward boundaries\n(n=242)")
ax.legend(handles=[red_patch, gray_patch], loc="lower right",
          fontsize=8, framealpha=0.85)

add_north_arrow(ax)
add_scalebar(ax)
ax.set_title("Map 1 — Study Area: Dhaka Metropolitan Region",
             **TITLE_FONT, pad=12)
ax.set_xlabel("Easting (m, EPSG:32646)", **LABEL_FONT)
ax.set_ylabel("Northing (m, EPSG:32646)", **LABEL_FONT)
ax.tick_params(labelsize=7)
add_caption(fig,
    "Data: GADM v4.1 | Boundary: dhaka_boundary_utm.shp (EPSG:32646) | "
    "Basemap: © CartoDB Positron")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "map1_study_area.png"), dpi=DPI, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "map1_study_area.pdf"), bbox_inches="tight")
plt.close(fig)
print("  → map1_study_area.png saved")


# %% ============================================================
# MAP 2 — Annual Rainfall Variability (2004–2023)
# Time-series + bar chart hybrid: not a spatial map but a key
# analytical figure for the rainfall chapter
# ============================================================
print("Generating Map 2 — Rainfall variability figure ...")

# Read CSV (skip 9-line NASA POWER header)
df_raw = pd.read_csv(RAINFALL_CSV, skiprows=9)
df_raw.columns = df_raw.columns.str.strip()
# Rename month columns
months = ["JAN","FEB","MAR","APR","MAY","JUN",
          "JUL","AUG","SEP","OCT","NOV","DEC"]
df = df_raw.copy()
df = df.rename(columns={"YEAR": "year", "ANN": "annual"})
df_m = df[["year"] + months].copy()

annual = df[["year", "annual"]].copy()
mean_ann = annual["annual"].mean()

fig, axes = plt.subplots(2, 1, figsize=(FIG_W, FIG_H),
                          gridspec_kw={"height_ratios": [1.6, 1]})

# ── Top: annual rainfall bar chart ──
ax = axes[0]
colors = ["#c0392b" if y == 2017 else "#2980b9" for y in annual["year"]]
bars = ax.bar(annual["year"], annual["annual"], color=colors,
              edgecolor="white", linewidth=0.4, width=0.75, zorder=2)
ax.axhline(mean_ann, color="#e67e22", linewidth=1.5, linestyle="--",
           label=f"Mean: {mean_ann:.0f} mm", zorder=3)
ax.axhspan(mean_ann - annual["annual"].std(),
           mean_ann + annual["annual"].std(),
           alpha=0.12, color="#e67e22", label="±1 SD", zorder=1)

# Annotate 2017 outlier
ax.annotate(
    f"2017 outlier\n4,631 mm",
    xy=(2017, 4630), xytext=(2018.5, 4300),
    fontsize=7.5, color="#c0392b", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1),
)

ax.set_ylabel("Annual rainfall (mm)", **LABEL_FONT)
ax.set_title("Map 2 — Annual Rainfall Variability, Dhaka 2004–2023",
             **TITLE_FONT, pad=10)
ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
ax.set_xlim(2003.5, 2023.5)
ax.set_ylim(0, 5200)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"{int(x):,}"))
ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
ax.tick_params(labelsize=7)

# ── Bottom: monthly mean + SD ──
ax2 = axes[1]
mon_mean = df_m[months].mean()
mon_sd   = df_m[months].std()
x = np.arange(len(months))
ax2.bar(x, mon_mean, color="#2980b9", edgecolor="white",
        linewidth=0.4, width=0.6, label="Monthly mean", zorder=2)
ax2.errorbar(x, mon_mean, yerr=mon_sd, fmt="none",
             ecolor="#c0392b", elinewidth=1, capsize=3, zorder=3)
ax2.set_xticks(x)
ax2.set_xticklabels(months, fontsize=8)
ax2.set_ylabel("Rainfall (mm)", **LABEL_FONT)
ax2.set_xlabel("Month", **LABEL_FONT)
ax2.legend(fontsize=8, framealpha=0.85)
ax2.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
ax2.tick_params(labelsize=7)

# Shade monsoon JJAS
for i, m in enumerate(months):
    if m in ["JUN","JUL","AUG","SEP"]:
        ax2.axvspan(i - 0.4, i + 0.4, alpha=0.12, color="#c0392b", zorder=1)
ax2.annotate("◄ Monsoon (JJAS) ►", xy=(5.5, mon_mean.max() * 0.9),
             fontsize=7.5, color="#c0392b", ha="center")

add_caption(fig,
    "Data: NASA POWER MERRA-2 (lat 23.81°, lon 90.41°) 2004–2023 | "
    "Error bars = ±1 SD | Red bar = 2017 outlier")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "map2_rainfall_variability.png"),
            dpi=DPI, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "map2_rainfall_variability.pdf"),
            bbox_inches="tight")
plt.close(fig)
print("  → map2_rainfall_variability.png saved")


# %% ============================================================
# MAP 3 — Flood Hazard Classes (4-class raster)
# TWI=35%, Elevation=30%, Slope=20%, CN=15%
# ============================================================
print("Generating Map 3 — Flood hazard classification ...")

hazard, extent_h, crs_h = read_raster(HAZARD_CLASS, nodata=255)

# Class definitions: 1=Low, 2=Medium, 3=High, 4=Very High
hazard_cmap = mcolors.ListedColormap(
    ["#2ecc71", "#f39c12", "#e74c3c", "#6c0a0a"]
)
hazard_norm = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5],
                                    hazard_cmap.N)
CLASS_LABELS = ["Low", "Medium", "High", "Very High"]
CLASS_AREAS  = [1.4, 29.4, 40.4, 26.6]          # % from Phase 2

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
im = ax.imshow(
    hazard, cmap=hazard_cmap, norm=hazard_norm,
    extent=extent_h, interpolation="nearest",
    zorder=2,
)

# Overlay ward boundaries
wards_utm = gpd.read_file(WARDS)
wards_utm.plot(ax=ax, facecolor="none", edgecolor="#333333",
               linewidth=0.25, zorder=3)
boundary_utm = gpd.read_file(BOUNDARY_UTM)
boundary_utm.plot(ax=ax, facecolor="none", edgecolor="#1a1a2e",
                  linewidth=1.5, zorder=4)

# Custom legend with area %
patches = [
    mpatches.Patch(
        color=hazard_cmap.colors[i],
        label=f"{CLASS_LABELS[i]}  ({CLASS_AREAS[i]}%)"
    )
    for i in range(4)
]
ax.legend(handles=patches, title="Flood hazard class",
          title_fontsize=8, fontsize=8,
          loc="lower right", framealpha=0.9)

add_north_arrow(ax)
add_scalebar(ax)
ax.set_title(
    "Map 3 — Flood Hazard Classification, Dhaka Metropolitan Region",
    **TITLE_FONT, pad=12,
)
ax.set_xlabel("Easting (m)", **LABEL_FONT)
ax.set_ylabel("Northing (m)", **LABEL_FONT)
ax.tick_params(labelsize=7)
add_caption(fig,
    "Flood Hazard Index: TWI 35% | Elevation 30% | Slope 20% | "
    "SCS Curve Number 15%  |  Source: SRTM 30m, ESA WorldCover 2021")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "map3_flood_hazard_class.png"),
            dpi=DPI, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "map3_flood_hazard_class.pdf"),
            bbox_inches="tight")
plt.close(fig)
print("  → map3_flood_hazard_class.png saved")


# %% ============================================================
# MAP 4 — Continuous Risk Index (0–1)
# ============================================================
print("Generating Map 4 — Risk index (continuous) ...")

risk, extent_r, crs_r = read_raster(RISK_INDEX, nodata=-9999)

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
im = ax.imshow(
    risk, cmap="YlOrRd",
    vmin=0, vmax=1,
    extent=extent_r, interpolation="bilinear",
    zorder=2,
)

wards_utm = gpd.read_file(WARDS)
wards_utm.plot(ax=ax, facecolor="none", edgecolor="#333333",
               linewidth=0.25, zorder=3)
boundary_utm = gpd.read_file(BOUNDARY_UTM)
boundary_utm.plot(ax=ax, facecolor="none", edgecolor="#1a1a2e",
                  linewidth=1.5, zorder=4)

cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, shrink=0.7)
cb.set_label("Risk index (0 = no risk, 1 = very high risk)", fontsize=8)
cb.ax.tick_params(labelsize=7)

add_north_arrow(ax)
add_scalebar(ax)
ax.set_title(
    "Map 4 — Composite Flood Risk Index, Dhaka Metropolitan Region",
    **TITLE_FONT, pad=12,
)
ax.set_xlabel("Easting (m)", **LABEL_FONT)
ax.set_ylabel("Northing (m)", **LABEL_FONT)
ax.tick_params(labelsize=7)
add_caption(fig,
    "Risk = Hazard × Exposure × Vulnerability (UNDRR framework)  |  "
    "Source: Phases 2 & 3 outputs | WorldPop 2020, ESA WorldCover 2021")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "map4_risk_index.png"),
            dpi=DPI, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "map4_risk_index.pdf"),
            bbox_inches="tight")
plt.close(fig)
print("  → map4_risk_index.png saved")


# %% ============================================================
# MAP 5 — Ward-Level Risk (choropleth, 5-class)
# With top-10 wards labelled
# ============================================================
print("Generating Map 5 — Ward-level risk choropleth ...")

wards_risk = gpd.read_file(WARD_RISK_SHP)   # has risk_score, risk_class, NAME_4

# Column name safety (shapefile 10-char limit)
# Try common variants produced by phase3 script
score_col = None
for candidate in ["risk_score", "risk_scor", "mean_risk"]:
    if candidate in wards_risk.columns:
        score_col = candidate
        break
if score_col is None:
    # Fallback: use first numeric column after geometry
    score_col = [c for c in wards_risk.columns
                 if wards_risk[c].dtype in [float, "float64"]
                 and c != "geometry"][0]

# Risk class colours matching phase3: 0=No Risk, 1=Low, 2=Med, 3=High, 4=VHigh
risk_colors = {
    0: "#d5e8d4",   # No Risk — very light green
    1: "#82b366",   # Low — green
    2: "#ffe599",   # Medium — yellow
    3: "#f0a30a",   # High — orange
    4: "#ae1c28",   # Very High — dark red
}
risk_labels = {
    0: "No risk (0.4%)",
    1: "Low (2.2%)",
    2: "Medium (4.0%)",
    3: "High (8.6%)",
    4: "Very high (84.9%)",
}

class_col = None
for candidate in ["risk_class", "risk_clas", "riskclass"]:
    if candidate in wards_risk.columns:
        class_col = candidate
        break

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

if class_col:
    for cls, color in risk_colors.items():
        subset = wards_risk[wards_risk[class_col] == cls]
        if not subset.empty:
            subset.plot(ax=ax, facecolor=color, edgecolor="#555555",
                        linewidth=0.3, zorder=2)
else:
    # Fallback: continuous choropleth on score
    wards_risk.plot(
        column=score_col, ax=ax, cmap="YlOrRd",
        edgecolor="#555555", linewidth=0.3, zorder=2,
        legend=True,
        legend_kwds={"label": "Risk score", "shrink": 0.6},
    )

boundary_utm = gpd.read_file(BOUNDARY_UTM)
boundary_utm.plot(ax=ax, facecolor="none", edgecolor="#1a1a2e",
                  linewidth=1.8, zorder=4)

# Label top-10 wards by risk score
name_col = None
for candidate in ["NAME_4", "name_4", "NAME4", "ward_name"]:
    if candidate in wards_risk.columns:
        name_col = candidate
        break

if name_col and score_col:
    top10 = wards_risk.nlargest(10, score_col).copy()
    top10["centroid"] = top10.geometry.centroid
    for _, row in top10.iterrows():
        name = str(row[name_col])[:12]   # truncate long names
        ax.annotate(
            name,
            xy=(row["centroid"].x, row["centroid"].y),
            fontsize=5.5, color="#1a1a2e", ha="center",
            fontweight="bold", zorder=5,
        )

# Legend
patches = [
    mpatches.Patch(color=risk_colors[k], label=risk_labels[k],
                   edgecolor="#555555", linewidth=0.5)
    for k in sorted(risk_colors)
]
ax.legend(handles=patches, title="Flood risk class\n(% of total pop.)",
          title_fontsize=8, fontsize=8,
          loc="lower right", framealpha=0.9)

add_north_arrow(ax)
add_scalebar(ax)
ax.set_title(
    "Map 5 — Ward-Level Flood Risk Classification, Dhaka Metropolitan Region",
    **TITLE_FONT, pad=12,
)
ax.set_xlabel("Easting (m)", **LABEL_FONT)
ax.set_ylabel("Northing (m)", **LABEL_FONT)
ax.tick_params(labelsize=7)
add_caption(fig,
    "Classification: two-tier quantile (zero pixels = No Risk; "
    "non-zero at 25/50/75th percentile)  |  "
    "Source: Phase 3 outputs, GADM v4.1 ward boundaries")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "map5_ward_risk.png"),
            dpi=DPI, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "map5_ward_risk.pdf"),
            bbox_inches="tight")
plt.close(fig)
print("  → map5_ward_risk.png saved")


# %% ============================================================
# Done
# ============================================================
print("\n✓ All 5 maps saved to:", OUT)
print("  map1_study_area.png")
print("  map2_rainfall_variability.png")
print("  map3_flood_hazard_class.png")
print("  map4_risk_index.png")
print("  map5_ward_risk.png")
print("\nNext: run scripts/phase4_stats.py for tables and report figures.")
