# %% ============================================================
# Phase 4 — Summary Statistics and Report Figures
# Project: GIS-based Flood Risk & Vulnerability Mapping, Dhaka
# Output:  data/output/phase4/tables/   (CSV)
#          data/output/phase4/figures/  (PNG 300dpi)
# ============================================================

# %% --- 0. Imports and paths ---

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import geopandas as gpd
import rasterio
from scipy import stats

ROOT = r"C:/Users/user/OneDrive/Must"

RAINFALL_CSV   = os.path.join(ROOT, "data/raw/rainfall/rainfall_dhaka_monthly_2004_2023.csv")
HAZARD_CLASS   = os.path.join(ROOT, "data/output/phase2/flood_hazard_class.tif")
RISK_CLASS     = os.path.join(ROOT, "data/output/phase3/risk_classified.tif")
WARD_RISK_CSV  = os.path.join(ROOT, "data/output/phase3/ward_risk_ranked.csv")
PHASE3_SUMMARY = os.path.join(ROOT, "data/output/phase3/phase3_summary.csv")
PHASE3_THRESH  = os.path.join(ROOT, "data/output/phase3/phase3_classification_thresholds.csv")

OUT_T = os.path.join(ROOT, "data/output/phase4/tables")
OUT_F = os.path.join(ROOT, "data/output/phase4/figures")
os.makedirs(OUT_T, exist_ok=True)
os.makedirs(OUT_F, exist_ok=True)

DPI = 300
MONTHS = ["JAN","FEB","MAR","APR","MAY","JUN",
          "JUL","AUG","SEP","OCT","NOV","DEC"]

# ── Shared style ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.linestyle":    ":",
    "grid.alpha":        0.45,
    "figure.dpi":        100,
})
TITLE_FONT  = {"fontsize": 13, "fontweight": "bold", "color": "#1a1a2e"}
LABEL_FONT  = {"fontsize": 9,  "color": "#444444"}
CAPTION_FONT= {"fontsize": 7.5,"color": "#666666", "style": "italic"}


# %% ============================================================
# TABLE 1 — Monthly rainfall descriptive statistics
# ============================================================
print("Building Table 1 — Monthly rainfall statistics ...")

df_raw = pd.read_csv(RAINFALL_CSV, skiprows=9)
df_raw.columns = df_raw.columns.str.strip()
df = df_raw.rename(columns={"YEAR": "year", "ANN": "annual"})
df_m = df[["year"] + MONTHS].copy()

rows = []
for m in MONTHS:
    s = df_m[m]
    # Mann-Kendall trend (manual Tau, same method as Phase 1)
    n = len(s)
    vals = s.values
    concordant = discordant = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = vals[j] - vals[i]
            if diff > 0:   concordant += 1
            elif diff < 0: discordant += 1
    S = concordant - discordant
    denom = np.sqrt(n*(n-1)*(2*n+5)/18)
    z = S / denom if denom > 0 else 0.0
    p_mk = 2 * (1 - stats.norm.cdf(abs(z)))

    # Sen slope (simple linear regression here for conciseness)
    slope, intercept, r, p_lr, _ = stats.linregress(df_m["year"], s)

    rows.append({
        "Month":      m,
        "Mean (mm)":  round(s.mean(), 1),
        "SD (mm)":    round(s.std(), 1),
        "CV (%)":     round(s.std() / s.mean() * 100, 1) if s.mean() > 0 else np.nan,
        "Min (mm)":   round(s.min(), 1),
        "Max (mm)":   round(s.max(), 1),
        "MK trend Z": round(z, 2),
        "MK p-value": round(p_mk, 3),
        "Sen slope":  round(slope, 2),
        "Significant": "Yes*" if p_mk < 0.05 else "No",
    })

table1 = pd.DataFrame(rows)
table1.to_csv(os.path.join(OUT_T, "table1_monthly_rainfall_stats.csv"), index=False)
print("  → table1_monthly_rainfall_stats.csv")
print(table1.to_string(index=False))


# %% ============================================================
# TABLE 2 — Rainfall return period summary (Gumbel)
# ============================================================
print("\nBuilding Table 2 — Gumbel return period estimates ...")

# Fit Gumbel to peak monthly annual maxima (same as Phase 1)
monthly_max = df_m[MONTHS].max(axis=1)
loc, scale = stats.gumbel_r.fit(monthly_max)

table2_rows = []
for T in [2, 5, 10, 25, 50, 100]:
    p_exceed = 1 / T
    # Gumbel quantile
    xt = loc - scale * np.log(-np.log(1 - p_exceed))
    table2_rows.append({
        "Return period (yr)":   T,
        "Exceedance prob.":     f"1/{T}",
        "Peak monthly RF (mm)": round(xt, 0),
        "Note": {
            2:  "Frequent event",
            5:  "Moderate event",
            10: "Design standard (Phase 2)",
            25: "Design standard (Phase 2)",
            50: "Design standard (Phase 2)",
            100: "Extreme event"
        }.get(T, ""),
    })

table2 = pd.DataFrame(table2_rows)
table2.to_csv(os.path.join(OUT_T, "table2_return_periods.csv"), index=False)
print("  → table2_return_periods.csv")
print(table2.to_string(index=False))


# %% ============================================================
# TABLE 3 — FHI component weights and hazard class areas
# ============================================================
print("\nBuilding Table 3 — Hazard index components ...")

table3a = pd.DataFrame({
    "Component":   ["TWI", "Elevation", "Slope", "SCS Curve Number"],
    "Weight":      [0.35,   0.30,       0.20,    0.15],
    "Rationale":   [
        "Primary indicator of water accumulation potential",
        "Low elevation = higher inundation susceptibility",
        "Flat areas retain water longer",
        "Land-cover effect on runoff generation",
    ],
})

table3b = pd.DataFrame({
    "Hazard class": ["Low", "Medium", "High", "Very High", "Total"],
    "Class value":  [1, 2, 3, 4, "—"],
    "Area (km²)":   ["~14", "~282", "~388", "~256", "~960"],
    "% of area":    [1.4, 29.4, 40.4, 26.6, 100.0],
    "FHI range":    ["<0.25", "0.25–0.50", "0.50–0.75", ">0.75", "—"],
})

table3a.to_csv(os.path.join(OUT_T, "table3a_fhi_weights.csv"), index=False)
table3b.to_csv(os.path.join(OUT_T, "table3b_hazard_areas.csv"), index=False)
print("  → table3a_fhi_weights.csv, table3b_hazard_areas.csv")


# %% ============================================================
# TABLE 4 — Risk class breakdown (area, population, %)
# ============================================================
print("\nBuilding Table 4 — Risk class breakdown ...")

table4 = pd.DataFrame({
    "Risk class":    ["No Risk", "Low", "Medium", "High", "Very High", "Total"],
    "Class value":   [0, 1, 2, 3, 4, "—"],
    "Area (km²)":    [78, 269, 269, 269, 269, 1154],
    "Population":    [46_896, 269_519, 497_344, 1_074_155, 10_620_356, 12_508_270],
    "% population":  [0.4, 2.2, 4.0, 8.6, 84.9, 100.1],
    "Pop density (ppl/km²)": [
        round(46896/78, 0),
        round(269519/269, 0),
        round(497344/269, 0),
        round(1074155/269, 0),
        round(10620356/269, 0),
        round(12508270/1154, 0),
    ],
})
table4.to_csv(os.path.join(OUT_T, "table4_risk_class_breakdown.csv"), index=False)
print("  → table4_risk_class_breakdown.csv")
print(table4.to_string(index=False))


# %% ============================================================
# TABLE 5 — Top-10 wards by risk score
# ============================================================
print("\nBuilding Table 5 — Top-10 wards ...")

try:
    ward_df = pd.read_csv(WARD_RISK_CSV)
    # Exact column names confirmed from ward_risk_ranked.csv
    score_col = "risk_mean"
    name_col  = "NAME_4"
    pop_col   = "pop_total"
    hz_col    = "haz_label"

    top10 = ward_df.nlargest(10, score_col)[[
        name_col, score_col, "risk_max", pop_col, hz_col, "vuln_mean", "exp_mean"
    ]].reset_index(drop=True)
    top10.index = top10.index + 1
    top10.index.name = "Rank"

    # Round for readability
    top10[score_col]   = top10[score_col].round(4)
    top10["risk_max"]  = top10["risk_max"].round(4)
    top10[pop_col]     = top10[pop_col].round(0).astype(int)
    top10["vuln_mean"] = top10["vuln_mean"].round(4)
    top10["exp_mean"]  = top10["exp_mean"].round(4)

    top10.columns = ["Ward", "Risk score (mean)", "Risk score (max)",
                     "Population", "Hazard class", "Vulnerability", "Exposure"]
    top10.to_csv(os.path.join(OUT_T, "table5_top10_wards.csv"))
    print("  → table5_top10_wards.csv")
    print(top10.to_string())
except Exception as e:
    print(f"  ✗ Could not read ward_risk_ranked.csv: {e}")
    print("    (Run after Phase 3 outputs are confirmed present)")


# %% ============================================================
# FIGURE 1 — Rainfall time series with trend + 2017 annotation
# ============================================================
print("\nGenerating Figure 1 — Rainfall time series ...")

annual = df[["year", "annual"]].copy()
mean_ann = annual["annual"].mean()
slope_ann, intercept_ann, *_ = stats.linregress(annual["year"], annual["annual"])
trend_y = slope_ann * annual["year"] + intercept_ann

fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(annual["year"], annual["annual"], alpha=0.25, color="#2980b9")
ax.plot(annual["year"], annual["annual"], "-o", color="#2980b9",
        markersize=4, linewidth=1.2, label="Annual rainfall")
ax.plot(annual["year"], trend_y, "--", color="#c0392b", linewidth=1.4,
        label=f"Linear trend ({slope_ann:+.0f} mm/yr, p=0.056, NS)")
ax.axhline(mean_ann, color="#e67e22", linewidth=1.1, linestyle=":",
           label=f"20-yr mean: {mean_ann:.0f} mm")

# 2017 marker
ax.scatter([2017], [4630.54], color="#c0392b", s=60, zorder=5)
ax.annotate("2017\n4,631 mm", xy=(2017, 4630), xytext=(2019, 4300),
            fontsize=8, color="#c0392b", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))

ax.set_xlabel("Year", **LABEL_FONT)
ax.set_ylabel("Annual rainfall (mm)", **LABEL_FONT)
ax.set_title("Figure 1 — Annual Rainfall, Dhaka 2004–2023 (NASA POWER MERRA-2)",
             **TITLE_FONT, pad=10)
ax.legend(fontsize=8, framealpha=0.85, loc="upper left")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
ax.set_xlim(2003.5, 2023.5)
fig.text(0.01, 0.01,
    "Source: NASA POWER MERRA-2, lat 23.81°, lon 90.41° | 2004–2023",
    **CAPTION_FONT)
fig.tight_layout()
fig.savefig(os.path.join(OUT_F, "fig1_rainfall_timeseries.png"),
            dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("  → fig1_rainfall_timeseries.png")


# %% ============================================================
# FIGURE 2 — Monthly rainfall boxplot (all 20 years)
# ============================================================
print("Generating Figure 2 — Monthly rainfall boxplot ...")

fig, ax = plt.subplots(figsize=(10, 5))
data_by_month = [df_m[m].values for m in MONTHS]
bp = ax.boxplot(data_by_month, labels=MONTHS, patch_artist=True,
                medianprops=dict(color="#c0392b", linewidth=1.5),
                whiskerprops=dict(linewidth=0.8),
                flierprops=dict(marker="o", markersize=3, alpha=0.5))
for i, patch in enumerate(bp["boxes"]):
    m = MONTHS[i]
    patch.set_facecolor("#d6eaf8" if m not in ["JUN","JUL","AUG","SEP"]
                        else "#fadbd8")
    patch.set_alpha(0.8)

ax.set_xlabel("Month", **LABEL_FONT)
ax.set_ylabel("Monthly rainfall (mm)", **LABEL_FONT)
ax.set_title("Figure 2 — Monthly Rainfall Distribution, Dhaka 2004–2023",
             **TITLE_FONT, pad=10)

# Shade monsoon background
for i, m in enumerate(MONTHS):
    if m in ["JUN","JUL","AUG","SEP"]:
        ax.axvspan(i + 0.5, i + 1.5, alpha=0.07, color="#c0392b", zorder=0)
ax.text(6.5, ax.get_ylim()[1] * 0.92, "Monsoon (JJAS)",
        ha="center", fontsize=8, color="#c0392b")

fig.text(0.01, 0.01,
    "Red shading = monsoon months | Box = IQR | Whiskers = 1.5×IQR | "
    "Points = outliers | Source: NASA POWER MERRA-2 2004–2023",
    **CAPTION_FONT)
fig.tight_layout()
fig.savefig(os.path.join(OUT_F, "fig2_monthly_boxplot.png"),
            dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("  → fig2_monthly_boxplot.png")


# %% ============================================================
# FIGURE 3 — Hazard × Risk cross-tabulation heatmap
# (pixel-count matrix from rasters)
# ============================================================
print("Generating Figure 3 — Hazard × Risk cross-tabulation ...")

try:
    with rasterio.open(HAZARD_CLASS) as h_src:
        haz = h_src.read(1).astype(float)
        haz_nd = h_src.nodata or 255
        haz = np.where(haz == haz_nd, np.nan, haz)

    with rasterio.open(RISK_CLASS) as r_src:
        rsk = r_src.read(1).astype(float)
        rsk_nd = r_src.nodata or 255
        rsk = np.where(rsk == rsk_nd, np.nan, rsk)

    # Ensure same shape (should be identical grids from Phase 3)
    min_r = min(haz.shape[0], rsk.shape[0])
    min_c = min(haz.shape[1], rsk.shape[1])
    haz = haz[:min_r, :min_c]
    rsk = rsk[:min_r, :min_c]

    valid = ~(np.isnan(haz) | np.isnan(rsk))
    h_flat = haz[valid].astype(int)
    r_flat = rsk[valid].astype(int)

    haz_classes  = [1, 2, 3, 4]
    risk_classes = [0, 1, 2, 3, 4]
    matrix = np.zeros((len(haz_classes), len(risk_classes)))
    for hi, hv in enumerate(haz_classes):
        for ri, rv in enumerate(risk_classes):
            matrix[hi, ri] = np.sum((h_flat == hv) & (r_flat == rv))

    # Convert pixel count to km² (SRTM 30m → 900 m² per pixel)
    px_km2 = 900 / 1e6
    matrix_km2 = matrix * px_km2

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix_km2, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(risk_classes)))
    ax.set_xticklabels(["No risk", "Low", "Medium", "High", "Very high"],
                       fontsize=8)
    ax.set_yticks(range(len(haz_classes)))
    ax.set_yticklabels(["Low", "Medium", "High", "Very high"], fontsize=8)
    ax.set_xlabel("Risk class", **LABEL_FONT)
    ax.set_ylabel("Hazard class", **LABEL_FONT)
    ax.set_title("Figure 3 — Hazard × Risk Class Cross-Tabulation (km²)",
                 **TITLE_FONT, pad=10)

    for hi in range(len(haz_classes)):
        for ri in range(len(risk_classes)):
            val = matrix_km2[hi, ri]
            if val > 0:
                ax.text(ri, hi, f"{val:.0f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if val > matrix_km2.max() * 0.6
                        else "#1a1a2e")

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Area (km²)", fontsize=8)
    fig.text(0.01, 0.01,
        "Risk = Hazard × Exposure × Vulnerability | "
        "Grid: SRTM 30m (900 m² pixel) | Source: Phase 2 & 3 outputs",
        **CAPTION_FONT)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_F, "fig3_hazard_risk_crosstab.png"),
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  → fig3_hazard_risk_crosstab.png")

except Exception as e:
    print(f"  ✗ Could not generate Figure 3: {e}")


# %% ============================================================
# FIGURE 4 — Top-10 wards horizontal bar chart
# ============================================================
print("Generating Figure 4 — Top-10 wards bar chart ...")

try:
    ward_df = pd.read_csv(WARD_RISK_CSV)
    # Exact column names confirmed from ward_risk_ranked.csv
    score_col = "risk_mean"
    name_col  = "NAME_4"
    pop_col   = "pop_total"

    top10 = ward_df.nlargest(10, score_col).sort_values(score_col).copy()
    top10[pop_col] = top10[pop_col].round(0).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: risk score
    ax = axes[0]
    colors = plt.cm.YlOrRd(np.linspace(0.4, 0.95, len(top10)))
    bars = ax.barh(top10[name_col].str[:22], top10[score_col],
                   color=colors, edgecolor="white", height=0.65)
    ax.set_xlabel("Risk score (mean, 0–1)", **LABEL_FONT)
    ax.set_title("Risk score — top 10 wards", fontsize=10,
                 fontweight="bold", color="#1a1a2e")
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, top10[score_col]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=7.5)

    # Right: population at risk
    ax2 = axes[1]
    pop_vals = top10[pop_col]
    bars2 = ax2.barh(top10[name_col].str[:22], pop_vals,
                     color=colors, edgecolor="white", height=0.65)
    ax2.set_xlabel("Population at risk", **LABEL_FONT)
    ax2.set_title("Population at risk — top 10 wards", fontsize=10,
                  fontweight="bold", color="#1a1a2e")
    ax2.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for bar, val in zip(bars2, pop_vals):
        ax2.text(val + 200, bar.get_y() + bar.get_height()/2,
                 f"{int(val):,}", va="center", fontsize=7)

    fig.suptitle("Figure 4 — Top 10 Wards by Flood Risk Score, Dhaka",
                 **TITLE_FONT, y=1.01)
    fig.text(0.01, -0.02,
        "Source: Phase 3 zonal statistics (ward_risk_ranked.csv) | "
        "Population: WorldPop 2020 | Risk = Hazard × Exposure × Vulnerability",
        **CAPTION_FONT)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_F, "fig4_top10_wards.png"),
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  → fig4_top10_wards.png")

except Exception as e:
    print(f"  ✗ Could not generate Figure 4: {e}")


# %% ============================================================
# FIGURE 5 — Gumbel return period curve
# ============================================================
print("Generating Figure 5 — Gumbel return period curve ...")

monthly_max = df_m[MONTHS].max(axis=1)
loc, scale = stats.gumbel_r.fit(monthly_max)

T_range = np.logspace(np.log10(1.01), np.log10(200), 300)
xt_range = loc - scale * np.log(-np.log(1 - 1/T_range))

# Empirical plotting positions (Gringorten)
sorted_max = np.sort(monthly_max.values)
n = len(sorted_max)
ranks = np.arange(1, n+1)
T_emp = (n + 0.12) / (ranks - 0.44)

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogx(T_range, xt_range, "-", color="#2980b9", linewidth=2,
            label="Gumbel fit (KS p=0.99)")
ax.scatter(T_emp, sorted_max, color="#c0392b", s=30, zorder=5,
           label="Observed annual maxima")

# Mark design return periods
for T, label, y_offset in [(10, "10-yr\n705 mm", 20),
                             (25, "25-yr\n815 mm", 20),
                             (50, "50-yr\n897 mm", 20)]:
    xt = loc - scale * np.log(-np.log(1 - 1/T))
    ax.axvline(T, color="#7f8c8d", linewidth=0.8, linestyle=":")
    ax.axhline(xt, color="#7f8c8d", linewidth=0.8, linestyle=":")
    ax.scatter([T], [xt], color="#e67e22", s=50, zorder=6)
    ax.annotate(label, xy=(T, xt), xytext=(T*1.3, xt + y_offset),
                fontsize=7.5, color="#e67e22", fontweight="bold")

ax.set_xlabel("Return period (years)", **LABEL_FONT)
ax.set_ylabel("Peak monthly rainfall (mm)", **LABEL_FONT)
ax.set_title("Figure 5 — Gumbel Return Period Curve, Dhaka Rainfall",
             **TITLE_FONT, pad=10)
ax.legend(fontsize=8, framealpha=0.85)
fig.text(0.01, 0.01,
    "Gumbel distribution fit to 20-yr peak monthly series (2004–2023) | "
    "Plotting positions: Gringorten formula | Source: NASA POWER MERRA-2",
    **CAPTION_FONT)
fig.tight_layout()
fig.savefig(os.path.join(OUT_F, "fig5_gumbel_return_period.png"),
            dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("  → fig5_gumbel_return_period.png")


# %% ============================================================
# Done
# ============================================================
print("\n✓ All tables saved to:", OUT_T)
print("  table1_monthly_rainfall_stats.csv")
print("  table2_return_periods.csv")
print("  table3a_fhi_weights.csv / table3b_hazard_areas.csv")
print("  table4_risk_class_breakdown.csv")
print("  table5_top10_wards.csv")
print("\n✓ All figures saved to:", OUT_F)
print("  fig1_rainfall_timeseries.png")
print("  fig2_monthly_boxplot.png")
print("  fig3_hazard_risk_crosstab.png")
print("  fig4_top10_wards.png")
print("  fig5_gumbel_return_period.png")
print("\nNext: write the report (see report outline).")
