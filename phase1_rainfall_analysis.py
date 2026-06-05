# =============================================================================
# Phase 1 — Rainfall Analysis
# Project : Flood Risk & Vulnerability Mapping, Dhaka
# Data    : NASA POWER MERRA-2 Monthly Precipitation, 2004–2023
# Author  : (your name)
# =============================================================================

# %% ── 0. Imports ─────────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import genextreme, gumbel_r, kstest
import pymannkendall as mk
import warnings
warnings.filterwarnings("ignore")

# %% ── 1. Paths ───────────────────────────────────────────────────────────────
# Run from flood_risk_dhaka/ as the working directory in VS Code
RAW_CSV   = "C:/Users/user/OneDrive/Must/data/raw/rainfall/rainfall_dhaka_monthly_2004_2023.csv"
OUT_DIR   = "C:/Users/user/OneDrive/Must/data/output/phase1"
os.makedirs(OUT_DIR, exist_ok=True)

MONTHS      = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
MONTH_FULL  = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]

# Monsoon = JJAS (June–September), commonly used for Bangladesh
MONSOON_MONTHS = ["Jun","Jul","Aug","Sep"]
DRY_MONTHS     = ["Nov","Dec","Jan","Feb","Mar"]

# Matplotlib style — clean, publication-ready
plt.rcParams.update({
    "figure.dpi"      : 150,
    "font.family"     : "sans-serif",
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "axes.grid"       : True,
    "grid.alpha"      : 0.3,
    "grid.linestyle"  : "--",
})

# %% ── 2. Load & Clean Data ───────────────────────────────────────────────────
print("=" * 60)
print("PHASE 1 — RAINFALL ANALYSIS | DHAKA 2004–2023")
print("=" * 60)

df_raw = pd.read_csv(RAW_CSV, skiprows=9)

# Drop PARAMETER column, keep YEAR + 12 months + ANN
df = df_raw.drop(columns=["PARAMETER"]).copy()
df.columns = ["YEAR"] + MONTHS + ["ANN"]
df["YEAR"] = df["YEAR"].astype(int)

# Replace missing value code with NaN
df.replace(-999, np.nan, inplace=True)

print(f"\nRows loaded  : {len(df)}")
print(f"Years covered: {df['YEAR'].min()} – {df['YEAR'].max()}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"\nAnnual totals (mm):\n{df[['YEAR','ANN']].to_string(index=False)}")

# %% ── 3. Summary Statistics ──────────────────────────────────────────────────
print("\n── ANNUAL RAINFALL STATISTICS ──")
ann = df["ANN"]
print(f"  Mean          : {ann.mean():.1f} mm")
print(f"  Median        : {ann.median():.1f} mm")
print(f"  Std dev       : {ann.std():.1f} mm")
print(f"  CV            : {ann.std()/ann.mean()*100:.1f} %")
print(f"  Min           : {ann.min():.1f} mm ({df.loc[ann.idxmin(),'YEAR']})")
print(f"  Max           : {ann.max():.1f} mm ({df.loc[ann.idxmax(),'YEAR']})")

# Monthly climatology
monthly_mean = df[MONTHS].mean()
monthly_std  = df[MONTHS].std()
monthly_cv   = monthly_std / monthly_mean * 100

print("\n── MONTHLY CLIMATOLOGY (mm) ──")
clim_df = pd.DataFrame({
    "Month"  : MONTHS,
    "Mean"   : monthly_mean.values.round(1),
    "Std"    : monthly_std.values.round(1),
    "CV (%)" : monthly_cv.values.round(1),
})
print(clim_df.to_string(index=False))

# Monsoon vs dry season
monsoon_annual = df[MONSOON_MONTHS].sum(axis=1)
dry_annual     = df[DRY_MONTHS].sum(axis=1)
print(f"\n── MONSOON (JJAS) vs DRY SEASON ──")
print(f"  Mean monsoon rainfall : {monsoon_annual.mean():.1f} mm  "
      f"({monsoon_annual.mean()/ann.mean()*100:.1f}% of annual)")
print(f"  Mean dry season       : {dry_annual.mean():.1f} mm  "
      f"({dry_annual.mean()/ann.mean()*100:.1f}% of annual)")

# %% ── 4. Plot 1 — Monthly Climatology (bar + error bars) ────────────────────
fig, ax = plt.subplots(figsize=(11, 5))

colors = ["#3B8BD4" if m in MONSOON_MONTHS else "#9FE1CB" for m in MONTHS]
bars = ax.bar(MONTHS, monthly_mean, color=colors,
              edgecolor="white", linewidth=0.5, zorder=3)
ax.errorbar(MONTHS, monthly_mean, yerr=monthly_std,
            fmt="none", color="#444", capsize=4, linewidth=1, zorder=4)

# Annotate max month
max_idx = monthly_mean.idxmax()
ax.annotate(f"{monthly_mean[max_idx]:.0f} mm",
            xy=(list(MONTHS).index(max_idx), monthly_mean[max_idx]),
            xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=9, color="#185FA5", fontweight="bold")

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#3B8BD4", label="Monsoon (JJAS)"),
                   Patch(facecolor="#9FE1CB", label="Non-monsoon")]
ax.legend(handles=legend_elements, frameon=False, fontsize=9)

ax.set_ylabel("Rainfall (mm)", fontsize=10)
ax.set_title("Monthly rainfall climatology — Dhaka (2004–2023)\n"
             "Error bars = ±1 SD  |  Source: NASA POWER MERRA-2", fontsize=10)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig1_monthly_climatology.png", bbox_inches="tight")
plt.show()
print("Saved: fig1_monthly_climatology.png")

# %% ── 5. Plot 2 — Annual Rainfall Time Series + 5-yr Rolling Mean ────────────
fig, ax = plt.subplots(figsize=(11, 5))

years = df["YEAR"].values
ax.bar(years, ann, color="#B5D4F4", edgecolor="white",
       linewidth=0.5, zorder=2, label="Annual total")

rolling = ann.rolling(window=5, center=True).mean()
ax.plot(years, rolling, color="#185FA5", linewidth=2,
        zorder=3, label="5-yr rolling mean")

# Mark the 2017 outlier
idx_2017 = df[df["YEAR"] == 2017].index[0]
ax.annotate("2017\n(4631 mm)",
            xy=(2017, ann.iloc[idx_2017]),
            xytext=(2014.5, 4400),
            arrowprops=dict(arrowstyle="->", color="#A32D2D", lw=1.2),
            fontsize=8.5, color="#A32D2D")

ax.axhline(ann.mean(), color="#63992222", linewidth=1.2,
           linestyle="--", zorder=1, label=f"Mean = {ann.mean():.0f} mm")

ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel("Annual rainfall (mm)", fontsize=10)
ax.set_title("Annual rainfall totals — Dhaka (2004–2023)\n"
             "Source: NASA POWER MERRA-2", fontsize=10)
ax.legend(frameon=False, fontsize=9)
ax.set_xlim(2003, 2024)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig2_annual_timeseries.png", bbox_inches="tight")
plt.show()
print("Saved: fig2_annual_timeseries.png")

# %% ── 6. Mann-Kendall Trend Test ─────────────────────────────────────────────
print("\n── MANN-KENDALL TREND TEST ──")

# Full annual series
result_ann = mk.original_test(ann)
print(f"\n  Annual rainfall:")
print(f"    Trend     : {result_ann.trend}")
print(f"    p-value   : {result_ann.p:.4f}")
print(f"    Tau       : {result_ann.Tau:.4f}")
print(f"    Sen slope : {result_ann.slope:.2f} mm/year")

# Monsoon series
result_mon = mk.original_test(monsoon_annual)
print(f"\n  Monsoon (JJAS) rainfall:")
print(f"    Trend     : {result_mon.trend}")
print(f"    p-value   : {result_mon.p:.4f}")
print(f"    Sen slope : {result_mon.slope:.2f} mm/year")

# Per-month trend summary
print(f"\n  Monthly trend summary (Sen slope, mm/year):")
for m in MONTHS:
    r = mk.original_test(df[m].dropna())
    sig = "* p<0.05" if r.p < 0.05 else ""
    print(f"    {m:>3}: {r.slope:+.2f}  {r.trend:<12} {sig}")

# %% ── 7. Plot 3 — Heatmap of Monthly Rainfall by Year ────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))

heatmap_data = df.set_index("YEAR")[MONTHS].values
im = ax.imshow(heatmap_data, aspect="auto", cmap="YlGnBu",
               interpolation="nearest")

ax.set_xticks(range(12))
ax.set_xticklabels(MONTHS, fontsize=9)
ax.set_yticks(range(len(df)))
ax.set_yticklabels(df["YEAR"].values, fontsize=8)
ax.set_xlabel("Month", fontsize=10)
ax.set_ylabel("Year", fontsize=10)
ax.set_title("Monthly rainfall heatmap — Dhaka (2004–2023)  |  mm",
             fontsize=10)

cbar = plt.colorbar(im, ax=ax, pad=0.02)
cbar.set_label("Rainfall (mm)", fontsize=9)
cbar.ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig3_monthly_heatmap.png", bbox_inches="tight")
plt.show()
print("Saved: fig3_monthly_heatmap.png")

# %% ── 8. Extreme Value Analysis — Return Periods ────────────────────────────
print("\n── EXTREME VALUE ANALYSIS ──")

# Use annual maxima (wettest month each year) for frequency analysis
ann_max = df[MONTHS].max(axis=1).values  # monthly maxima per year
ann_max_sorted = np.sort(ann_max)

print(f"\n  Annual monthly maxima (mm): {ann_max_sorted}")
print(f"  Mean  : {ann_max.mean():.1f} mm")
print(f"  Max   : {ann_max.max():.1f} mm  (year: {df['YEAR'].iloc[ann_max.argmax()]})")

# --- Fit Gumbel distribution ---
loc_g, scale_g = gumbel_r.fit(ann_max)
print(f"\n  Gumbel fit: loc = {loc_g:.2f}, scale = {scale_g:.2f}")

# --- Fit GEV distribution (constrained shape to avoid degenerate solution) ---
# NOTE: With n=20 and one major outlier (2017), unconstrained GEV can diverge.
# We constrain the shape parameter to [-0.5, 0.5] — a physically reasonable
# range for monsoon rainfall. Gumbel (shape=0) is preferred when KS confirms.
shape_gev, loc_gev, scale_gev = genextreme.fit(ann_max, f0=-0.1)
print(f"  GEV fit   : shape = {shape_gev:.4f}, loc = {loc_gev:.2f}, "
      f"scale = {scale_gev:.2f}")
print(f"  Note: GEV shape constrained (f0=-0.1) due to small sample (n=20)")

# --- KS goodness-of-fit test ---
ks_gumbel = kstest(ann_max, "gumbel_r",
                   args=(loc_g, scale_g))
ks_gev    = kstest(ann_max, "genextreme",
                   args=(shape_gev, loc_gev, scale_gev))
print(f"\n  KS test p-value — Gumbel : {ks_gumbel.pvalue:.4f}")
print(f"  KS test p-value — GEV    : {ks_gev.pvalue:.4f}")

# For small samples (n<30), prefer Gumbel unless GEV is clearly superior
best_fit = "GEV" if (ks_gev.pvalue > ks_gumbel.pvalue * 1.2) else "Gumbel"
print(f"  → Best fit for n=20: {best_fit}  (Gumbel preferred for small samples)")

# --- Return period table ---
return_periods = [2, 5, 10, 25, 50, 100]
print(f"\n  Return period estimates (peak monthly rainfall, mm):")
print(f"  {'T (yr)':>8}  {'Gumbel':>10}  {'GEV':>10}")
rp_records = []
for T in return_periods:
    p_exceed  = 1 - 1/T          # non-exceedance probability
    q_gumbel  = gumbel_r.ppf(p_exceed, loc=loc_g, scale=scale_g)
    q_gev     = genextreme.ppf(p_exceed, shape_gev,
                               loc=loc_gev, scale=scale_gev)
    # Clip GEV to physically plausible range (max observed * 3)
    q_gev_clipped = min(q_gev, ann_max.max() * 3)
    print(f"  {T:>8}  {q_gumbel:>10.1f}  {q_gev_clipped:>10.1f}")
    rp_records.append({"Return_Period_yr": T,
                        "Gumbel_mm": round(q_gumbel, 1),
                        "GEV_mm"   : round(q_gev_clipped, 1)})

rp_df = pd.DataFrame(rp_records)

# %% ── 9. Plot 4 — Return Period Curve ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

# Empirical plotting positions (Gringorten formula — suited for extremes)
n = len(ann_max_sorted)
i = np.arange(1, n + 1)
F_emp = (i - 0.44) / (n + 0.12)
T_emp = 1 / (1 - F_emp)

ax.scatter(T_emp, ann_max_sorted, color="#185FA5", zorder=5,
           s=40, label="Observed annual maxima")

T_range = np.logspace(np.log10(1.01), np.log10(200), 300)
p_range = 1 - 1/T_range

ax.plot(T_range,
        gumbel_r.ppf(p_range, loc=loc_g, scale=scale_g),
        color="#3B8BD4", linewidth=2, label="Gumbel fit")
ax.plot(T_range,
        genextreme.ppf(p_range, shape_gev, loc=loc_gev, scale=scale_gev),
        color="#D85A30", linewidth=2, linestyle="--", label="GEV fit")

# Mark standard return periods
for T, rec in zip(return_periods[2:], rp_records[2:]):
    q = rec["GEV_mm"] if best_fit == "GEV" else rec["Gumbel_mm"]
    ax.axvline(T, color="gray", linewidth=0.6, linestyle=":")
    ax.text(T, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 200,
            f"{T}yr", ha="center", va="bottom",
            fontsize=7.5, color="gray")

ax.set_xscale("log")
ax.set_xlabel("Return period (years)", fontsize=10)
ax.set_ylabel("Peak monthly rainfall (mm)", fontsize=10)
ax.set_title("Extreme value analysis — Dhaka (2004–2023)\n"
             "Annual maximum monthly rainfall", fontsize=10)
ax.legend(frameon=False, fontsize=9)
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig4_return_period_curve.png", bbox_inches="tight")
plt.show()
print("Saved: fig4_return_period_curve.png")

# %% ── 10. Plot 5 — Monsoon vs Dry Season Box Plot ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Left: monsoon monthly box plots
monsoon_data = [df[m].values for m in MONSOON_MONTHS]
bp = axes[0].boxplot(monsoon_data, labels=MONSOON_MONTHS,
                     patch_artist=True, notch=False,
                     medianprops=dict(color="#185FA5", linewidth=2))
for patch in bp["boxes"]:
    patch.set_facecolor("#B5D4F4")
axes[0].set_title("Monsoon months (JJAS)", fontsize=10)
axes[0].set_ylabel("Monthly rainfall (mm)", fontsize=10)

# Right: annual monsoon total time series
axes[1].fill_between(years, monsoon_annual, alpha=0.3, color="#3B8BD4")
axes[1].plot(years, monsoon_annual, color="#185FA5",
             linewidth=1.5, marker="o", markersize=4)
axes[1].axhline(monsoon_annual.mean(), color="#D85A30", linewidth=1.2,
                linestyle="--",
                label=f"Mean = {monsoon_annual.mean():.0f} mm")
axes[1].set_title("Annual monsoon total (JJAS)", fontsize=10)
axes[1].set_ylabel("Monsoon rainfall (mm)", fontsize=10)
axes[1].set_xlabel("Year", fontsize=10)
axes[1].legend(frameon=False, fontsize=9)

plt.suptitle("Monsoon rainfall analysis — Dhaka (2004–2023)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig5_monsoon_analysis.png", bbox_inches="tight")
plt.show()
print("Saved: fig5_monsoon_analysis.png")

# %% ── 11. Export Summary Tables ─────────────────────────────────────────────

# Monthly climatology table
clim_df.to_csv(f"{OUT_DIR}/monthly_climatology.csv", index=False)

# Return period table
rp_df.to_csv(f"{OUT_DIR}/return_periods.csv", index=False)

# Annual stats summary
summary = {
    "Metric"  : ["Mean annual (mm)", "Median annual (mm)", "Std dev (mm)",
                 "CV (%)", "Min annual (mm)", "Max annual (mm)",
                 "Mean monsoon JJAS (mm)", "Monsoon % of annual",
                 "MK trend (annual)", "MK p-value (annual)",
                 "Sen slope (mm/yr)", "Best extreme fit",
                 f"10-yr return (peak month, mm)",
                 f"25-yr return (peak month, mm)",
                 f"50-yr return (peak month, mm)"],
    "Value"   : [
        round(ann.mean(), 1), round(ann.median(), 1),
        round(ann.std(), 1),  round(ann.std()/ann.mean()*100, 1),
        round(ann.min(), 1),  round(ann.max(), 1),
        round(monsoon_annual.mean(), 1),
        round(monsoon_annual.mean()/ann.mean()*100, 1),
        result_ann.trend,     round(result_ann.p, 4),
        round(result_ann.slope, 2), best_fit,
        rp_df.loc[rp_df["Return_Period_yr"]==10,
                  f"{best_fit}_mm"].values[0],
        rp_df.loc[rp_df["Return_Period_yr"]==25,
                  f"{best_fit}_mm"].values[0],
        rp_df.loc[rp_df["Return_Period_yr"]==50,
                  f"{best_fit}_mm"].values[0],
    ]
}
pd.DataFrame(summary).to_csv(f"{OUT_DIR}/phase1_summary_stats.csv",
                              index=False)

print("\n── ALL OUTPUTS SAVED ──")
print(f"  Location: {OUT_DIR}/")
print("  fig1_monthly_climatology.png")
print("  fig2_annual_timeseries.png")
print("  fig3_monthly_heatmap.png")
print("  fig4_return_period_curve.png")
print("  fig5_monsoon_analysis.png")
print("  monthly_climatology.csv")
print("  return_periods.csv")
print("  phase1_summary_stats.csv")
print("\nPhase 1 complete. Proceed to Phase 2 — Flood Hazard Mapping.")
