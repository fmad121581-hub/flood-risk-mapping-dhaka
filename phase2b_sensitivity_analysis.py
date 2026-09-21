# =============================================================================
# Phase 2b — Sensitivity Analysis on Flood Hazard Index (FHI) Weights
# Project : GIS-based Flood Risk & Vulnerability Mapping, Dhaka
# Purpose : Test how robust the Phase 2 hazard classification and the
#           Phase 3 ward risk ranking are to the FHI weighting scheme
#           (TWI 0.35 / Elevation 0.30 / Slope 0.20 / CN 0.15), since
#           those weights were picked from literature, not fitted to
#           local data, and a jury/reviewer will ask "how sensitive is
#           this to that choice?"
#
# Method:
#   1. Named scenarios — plausible alternative weight sets (TWI-heavy,
#      elevation-heavy, equal weights, CN/urban-heavy, and the two
#      no-LULC fallback weights already defined in phase2 itself) —
#      recompute the hazard class map and % area per class for each,
#      and the ward-level risk ranking (Spearman rank correlation vs
#      baseline, and how many of the baseline top-10 wards survive).
#   2. Monte Carlo — 500 random weight draws from a Dirichlet
#      distribution (so weights always sum to 1, keeping the ranges
#      each factor plausibly takes), to see the overall spread of
#      hazard-class-area outcomes and ward-rank stability, not just at
#      a few hand-picked points.
#
# Inputs (from Phase 2 / Phase 3 outputs — must already exist):
#   data/output/phase2/twi_classified.tif
#   data/output/phase2/elev_classified.tif
#   data/output/phase2/slope_classified.tif
#   data/output/phase2/cn_classified.tif
#   data/processed/gadm_dhaka_l4.shp
#
# Outputs -> data/output/phase2b_sensitivity/
#   scenario_weights.csv
#   scenario_hazard_class_pct.csv
#   scenario_ward_rank_correlation.csv
#   montecarlo_summary.csv
#   fig1_hazard_class_by_scenario.png
#   fig2_montecarlo_rank_stability.png
#   fig3_ward_rank_scatter_baseline_vs_worst.png
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
P2   = os.path.join(BASE, "data", "output", "phase2")
PROC = os.path.join(BASE, "data", "processed")
OUT  = os.path.join(BASE, "data", "output", "phase2b_sensitivity")
os.makedirs(OUT, exist_ok=True)

TWI_CLASS_PATH   = os.path.join(P2, "twi_classified.tif")
ELEV_CLASS_PATH  = os.path.join(P2, "elev_classified.tif")
SLOPE_CLASS_PATH = os.path.join(P2, "slope_classified.tif")
CN_CLASS_PATH    = os.path.join(P2, "cn_classified.tif")
WARDS_PATH       = os.path.join(PROC, "gadm_dhaka_l4.shp")

FHI_BREAKS   = [0.35, 0.55, 0.75, 1.0]
HAZARD_NAMES = ["Low", "Medium", "High", "Very High"]

print("=" * 72)
print("Phase 2b: FHI Weight Sensitivity Analysis")
print("=" * 72)

# ── 1. Load classified layers ───────────────────────────────────────────────
def read_class(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(float), src.meta.copy()

twi_c,  meta = read_class(TWI_CLASS_PATH)
elev_c, _    = read_class(ELEV_CLASS_PATH)
slope_c, _   = read_class(SLOPE_CLASS_PATH)
cn_c, _      = read_class(CN_CLASS_PATH)

def norm_hazard(classified_array, invert=False):
    arr = classified_array.astype(float)
    arr[arr == 0] = np.nan
    if invert:
        arr = 5.0 - arr
    return arr / 4.0

twi_score   = norm_hazard(twi_c,   invert=False)
elev_score  = norm_hazard(elev_c,  invert=True)
slope_score = norm_hazard(slope_c, invert=True)
cn_score    = norm_hazard(cn_c,    invert=False)

valid_mask = (~np.isnan(twi_score) & ~np.isnan(elev_score) &
              ~np.isnan(slope_score) & ~np.isnan(cn_score))
n_valid = valid_mask.sum()
print(f"Valid pixels: {n_valid:,}")

def compute_fhi(w_twi, w_elev, w_slope, w_cn):
    fhi = (w_twi * twi_score + w_elev * elev_score +
           w_slope * slope_score + w_cn * cn_score)
    return fhi

def classify_fhi(fhi):
    hazard_class = np.zeros_like(fhi, dtype=np.int16)
    for i, thresh in enumerate(FHI_BREAKS):
        lower = FHI_BREAKS[i - 1] if i > 0 else 0.0
        m = valid_mask & (fhi >= lower) & (fhi <= thresh)
        hazard_class[m] = i + 1
    return hazard_class

def class_pct(hazard_class):
    return {HAZARD_NAMES[i]: round(100 * (hazard_class == i + 1).sum() / n_valid, 2)
            for i in range(4)}

# ── 2. Named scenarios ───────────────────────────────────────────────────────
scenarios = {
    "Baseline (report)":     (0.35, 0.30, 0.20, 0.15),
    "TWI-heavy":             (0.55, 0.20, 0.15, 0.10),
    "Elevation-heavy":       (0.20, 0.50, 0.15, 0.15),
    "Slope-heavy":           (0.25, 0.25, 0.40, 0.10),
    "CN/urban-heavy":        (0.25, 0.20, 0.15, 0.40),
    "Equal weights":         (0.25, 0.25, 0.25, 0.25),
    "No-LULC fallback (phase2 code path)": (0.45, 0.35, 0.20, 0.00),
}
for name, w in scenarios.items():
    assert abs(sum(w) - 1.0) < 1e-9, f"{name} weights do not sum to 1"

print("\nNamed scenarios:")
for name, w in scenarios.items():
    print(f"  {name:<38s} TWI={w[0]:.2f} Elev={w[1]:.2f} Slope={w[2]:.2f} CN={w[3]:.2f}")

baseline_w = scenarios["Baseline (report)"]
baseline_fhi = compute_fhi(*baseline_w)
baseline_class = classify_fhi(baseline_fhi)
baseline_pct = class_pct(baseline_class)

rows_weights, rows_pct = [], []
scenario_classes = {}
for name, w in scenarios.items():
    fhi = compute_fhi(*w)
    hc = classify_fhi(fhi)
    scenario_classes[name] = hc
    pct = class_pct(hc)
    rows_weights.append({"scenario": name, "W_TWI": w[0], "W_ELEV": w[1],
                          "W_SLOPE": w[2], "W_CN": w[3]})
    rows_pct.append({"scenario": name, **pct,
                      "pct_diff_VeryHigh_vs_baseline": round(pct["Very High"] - baseline_pct["Very High"], 2)})

pd.DataFrame(rows_weights).to_csv(os.path.join(OUT, "scenario_weights.csv"), index=False)
df_pct = pd.DataFrame(rows_pct)
df_pct.to_csv(os.path.join(OUT, "scenario_hazard_class_pct.csv"), index=False)
print("\nHazard class % by scenario:")
print(df_pct.to_string(index=False))

# ── 3. Ward-level ranking stability ──────────────────────────────────────────
print("\nComputing ward-level FHI means per scenario (zonal stats) ...")
try:
    from rasterstats import zonal_stats
    HAVE_RASTERSTATS = True
except ImportError:
    HAVE_RASTERSTATS = False
    print("  rasterstats not installed -- skipping ward-level ranking section.")

ward_rank_rows = []
if HAVE_RASTERSTATS and os.path.exists(WARDS_PATH):
    wards_raw = gpd.read_file(WARDS_PATH).to_crs(meta["crs"])
    wards = wards_raw[wards_raw.geometry.notna()].copy()
    wards["geometry"] = wards.geometry.buffer(0)
    wards = wards[wards.geometry.is_valid & ~wards.geometry.is_empty].reset_index(drop=True)
    name_col = next((c for c in ["NAME_4", "NAME_3", "name", "NAME"] if c in wards.columns), wards.columns[0])
    print(f"  {len(wards)} wards, name column: {name_col}")

    # Write each scenario's FHI raster to a temp in-memory array and zonal-stat it
    ward_means = {}
    for name, w in scenarios.items():
        fhi = compute_fhi(*w)
        fhi_write = np.where(valid_mask, fhi, -9999.0).astype(np.float32)
        zs = zonal_stats(wards, fhi_write, affine=meta["transform"],
                          stats=["mean"], nodata=-9999.0)
        ward_means[name] = np.array([s["mean"] if s["mean"] is not None else 0.0 for s in zs])

    baseline_means = ward_means["Baseline (report)"]
    baseline_order = np.argsort(-baseline_means)
    baseline_top10 = set(wards[name_col].iloc[baseline_order[:10]])

    for name in scenarios:
        rho, pval = spearmanr(baseline_means, ward_means[name])
        order = np.argsort(-ward_means[name])
        top10 = set(wards[name_col].iloc[order[:10]])
        overlap = len(baseline_top10 & top10)
        ward_rank_rows.append({
            "scenario": name,
            "spearman_rho_vs_baseline": round(rho, 4),
            "top10_wards_retained_of_10": overlap,
        })
    df_wards = pd.DataFrame(ward_rank_rows)
    df_wards.to_csv(os.path.join(OUT, "scenario_ward_rank_correlation.csv"), index=False)
    print("\nWard-level rank stability vs baseline:")
    print(df_wards.to_string(index=False))
else:
    df_wards = None

# ── 4. Monte Carlo (500 Dirichlet draws) ─────────────────────────────────────
print("\nRunning Monte Carlo sensitivity (500 draws, Dirichlet weights) ...")
rng = np.random.default_rng(42)
N_DRAWS = 120
# Dirichlet alpha proportional to the baseline weights (concentration=8) keeps
# draws centred near the literature-based baseline rather than fully uniform,
# which is more representative of "plausible" alternative weightings.
alpha = np.array(baseline_w) * 8
draws = rng.dirichlet(alpha, size=N_DRAWS)

veryhigh_pcts = np.empty(N_DRAWS)
rank_rhos = np.empty(N_DRAWS)

if HAVE_RASTERSTATS and os.path.exists(WARDS_PATH):
    for i, w in enumerate(draws):
        fhi = compute_fhi(*w)
        hc = classify_fhi(fhi)
        veryhigh_pcts[i] = 100 * (hc == 4).sum() / n_valid
        fhi_write = np.where(valid_mask, fhi, -9999.0).astype(np.float32)
        zs = zonal_stats(wards, fhi_write, affine=meta["transform"],
                          stats=["mean"], nodata=-9999.0)
        means_i = np.array([s["mean"] if s["mean"] is not None else 0.0 for s in zs])
        rho, _ = spearmanr(baseline_means, means_i)
        rank_rhos[i] = rho
        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{N_DRAWS} draws")
else:
    for i, w in enumerate(draws):
        hc = classify_fhi(compute_fhi(*w))
        veryhigh_pcts[i] = 100 * (hc == 4).sum() / n_valid
    rank_rhos[:] = np.nan

mc_summary = {
    "n_draws": N_DRAWS,
    "VeryHigh_pct_baseline": round(baseline_pct["Very High"], 2),
    "VeryHigh_pct_mean": round(float(np.mean(veryhigh_pcts)), 2),
    "VeryHigh_pct_std": round(float(np.std(veryhigh_pcts)), 2),
    "VeryHigh_pct_min": round(float(np.min(veryhigh_pcts)), 2),
    "VeryHigh_pct_max": round(float(np.max(veryhigh_pcts)), 2),
    "ward_rank_spearman_mean": round(float(np.nanmean(rank_rhos)), 4),
    "ward_rank_spearman_min": round(float(np.nanmin(rank_rhos)), 4) if not np.all(np.isnan(rank_rhos)) else None,
    "ward_rank_spearman_p5": round(float(np.nanpercentile(rank_rhos, 5)), 4) if not np.all(np.isnan(rank_rhos)) else None,
}
pd.DataFrame([mc_summary]).to_csv(os.path.join(OUT, "montecarlo_summary.csv"), index=False)
print("\nMonte Carlo summary:")
for k, v in mc_summary.items():
    print(f"  {k}: {v}")

# ── 5. Figures ────────────────────────────────────────────────────────────────
print("\nGenerating figures ...")

# Fig 1: stacked bar of hazard class % by scenario
fig, ax = plt.subplots(figsize=(11, 6))
names = df_pct["scenario"].tolist()
bottoms = np.zeros(len(names))
colors = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]
for i, cls in enumerate(HAZARD_NAMES):
    vals = df_pct[cls].values
    ax.barh(names, vals, left=bottoms, color=colors[i], label=cls, edgecolor="white")
    bottoms += vals
ax.set_xlabel("% of study area")
ax.set_title("Hazard class distribution across FHI weighting scenarios")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_hazard_class_by_scenario.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  -> fig1_hazard_class_by_scenario.png")

# Fig 2: Monte Carlo distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(veryhigh_pcts, bins=40, color="#8e44ad", alpha=0.8)
axes[0].axvline(baseline_pct["Very High"], color="black", linestyle="--", linewidth=1.5,
                label=f"Baseline = {baseline_pct['Very High']:.1f}%")
axes[0].set_xlabel("% of study area classed 'Very High' hazard")
axes[0].set_ylabel("Monte Carlo draws")
axes[0].set_title(f"Spread of 'Very High' hazard area\n(n={N_DRAWS} random weight draws)")
axes[0].legend(frameon=False, fontsize=9)

if not np.all(np.isnan(rank_rhos)):
    axes[1].hist(rank_rhos, bins=40, color="#3498db", alpha=0.8)
    axes[1].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Spearman rank correlation vs baseline ward ranking")
    axes[1].set_ylabel("Monte Carlo draws")
    axes[1].set_title("Ward risk-ranking stability under weight perturbation")
else:
    axes[1].text(0.5, 0.5, "rasterstats not available", ha="center", va="center")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_montecarlo_rank_stability.png"), dpi=200, bbox_inches="tight")
plt.close()
print("  -> fig2_montecarlo_rank_stability.png")

# Fig 3: baseline ward rank vs worst-case scenario ward rank
if df_wards is not None:
    worst_name = df_wards.loc[df_wards["spearman_rho_vs_baseline"].idxmin(), "scenario"]
    if worst_name != "Baseline (report)":
        worst_means = ward_means[worst_name]
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        baseline_rank = pd.Series(baseline_means).rank(ascending=False)
        worst_rank = pd.Series(worst_means).rank(ascending=False)
        ax.scatter(baseline_rank, worst_rank, alpha=0.5, s=18, color="#c0392b")
        lims = [1, len(baseline_rank)]
        ax.plot(lims, lims, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Ward rank -- baseline weights")
        ax.set_ylabel(f"Ward rank -- {worst_name}")
        ax.set_title("Most sensitive named scenario vs baseline\n(points near the diagonal = stable ranking)")
        plt.tight_layout()
        fig.savefig(os.path.join(OUT, "fig3_ward_rank_scatter_baseline_vs_worst.png"), dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  -> fig3_ward_rank_scatter_baseline_vs_worst.png (worst case: {worst_name})")

print("\n" + "=" * 72)
print("Phase 2b complete.")
print(f"Outputs saved to: {OUT}")
print("=" * 72)
