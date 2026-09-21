# =============================================================================
# Phase 3b — External Validation of Ward Flood Risk Ranking
# Project : GIS-based Flood Risk & Vulnerability Mapping, Dhaka
# Purpose : Check the Phase 3 ward risk ranking (ward_risk_ranked.csv) against
#           an independent, peer-reviewed source of Dhaka water-logging
#           vulnerability, since the FHI weights and vulnerability formula
#           were both picked from literature, not fitted or ground-truthed
#           against any observed flood data. This script is that ground
#           truth check.
#
# Reference source:
#   Alam, R., Quayyum, Z., Moulds, S., Radia, M.A., Sara, H.H., Hasan, M.T.,
#   & Butler, A. (2023). "Dhaka city water logging hazards: area
#   identification and vulnerability assessment through GIS-remote sensing
#   techniques." Environmental Monitoring and Assessment, 195(5), 543.
#   https://doi.org/10.1007/s10661-023-11106-y
#
#   That paper independently classifies DNCC/DSCC wards and peripheral
#   unions by water-logging vulnerability using its own GIS/remote-sensing
#   method (not this project's data or code). It reports:
#     - Wards 2, 5, 7, 14: "serious water logging problem"
#     - Wards 8, 9, 33: low vulnerability
#     - Peripheral unions -- Uttar Khan, Dakshin Khan, Bhatara, Dakshingaon,
#       Manda, Matuail, Saralia, Shyampur, Sultanganj: low/very low
#       vulnerability
#     - Dumni, Beraid, Satarkul, Nasirabad, Demra: "very low vulnerability"
#
# Method:
#   For each named ward/union in that paper, find its percentile rank in
#   this project's ward_risk_ranked.csv (rank 1 = highest modelled risk).
#   A well-behaved model should rank the paper's "high vulnerability" wards
#   near the top and its "low vulnerability" wards near the bottom.
#
# Outputs -> data/output/phase3b_validation/
#   validation_matches.csv        per-ward comparison table
#   validation_summary.csv        group-level agreement statistics
#   fig1_validation_percentiles.png
# =============================================================================

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANKED_CSV = os.path.join(BASE, "data", "output", "phase3", "ward_risk_ranked.csv")
OUT = os.path.join(BASE, "data", "output", "phase3b_validation")
os.makedirs(OUT, exist_ok=True)

print("=" * 72)
print("Phase 3b: External Validation vs Alam et al. (2023)")
print("=" * 72)

df = pd.read_csv(RANKED_CSV)
name_col = "NAME_4"
df = df.sort_values("risk_mean", ascending=False).reset_index(drop=True)
df["risk_rank"] = np.arange(1, len(df) + 1)
df["risk_percentile"] = 100 * (1 - (df["risk_rank"] - 1) / (len(df) - 1))  # 100 = highest risk

# ── Reference classification from Alam et al. (2023) ────────────────────────
# "Ward No-XX" matches the DNCC/DSCC numbered wards in that paper directly.
# Peripheral unions are matched by name (case-insensitive, ignoring "(part)"
# / "(Part)" suffixes that GADM splits some unions into multiple polygons).
REFERENCE = {
    "High vulnerability (paper)": {
        "type": "ward_number", "items": [2, 5, 7, 14],
    },
    "Low vulnerability (paper)": {
        "type": "ward_number", "items": [8, 9, 33],
    },
    "Very low vulnerability (paper, peripheral unions)": {
        "type": "union_name",
        "items": ["Dumni", "Beraid", "Satarkul", "Nasirabad", "Demra"],
    },
    "Low vulnerability (paper, peripheral unions)": {
        "type": "union_name",
        "items": ["Uttar Khan", "Dakshinkhan", "Dakshin Khan", "Bhatara",
                  "Dakshingaon", "Manda", "Matuail", "Saralia", "Shyampur",
                  "Sultanganj"],
    },
}

def clean_name(s):
    s = re.sub(r"\(.*?\)", "", str(s))          # drop "(Part)" etc.
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

df["_clean_name"] = df[name_col].apply(clean_name)

def match_ward_number(n):
    pat = re.compile(rf"^ward no-0*{n}$")
    return df[df["_clean_name"].str.match(pat)]

def match_union_name(name):
    target = clean_name(name)
    return df[df["_clean_name"] == target]

rows = []
for group, spec in REFERENCE.items():
    for item in spec["items"]:
        if spec["type"] == "ward_number":
            matches = match_ward_number(item)
            label = f"Ward No-{item:02d}"
        else:
            matches = match_union_name(item)
            label = item
        if matches.empty:
            rows.append({"group": group, "item": label, "matched_ward": None,
                         "risk_percentile": None, "risk_rank": None, "note": "not found in GADM ward list"})
        else:
            for _, r in matches.iterrows():
                rows.append({"group": group, "item": label,
                             "matched_ward": r[name_col],
                             "risk_percentile": round(r["risk_percentile"], 1),
                             "risk_rank": int(r["risk_rank"]),
                             "note": ""})

df_val = pd.DataFrame(rows)
df_val.to_csv(os.path.join(OUT, "validation_matches.csv"), index=False)
print("\nPer-ward validation matches:")
print(df_val.to_string(index=False))

n_total = len(df)
print(f"\n(risk_percentile: 100 = highest-modelled-risk ward of {n_total}, 0 = lowest)")

# ── Group-level summary ──────────────────────────────────────────────────────
summary_rows = []
for group in REFERENCE:
    sub = df_val[(df_val["group"] == group) & df_val["risk_percentile"].notna()]
    if len(sub) == 0:
        continue
    summary_rows.append({
        "group": group,
        "n_matched": len(sub),
        "mean_risk_percentile": round(sub["risk_percentile"].mean(), 1),
        "median_risk_percentile": round(sub["risk_percentile"].median(), 1),
    })
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(OUT, "validation_summary.csv"), index=False)
print("\nGroup-level agreement (mean risk percentile; expect High group near 100, Low/Very-low groups near 0):")
print(df_summary.to_string(index=False))

# ── Figure: percentile distribution by reference group ──────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
groups_plot = [g for g in REFERENCE if not df_val[(df_val["group"] == g) & df_val["risk_percentile"].notna()].empty]
data_plot = [df_val[(df_val["group"] == g) & df_val["risk_percentile"].notna()]["risk_percentile"].values
             for g in groups_plot]
bp = ax.boxplot(data_plot, vert=False, patch_artist=True, widths=0.55)
colors = ["#c0392b", "#27ae60", "#2ecc71", "#82e0aa"]
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.75)
for i, vals in enumerate(data_plot):
    ax.scatter(vals, np.full(len(vals), i + 1), color="black", s=18, zorder=5, alpha=0.7)
ax.set_yticks(range(1, len(groups_plot) + 1))
ax.set_yticklabels([g.replace(" (paper", "\n(paper") for g in groups_plot], fontsize=9)
ax.set_xlabel("This model's risk percentile (100 = highest modelled risk)")
ax.set_title("Model risk ranking vs Alam et al. (2023) field-independent\nwater-logging vulnerability classification")
ax.axvline(50, color="gray", linestyle="--", linewidth=1)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_validation_percentiles.png"), dpi=200, bbox_inches="tight")
plt.close()
print("\n  -> fig1_validation_percentiles.png")

print("\n" + "=" * 72)
print("Phase 3b complete.")
print(f"Outputs saved to: {OUT}")
print("=" * 72)
