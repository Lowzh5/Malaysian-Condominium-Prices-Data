import numpy as np
import pandas as pd
import os
import joblib 

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

pd.set_option('display.max_columns', None)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, "data", "raw", "houses.csv")
df = pd.read_csv(csv_path)

DOWNLOADS_DIR = os.path.join(os.path.expanduser("~"), "Downloads")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ============================================================
# Step 0 — Review (observe only, no changes made)
# ============================================================
print("="*60)
print("STEP 0: DATASET REVIEW")
print("="*60)
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
print(df.info())
print()
print(df.head())
print()
print("Unique value counts per column:")
print(df.nunique())
print()
print(f"Exact duplicate rows: {df.duplicated().sum()}")

# ============================================================
# 3.1 Missing Value Standardisation
# ============================================================
print("\n" + "="*60)
print("STEP 3.1: MISSING VALUE STANDARDISATION")
print("="*60)

CANDIDATE_SENTINELS = ['-', '', 'N/A', 'n/a', 'NA', 'None', 'none', 'NULL', 'null',
                        'nan', 'NAN', '--', 'NIL', '?', 'unknown', 'Unknown', ' ']

obj_cols = df.select_dtypes(include=['object', 'string']).columns
df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())

found_sentinels = set()
for col in obj_cols:
    vals = df[col].dropna().astype(str)
    hits = vals[vals.isin(CANDIDATE_SENTINELS)]
    if len(hits) > 0:
        found_sentinels.update(hits.unique())

print(f"Candidate sentinels checked:      {CANDIDATE_SENTINELS}")
print(f"Sentinels actually found in data: {found_sentinels}")

before_na = df.isna().sum().sum()
df = df.replace(list(found_sentinels), np.nan)
after_na = df.isna().sum().sum()

print(f"\nNaN count before standardisation: {before_na:,}")
print(f"NaN count after standardisation:  {after_na:,}")
print(f"Newly identified missing values:  {after_na - before_na:,}")

print("\n--- Missing value summary by column ---")
miss_summary = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_pct": (df.isna().sum() / len(df) * 100).round(1)
}).sort_values("missing_count", ascending=False)
print(miss_summary[miss_summary["missing_count"] > 0])

print("\n--- Missing value count per row (distribution) ---")
missing_per_row = df.isna().sum(axis=1)
print(missing_per_row.describe())

fig, ax = plt.subplots(figsize=(11, 6))
sns.heatmap(df.isna(), cbar=False, cmap="viridis", ax=ax)
ax.set_title("Missing Value Map After Standardisation")
ax.set_xlabel("Column")
ax.set_ylabel("Row index")
legend_handles = [
    Patch(facecolor=plt.get_cmap("viridis")(0.0), label="Non-null (value present)"),
    Patch(facecolor=plt.get_cmap("viridis")(1.0), label="Null (missing)")
]
ax.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0, -0.45), frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(DOWNLOADS_DIR, "fig01_missing_map.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 3.2 Duplicate Removal
# ============================================================
print("\n" + "="*60)
print("STEP 3.2: DUPLICATE REMOVAL")
print("="*60)

shape_before = df.shape[0]

# Stage 1 — exact duplicates
exact_dup_count = df.duplicated().sum()
df = df.drop_duplicates()
shape_after_exact = df.shape[0]

print(f"Rows before 3.2:                 {shape_before}")
print(f"Exact duplicate rows removed:    {exact_dup_count}")
print(f"Rows after exact dedup:          {shape_after_exact}")

# Stage 2 — duplicated Ad List
# Ad List is the unique listing identifier, so a repeated value means the same
# advertisement was captured more than once. All columns (not just core fields)
# are compared to detect any genuine conflict between the records.
dup_adlist_mask = df['Ad List'].duplicated(keep=False)
dup_groups = df.loc[dup_adlist_mask, 'Ad List'].unique()

conflict_detail = []
for gid in dup_groups:
    sub = df[df['Ad List'] == gid]
    conflicting_cols = []
    for c in df.columns:
        vals = sub[c].dropna().astype(str).unique()
        if len(vals) > 1:
            conflicting_cols.append((c, list(vals)))
    if conflicting_cols:
        conflict_detail.append((gid, conflicting_cols))

print(f"\nAd List duplicate groups found:      {len(dup_groups)}")
print(f"Groups with conflict in any column:  {len(conflict_detail)}")

if conflict_detail:
    print("\nConflicting records requiring review:")
    for gid, cols in conflict_detail:
        print(f"  Ad List {gid}:")
        for c, vals in cols:
            print(f"     {c}: {vals}")
    print("\nNote: the dataset contains no timestamp column, so record recency")
    print("cannot be verified. Resolution rule applied: retain the record with")
    print("more non-null fields, as the more complete source.")

# Quantify what a merge would recover before applying it, so the choice of
# strategy is evidence-based rather than assumed.
merge_gain = 0
for gid in dup_groups:
    sub = df[df['Ad List'] == gid]
    best_single = sub.notna().sum(axis=1).max()
    merged_nonnull = sub.notna().any(axis=0).sum()
    merge_gain += (merged_nonnull - best_single)

print(f"\nAdditional non-null cells recoverable by merging: {merge_gain}")

# Merge rather than blindly keeping the first record: groupby().first() takes
# the first non-null value per column, so complementary fields present only in
# a later record are preserved instead of discarded.
df = df.sort_values('Ad List', kind='stable')
df = df.groupby('Ad List', as_index=False, sort=False).first()
shape_after_adlist = df.shape[0]

print(f"Rows after Ad List merge:            {shape_after_adlist}")

print("\n--- Before / After summary table ---")
summary = pd.DataFrame({
    "Stage": ["After 3.1 (standardisation)", "After exact duplicate removal", "After Ad List merge"],
    "Rows": [shape_before, shape_after_exact, shape_after_adlist]
})
print(summary.to_string(index=False))

print(f"\nFinal duplicate check: {df.duplicated().sum()} exact dups, "
      f"{df['Ad List'].duplicated().sum()} Ad List dups remaining")

# Stage 3 — supplementary check: near-duplicate re-listings
# (duplicated Ad List already handled above; this checks records with
#  different Ad List values but otherwise identical content, which likely
#  represent re-listings of the same property over time rather than a
#  data collection error, hence retained after review)
print("\n--- Supplementary check: re-listings ignoring Ad List / description ---")
ignore_cols = ['Ad List', 'description']
check_cols = [c for c in df.columns if c not in ignore_cols]
near_dup_count = df.duplicated(subset=check_cols).sum()
print(f"Rows identical except Ad List/description: {near_dup_count}")
print("Interpretation: different Ad List values confirm these are independent")
print("listing events, not duplicate scrapes of the same ad. Likely represents")
print("re-listing of the same property over time. Retained (not removed) as")
print("there is no evidence this is a data collection error.")

df.to_csv(os.path.join(PROCESSED_DIR, "houses_step32.csv"), index=False)
joblib.dump(df, os.path.join(PROCESSED_DIR, "houses_step32.pkl"))
print(f"\nSaved to {PROCESSED_DIR}: {df.shape}")