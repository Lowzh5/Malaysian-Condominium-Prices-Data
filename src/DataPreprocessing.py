import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, "data", "raw", "houses.csv")
df = pd.read_csv(csv_path)

DOWNLOADS_DIR = os.path.join(os.path.expanduser("~"), "Downloads")

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

plt.figure(figsize=(11, 6))
sns.heatmap(df.isna(), cbar=False, cmap="viridis")
plt.title("Figure(1): Missing Value Map After Standardisation")
plt.tight_layout()
plt.savefig(os.path.join(DOWNLOADS_DIR, "fig01_missing_map.png"), dpi=150)
plt.close()

df.to_csv(os.path.join(DOWNLOADS_DIR, "houses_step31.csv"), index=False)
print(f"\nSaved to Downloads: {df.shape}")

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
core_cols = ['price', 'Bedroom', 'Bathroom', 'Property Size', 'Address']
dup_adlist_mask = df['Ad List'].duplicated(keep=False)
dup_groups = df.loc[dup_adlist_mask, 'Ad List'].unique()

conflicts = []
for gid in dup_groups:
    sub = df[df['Ad List'] == gid][core_cols].astype(str)
    if sub.nunique().gt(1).any():
        conflicts.append(gid)

print(f"\nAd List duplicate groups found:    {len(dup_groups)}")
print(f"Groups with conflicting core data: {len(conflicts)}")

df = df.drop_duplicates(subset='Ad List', keep='first')
shape_after_adlist = df.shape[0]

print(f"Rows after Ad List dedup:          {shape_after_adlist}")

print("\n--- Before / After summary table ---")
summary = pd.DataFrame({
    "Stage": ["After 3.1 (standardisation)", "After exact duplicate removal", "After Ad List duplicate removal"],
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

df.to_csv(os.path.join(DOWNLOADS_DIR, "houses_step32.csv"), index=False)
print(f"\nSaved to Downloads: {df.shape}")