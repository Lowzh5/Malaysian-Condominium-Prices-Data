import numpy as np
import pandas as pd
import os
import re
import joblib

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import chi2

pd.set_option('display.max_columns', None)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, "data", "raw", "houses.csv")
df = pd.read_csv(csv_path)
RAW_COLUMNS = list(df.columns)  # snapshot for 3.12's summary - kept vs dropped vs engineered

DOWNLOADS_DIR = os.path.join(os.path.expanduser("~"), "Downloads")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
EXTRA_DIR = os.path.join(BASE_DIR, "data", "extra")
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(EXTRA_DIR, exist_ok=True)

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

missing_cols = miss_summary[miss_summary["missing_count"] > 0].sort_values("missing_count")
fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(missing_cols))))
bars = ax.barh(missing_cols.index, missing_cols["missing_count"], color="#4C72B0")
ax.set_title("Missing Value Count per Column After Standardisation")
ax.set_xlabel("Missing count")
ax.bar_label(bars, labels=[f"{c} ({p}%)" for c, p in zip(missing_cols["missing_count"], missing_cols["missing_pct"])],
             padding=3, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(DOWNLOADS_DIR, "fig01_missing_bar.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 3.2.1 Duplicate Removal
# ============================================================
print("\n" + "="*60)
print("STEP 3.2.1: DUPLICATE HANDLING")
print("="*60)

shape_before = df.shape[0]

# Stage 1 — exact duplicates
exact_dup_count = df.duplicated().sum()
df = df.drop_duplicates()
shape_after_exact = df.shape[0]

# Stage 2 — duplicated Ad List: a repeated ID means the same ad was captured twice.
dup_adlist_mask = df['Ad List'].duplicated(keep=False)
dup_groups = df.loc[dup_adlist_mask, 'Ad List'].unique()
rows_in_dup_groups = dup_adlist_mask.sum()

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

if conflict_detail:
    print("\nConflicting records requiring review:")
    for gid, cols in conflict_detail:
        print(f"  Ad List {gid}:")
        for c, vals in cols:
            print(f"     {c}: {vals}")
    print("\nNote: the dataset contains no timestamp column, so record recency")
    print("cannot be verified. Resolution rule applied: retain the record with")
    print("more non-null fields, as the more complete source.")

# Snapshot pre-merge rows for the before/after comparison below.
before_merge_snapshot = df.loc[dup_adlist_mask].copy()
before_merge_snapshot.insert(0, 'stage', 'before')

# Keep the most complete record per Ad List group instead of an arbitrary first row.
df['_completeness'] = df.notna().sum(axis=1)
df = df.sort_values(['Ad List', '_completeness'], ascending=[True, False], kind='stable')
df = df.groupby('Ad List', as_index=False, sort=False).first() 
df = df.drop(columns='_completeness')
shape_after_adlist = df.shape[0]
rows_removed_by_merge = rows_in_dup_groups - len(dup_groups)

print(f"\nOriginal number of rows:                          {shape_before}")
print(f"Exact duplicate rows removed:                      {exact_dup_count}")
print(f"Rows after exact duplicate removal:                {shape_after_exact}")
print(f"\nRemaining non-identical duplicated Ad List groups: {len(dup_groups)}")
print(f"Rows involved in these groups:                     {rows_in_dup_groups}")
print(f"Groups with conflicting non-missing values:        {len(conflict_detail)}")
print(f"Rows removed after merging duplicate listings:  {rows_removed_by_merge}")
print(f"\nFinal number of rows:                              {shape_after_adlist}")
print(f"Remaining exact duplicate rows:                     {df.duplicated().sum()}")
print(f"Remaining duplicated Ad List groups:                {df['Ad List'].duplicated().sum()}")

# Before/after evidence for the report.
after_merge_snapshot = df[df['Ad List'].isin(dup_groups)].copy()
after_merge_snapshot.insert(0, 'stage', 'after_merge')
comparison_df = pd.concat([before_merge_snapshot, after_merge_snapshot], ignore_index=True)
comparison_df = comparison_df.sort_values(['Ad List', 'stage'])
comparison_df.to_csv(os.path.join(PROCESSED_DIR, "adlist_merge_before_after.csv"), index=False)
print(f"\nBefore/after merge comparison saved to {PROCESSED_DIR}\\adlist_merge_before_after.csv")

# Stage 3 — near-duplicate re-listings: same content, different Ad List
# (likely re-listed over time, not a collection error).
print("\n--- Supplementary check: re-listings ignoring Ad List / description ---")
ignore_cols = ['Ad List', 'description']
check_cols = [c for c in df.columns if c not in ignore_cols]
near_dup_count = df.duplicated(subset=check_cols).sum()
print(f"Rows identical except Ad List/description: {near_dup_count}")
print("Interpretation: different Ad List values confirm these are independent")
print("listing events, not duplicate scrapes of the same ad. Likely represents")
print("re-listing of the same property over time. Retained (not removed) as")
print("there is no evidence this is a data collection error.")

# near_dup_count above (25) counts DUPLICATE ROWS under pandas' default
# keep='first' - every row in a group except its first occurrence. Grouping
# the same rows by check_cols instead counts GROUPS, a different unit: 23
# groups covering 48 rows total (48 - 23 = 25, so the two numbers agree,
# they just measure rows-beyond-the-first vs. groups).
near_dup_mask = df.duplicated(subset=check_cols, keep=False)
near_dup_df = df.loc[near_dup_mask].copy()
group_ids = near_dup_df.groupby(check_cols, dropna=False, sort=False).ngroup()
near_dup_df.insert(0, 'group_id', group_ids)

# Verify the premise: within each group, does 'description' actually differ,
# or does it also happen to match (making the pair closer to an exact
# duplicate that only differs by Ad List)?
near_dup_df.insert(1, 'description_identical', near_dup_df.groupby('group_id')['description']
                    .transform(lambda s: s.nunique(dropna=False) <= 1))
near_dup_df = near_dup_df.sort_values(['group_id', 'Ad List'], kind='stable')

print(f"\nNear-duplicate groups: {near_dup_df['group_id'].nunique()} ({len(near_dup_df)} rows involved)")
print(f"Groups where description is ALSO identical (closer to an exact duplicate): "
      f"{near_dup_df.loc[near_dup_df['description_identical'], 'group_id'].nunique()}")

near_dup_df.to_csv(os.path.join(EXTRA_DIR, "near_duplicate.csv"), index=False)
print(f"Saved to {EXTRA_DIR}\\near_duplicate.csv")

# ============================================================
# 3.2.2 Unrelated Row Removal
# ============================================================
print("\n" + "="*60)
print("STEP 3.2.2: UNRELATED ROW REMOVAL")
print("="*60)

"""
Property Type == 'Others' is a catch-all label that doesn't identify a real
condominium/apartment-family property type, unlike the other 7 categories
(Condominium, Apartment, Service Residence, Flat, Studio, Duplex, Townhouse
Condo). Since this project analyses condominium prices specifically, these
rows are out of scope - removed as unrelated records, not a data quality fix.
"""
shape_before_unrelated = df.shape[0]
other_count = (df['Property Type'] == 'Others').sum()

df = df[df['Property Type'] != 'Others']
shape_after_unrelated = df.shape[0]

print(f"Rows with Property Type == 'Others': {other_count}")
print(f"Rows before removal (end of 3.2.1): {shape_before_unrelated}")
print(f"Rows after removing Property Type == 'Others': {shape_after_unrelated}")

df.to_csv(os.path.join(PROCESSED_DIR, "houses_cleaned.csv"), index=False)
joblib.dump(df, os.path.join(PROCESSED_DIR, "houses_cleaned.pkl"))
print(f"\nSaved to {PROCESSED_DIR}: {df.shape}")

# ============================================================
# 3.3 Data Type Conversion
# ============================================================
print("\n" + "="*60)
print("STEP 3.3: DATA TYPE CONVERSION")
print("="*60)

conversion_log = []

def convert_and_report(col_name, converter, rule):
    before_non_null = df[col_name].notna().sum()
    converted = converter(df[col_name])
    after_non_null = converted.notna().sum()
    conversion_log.append({
        "Column": col_name,
        "Rule": rule,
        "Non-null before": before_non_null,
        "Non-null after": after_non_null,
        "Conversion failures": before_non_null - after_non_null,
    })
    return converted

# price: "RM 340 000" -> 340000
df['price'] = convert_and_report(
    'price',
    lambda s: pd.to_numeric(s.str.replace('RM', '', regex=False).str.replace(' ', '', regex=False), errors='coerce'),
    "Remove 'RM' prefix and internal spaces, then convert to numeric"
)

# Extract the number preceding "sq.ft." — stripping non-numeric chars would
# corrupt values like "1000.." from the unit's own periods.
df['Property Size'] = convert_and_report(
    'Property Size',
    lambda s: pd.to_numeric(
        s.str.extract(r'([\d,]+(?:\.\d+)?)\s*sq\.ft\.')[0].str.replace(',', '', regex=False),
        errors='coerce'
    ),
    "Extract numeric value preceding 'sq.ft.', strip thousands separators, convert to numeric"
)

# Already plain numeric strings; only need type coercion.
simple_numeric_cols = ['Bedroom', 'Bathroom', 'Completion Year', '# of Floors', 'Total Units', 'Parking Lot']
for col in simple_numeric_cols:
    df[col] = convert_and_report(
        col,
        lambda s: pd.to_numeric(s, errors='coerce'),
        "Direct conversion to numeric (already plain integer strings)"
    )

conversion_df = pd.DataFrame(conversion_log)
print("\n--- Conversion summary ---")
print(conversion_df.to_string(index=False))

print("\nNote: pd.to_numeric selects the dtype automatically - columns with no")
print("missing values (price, Property Size) become int64 directly, while columns")
print("still carrying NaN (pending Step 3.5 imputation) become float64, since NaN")
print("cannot be stored in int64. No manual dtype casting is applied at this stage.")

print("\n--- Dtypes after 3.3 ---")
print(df[['price', 'Property Size'] + simple_numeric_cols].dtypes)

df.to_csv(os.path.join(PROCESSED_DIR, "houses_cleaned.csv"), index=False)
joblib.dump(df, os.path.join(PROCESSED_DIR, "houses_cleaned.pkl"))
print(f"\nSaved to {PROCESSED_DIR}: {df.shape}")

# ============================================================
# 3.4 Invalid Value Correction
# ============================================================
print("\n" + "="*60)
print("STEP 3.4: INVALID VALUE CORRECTION")
print("="*60)

df['Property Size'] = df['Property Size'].astype(float)  # a correction may add a decimal or NaN

print("\n--- Property Size: bottom 1% before correction ---")
size_p1_threshold = df['Property Size'].quantile(0.01)
print(f"1st percentile threshold: {size_p1_threshold:.1f} sq.ft.")
print(df.loc[df['Property Size'] <= size_p1_threshold, ['Ad List', 'Property Size']].to_string(index=False))

"""
Property Size is cross-checked against the 'sq.ft' figure independently
stated in the description. Ad List 103788197 (1 sq.ft.) and 103423738
(9 sq.ft.) are the two rows the bottom 1% scan above surfaces as
physically impossible - a plain extraction failure, not a real property
size. Targeted directly by Ad List (same reasoning as the Facilities '10'
fix below): the bottom 1% scan already confirms these are the only two
affected rows, so no blanket below-threshold scan is needed.
"""
SIZE_PATTERN = re.compile(
    r'(\d[\d,]*\.?\d*)\s*(?:sq\.?\s*ft\.?|sqft|square\s*feet|sf)(?!\w)',
    re.IGNORECASE
)

def extract_size_from_description(text):
    if pd.isna(text):
        return np.nan
    match = SIZE_PATTERN.search(str(text))
    return float(match.group(1).replace(',', '')) if match else np.nan

PROPERTY_SIZE_FIX_ADLIST = [103788197, 103423738]
size_log = []
for adlist in PROPERTY_SIZE_FIX_ADLIST:
    idx = df.index[df['Ad List'] == adlist][0]
    old_val = df.loc[idx, 'Property Size']
    dval = extract_size_from_description(df.loc[idx, 'description'])
    df.loc[idx, 'Property Size'] = dval
    size_log.append({
        "Ad List": adlist,
        "Original Size": old_val,
        "Description Size": dval,
        "Final Size": round(dval) if pd.notna(dval) else np.nan,
    })

size_log_df = pd.DataFrame(size_log)
print("\n--- Property Size Correction Log (Ad List, original vs. description-confirmed value) ---")
print(size_log_df.to_string(index=False))

# Property Size is integer sq.ft. in this dataset; round the one decimal the
# description correction introduced, then cast back to int64 (still 0-missing).
df['Property Size'] = df['Property Size'].round().astype(int)
print("\n" + "-"*60)

print("\n--- Facilities: remove scraping artefact '10' ---")

"""
Ad List 95706905's Facilities value ends in a stray "10" - not a real
facility name, a scraping artefact. Confirmed unique to this one row (every
row's comma-separated Facilities items were checked for a standalone
digit-only token; no other row has this issue), so the fix targets this Ad
List directly rather than a blanket rule, and only removes the "10" token,
not the row.
"""
FACILITIES_FIX_ADLIST = 95706905
old_facilities = df.loc[df['Ad List'] == FACILITIES_FIX_ADLIST, 'Facilities'].iloc[0]
new_facilities = re.sub(r',\s*10\s*$', '', old_facilities)
df.loc[df['Ad List'] == FACILITIES_FIX_ADLIST, 'Facilities'] = new_facilities
print(f"Ad List {FACILITIES_FIX_ADLIST}")
print(f"  Before: {old_facilities}")
print(f"  After:  {new_facilities}")
print("\n" + "-"*60)

# No independent field verifies floor count, so any generic round-number
# cutoff works as a screen. Malaysia's tallest building (Merdeka 118, 118
# storeys) is no longer used as the benchmark - 78 is used instead as a
# plain threshold not tied to any specific building. The data has a natural
# gap between the highest plausible value (63) and the next value up (135),
# so 78 and 118 flag the exact same rows today; the lower threshold is what
# keeps this screen meaningful if a future data pull adds values in between.
FLOOR_MAX = 78
floor_invalid_mask = df['# of Floors'] > FLOOR_MAX
print(f"\n'# of Floors' > {FLOOR_MAX} (unlikely): {floor_invalid_mask.sum()}")
print(df.loc[floor_invalid_mask, ['Ad List', '# of Floors']].to_string(index=False))
df.loc[floor_invalid_mask, '# of Floors'] = np.nan

print(f"\nProperty Size range after 3.4: {df['Property Size'].min()} - {df['Property Size'].max()}")
print(f"# of Floors range after 3.4:   {df['# of Floors'].min()} - {df['# of Floors'].max()}")

df.to_csv(os.path.join(PROCESSED_DIR, "houses_cleaned.csv"), index=False)
joblib.dump(df, os.path.join(PROCESSED_DIR, "houses_cleaned.pkl"))
print(f"\nSaved to {PROCESSED_DIR}: {df.shape}")

# ============================================================
# 3.5 Missing-value Handling
# ============================================================
print("\n" + "="*60)
print("STEP 3.5: MISSING-VALUE HANDLING")
print("="*60)

missing_before_35 = df.isna().sum()

# Recover from description (same evidence-based approach as 3.4) instead of a
# blanket median. Generic regex, not hardcoded to the one row it currently matches.
BEDROOM_PATTERN = re.compile(r'(\d+)\s*bed\s*rooms?\b', re.IGNORECASE)
BATHROOM_PATTERN = re.compile(r'(\d+)\s*bath\s*rooms?\b', re.IGNORECASE)

def extract_room_count(text, pattern):
    if pd.isna(text):
        return np.nan
    match = pattern.search(str(text))
    return float(match.group(1)) if match else np.nan

room_recovery_log = []
for col, pattern in [('Bedroom', BEDROOM_PATTERN), ('Bathroom', BATHROOM_PATTERN)]:
    for idx in df.index[df[col].isna()]:
        recovered = extract_room_count(df.loc[idx, 'description'], pattern)
        room_recovery_log.append({
            "Ad List": df.loc[idx, 'Ad List'],
            "Variable": col,
            "Original": np.nan,
            "Description evidence": recovered,
            "Final": recovered,
        })
        if pd.notna(recovered):
            df.loc[idx, col] = recovered

room_recovery_df = pd.DataFrame(room_recovery_log)
print("\n--- Bedroom / Bathroom recovery from description ---")
if room_recovery_df.empty:
    print("No missing Bedroom/Bathroom values found.")
else:
    print(room_recovery_df.to_string(index=False))

# Floor Range / Completion Year / # of Floors / Total Units / Parking Lot:
# median/mode + missing-indicator imputation is deferred to the 3.11
# modelling pipeline (fit on X_train only) to avoid test-set leakage. Left
# as real NaN here for EDA.

# Columns deferred to 3.7 (dropped) / 3.8 (transformed) are documented in
# notes_missing_value_decision_audit.md.

missing_after_35 = df.isna().sum()
change_summary = pd.DataFrame({
    "missing_before_3.5": missing_before_35,
    "missing_after_3.5": missing_after_35,
})
print("\n--- Missing count before vs after Section 3.5's direct actions ---")
print(change_summary[(change_summary["missing_before_3.5"] > 0) | (change_summary["missing_after_3.5"] > 0)])

df.to_csv(os.path.join(PROCESSED_DIR, "houses_cleaned.csv"), index=False)
joblib.dump(df, os.path.join(PROCESSED_DIR, "houses_cleaned.pkl"))
print(f"\nSaved to {PROCESSED_DIR}: {df.shape}")

# ============================================================
# 3.11 Train-Test Split (moved ahead of 3.7/3.8/3.9 and ahead of 3.6 -
# executed
# immediately after 3.6, before any feature engineering, encoding
# or selection, so every fit-dependent step below can be fit on
# X_train only and simply applied to X_test)
# ============================================================
print("\n" + "="*60)
print("STEP 3.11: TRAIN-TEST SPLIT")
print("="*60)

print("\n--- Split ---")

"""
Split happens before feature engineering (3.8), categorical encoding (3.9)
and feature selection (3.7) - not after, as in an earlier version of this
pipeline - so every statistic-fitting step in those sections (rare-category
merge thresholds, one-hot category lists, the Facilities vocabulary, the
missing-value imputation medians, 3.10's scaler) can be fit on X_train only
and applied to X_test, without having to specially reorder any one of them
ahead of the split.

Standard train_test_split (random_state=42), not a group-aware split -
simpler than the GroupShuffleSplit approach used in an earlier version of
this pipeline. See the known-limitation note below for the trade-off this
accepts.

X and y are split in a single call, not two separate calls, so X_train/
y_train are guaranteed to be the same rows (two separate calls risk
misaligned rows even with the same random_state).

Regression target (price, continuous), not classification, so no stratify=.
"""
X = df.drop(columns=['price'])
y = np.log(df['price'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape} | y_test:  {y_test.shape}")
print(f"Actual test proportion: {len(X_test) / (len(X_train) + len(X_test)):.3f}")
assert len(X_train) == len(y_train)
assert len(X_test) == len(y_test)

"""
Known limitation: once identifier columns (Ad List, Address, description,
Building Name, etc.) are dropped in 3.7 below, a handful of otherwise-
distinct listings become row-identical on the remaining feature columns -
different real properties that simply share every retained attribute. A
group-aware split (GroupShuffleSplit) would keep every such group on one
side of the split; the plain random split used here does not, so some of
these groups may be torn across train/test, meaning a small number of test
rows could be feature-identical to a training row. This is accepted here as
a trade-off for pipeline simplicity, not an oversight - the actual count of
groups affected is measured after feature selection/encoding and reported
in 3.12's summary table ("Row-identical groups split across train/test").
"""

# ============================================================
# 3.6 Outlier Treatment (moved after 3.11's split - boxplot, Z-score and
# Mahalanobis distance statistics are all fit on X_train only, the same
# leakage rule every other fit-dependent step in this pipeline follows.
# Detection/visualisation only - no correction is applied in this section
# yet, candidates are listed for manual review first.)
# ============================================================
print("\n" + "="*60)
print("STEP 3.6: OUTLIER TREATMENT")
print("="*60)

Z_THRESHOLD = 3

def boxplot_zscore_figure(train_series, label, filename):
    s = train_series.dropna()
    mean, std = s.mean(), s.std()
    z = (s - mean) / std
    z_outlier = z.abs() > Z_THRESHOLD

    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    iqr_outlier = (s < lo) | (s > hi)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(x=s, ax=axes[0], color="#4C72B0")
    axes[0].set_title(f"{label} (X_train) - Boxplot (IQR)")
    axes[0].set_xlabel(label)

    axes[1].scatter(z.index, z, s=12, alpha=0.5, color="#4C72B0", label="within threshold")
    axes[1].scatter(z.index[z_outlier], z[z_outlier], s=25, color="#C44E52", label=f"|Z| > {Z_THRESHOLD}")
    axes[1].axhline(Z_THRESHOLD, color="#C44E52", linestyle="--", linewidth=1)
    axes[1].axhline(-Z_THRESHOLD, color="#C44E52", linestyle="--", linewidth=1)
    axes[1].set_title(f"{label} (X_train) - Z-score")
    axes[1].set_xlabel("Row index")
    axes[1].set_ylabel("Z-score")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(DOWNLOADS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n{label}: IQR bounds [{lo:.1f}, {hi:.1f}], IQR outliers: {iqr_outlier.sum()}")
    print(f"{label}: Z-score outliers (|Z| > {Z_THRESHOLD}): {z_outlier.sum()}")

print("\n--- Property Size: boxplot (IQR) + Z-score ---")
boxplot_zscore_figure(X_train['Property Size'], "Property Size", "fig06_property_size_outliers.png")

print("\n--- Bedroom: boxplot (IQR) + Z-score ---")
print(f"(Bedroom Q1={X_train['Bedroom'].quantile(.25)}, Q3={X_train['Bedroom'].quantile(.75)} "
      f"- IQR may be degenerate if Q1==Q3)")
boxplot_zscore_figure(X_train['Bedroom'], "Bedroom", "fig07_bedroom_outliers.png")

print("\n--- Bathroom: boxplot (IQR) + Z-score ---")
print(f"(Bathroom Q1={X_train['Bathroom'].quantile(.25)}, Q3={X_train['Bathroom'].quantile(.75)} "
      f"- IQR may be degenerate if Q1==Q3)")
boxplot_zscore_figure(X_train['Bathroom'], "Bathroom", "fig08_bathroom_outliers.png")

print("\n--- Parking Lot: boxplot (IQR) + Z-score ---")
boxplot_zscore_figure(X_train['Parking Lot'], "Parking Lot", "fig11_parking_lot_outliers.png")


def mahalanobis_figure(pair_df, xcol, ycol, title, filename):
    # Reindex to [xcol, ycol] explicitly - mean_vec/cov/ellipse below are all
    # computed in this column order, so it must match the plot's x/y axes
    # regardless of what order the caller's columns happen to be in.
    pair = pair_df[[xcol, ycol]].dropna()
    mean_vec = pair.mean().values
    cov = np.cov(pair.values, rowvar=False)
    inv_cov = np.linalg.inv(cov)
    diff = pair.values - mean_vec
    mahal_sq = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
    threshold = chi2.ppf(0.975, df=2)
    outlier = mahal_sq > threshold

    # Confidence ellipse at the same chi2 threshold as the outlier cutoff -
    # its axes come from the covariance matrix's eigenvectors/eigenvalues,
    # so the ellipse is the direct visual boundary of "Mahalanobis distance
    # <= threshold": any point outside it is exactly what's flagged red.
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * np.sqrt(eigvals * threshold)

    fig, ax = plt.subplots(figsize=(9, 6))
    ellipse = Ellipse(xy=mean_vec, width=width, height=height, angle=angle,
                       facecolor="#4C72B0", alpha=0.15, edgecolor="gray",
                       linewidth=1.5, linestyle="--", zorder=1)
    ax.add_patch(ellipse)
    ax.scatter(pair.loc[~outlier, xcol], pair.loc[~outlier, ycol], s=15, alpha=0.6,
               color="black", label="within threshold", zorder=2)
    ax.scatter(pair.loc[outlier, xcol], pair.loc[outlier, ycol], s=40,
               color="#DD8452", label="Mahalanobis outlier", zorder=3)
    ax.scatter(*mean_vec, color="#4C72B0", marker="X", s=120, label="mean", zorder=4)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(f"{title} (X_train, Mahalanobis Distance)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DOWNLOADS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Mahalanobis threshold (chi2, df=2, 97.5%): {threshold:.2f}")
    print(f"Outliers flagged: {outlier.sum()} / {len(pair)}")
    return pair.index[outlier]

print("\n--- Total Units vs # of Floors: Mahalanobis Distance ---")
tu_outlier_idx = mahalanobis_figure(
    X_train[['Total Units', '# of Floors']], '# of Floors', 'Total Units',
    "Total Units vs # of Floors", "fig09_units_floors_mahalanobis.png"
)
print("Outlier Ad List (for manual review, no correction applied yet):")
print(X_train.loc[tu_outlier_idx, ['Ad List', 'Total Units', '# of Floors']].to_string(index=False))

print("\n--- Parking Lot vs price: Mahalanobis Distance ---")
# y_train is log(price) (see 3.11 above) - converted back to RM here for interpretability.
price_train_raw = np.exp(y_train)
pl_pair = pd.DataFrame({'Parking Lot': X_train['Parking Lot'], 'price': price_train_raw})
pl_outlier_idx = mahalanobis_figure(
    pl_pair, 'Parking Lot', 'price', "Parking Lot vs price", "fig10_parking_price_mahalanobis.png"
)
print("Outlier Ad List (for manual review, no correction applied yet):")
pl_review = X_train.loc[pl_outlier_idx, ['Ad List', 'Parking Lot']].copy()
pl_review['price'] = price_train_raw.loc[pl_outlier_idx]
print(pl_review.to_string(index=False))

print("\n--- Confirmed corrections (manually reviewed against description) ---")

"""
Applied directly to whichever of X_train/X_test the row falls in - these are
description-verified facts (not a fitted statistic like a median or an IQR
bound), so there is no train/test leakage concern in correcting both splits
the same way. More candidates from the outlier lists above remain under
review and will be added here as they're confirmed.
"""
def apply_confirmed_correction(adlist, col, value, note):
    for split_name, frame in [('X_train', X_train), ('X_test', X_test)]:
        mask = frame['Ad List'] == adlist
        if mask.any():
            frame.loc[mask, col] = value
            print(f"Ad List {adlist}: {col} -> {value} in {split_name} ({note})")
            return
    print(f"Ad List {adlist}: not found in X_train or X_test")

"""
Property Size: rule-based re-check of every row the Z-score screen above
(fig06) flags, not a one-off patch for the 2 rows found so far - if a
future data pull changes which rows are extreme, this rule still runs
against whatever fig06 flags at that time. Priority order: an explicit
'Land Area' label overrides a 'Built Up'/generic mention (the buyer-facing
land size is preferred, e.g. Ad List 103792765's 9800 Built Up vs 6800 Land
Area); otherwise a ~10x/100x/1000x ratio against the generic mention is a
digit-shift extraction artefact (e.g. 122774 -> 1227.74, 14500 -> 1450);
no description evidence at all drops the value to NaN (e.g. 103729938,
'For sale' only); anything else means the description independently
confirms the original value (e.g. 96973074, 9376 both sides), so it is
left unchanged. Run against X_train AND X_test - X_test's own flagged list
may be empty, but the rule still executes against it either way.
"""
LAND_AREA_PATTERN = re.compile(r'Land\s*Area\s*[:\-]?\s*(\d[\d,]*\.?\d*)\s*sq\.?\s*ft', re.IGNORECASE)

def extract_size_with_source(text):
    if pd.isna(text):
        return np.nan, False
    text = str(text)
    m = LAND_AREA_PATTERN.search(text)
    if m:
        return float(m.group(1).replace(',', '')), True
    m = SIZE_PATTERN.search(text)
    return (float(m.group(1).replace(',', '')), False) if m else (np.nan, False)

print("\n--- Property Size: description-based rule applied to Z-score outliers ---")
for split_name, frame in [('X_train', X_train), ('X_test', X_test)]:
    s = frame['Property Size'].dropna()
    z = (s - s.mean()) / s.std()
    flagged_idx = z.index[z.abs() > Z_THRESHOLD]
    if len(flagged_idx) == 0:
        print(f"{split_name}: no Property Size Z-score outliers - rule has no rows to apply to")
        continue
    # Cast to float first - a correction below may introduce NaN or a
    # fractional value (e.g. 1227.74), neither of which fits the int64
    # dtype Property Size was cast to back in 3.4.
    frame['Property Size'] = frame['Property Size'].astype(float)
    for idx in flagged_idx:
        original = frame.loc[idx, 'Property Size']
        desc_size, is_land_area = extract_size_with_source(frame.loc[idx, 'description'])
        if pd.isna(desc_size):
            corrected, note = np.nan, "no size evidence in description -> NaN"
        elif is_land_area:
            corrected, note = round(desc_size), f"Land Area preferred over Built Up ({original} -> {desc_size:.0f})"
        else:
            ratio = original / desc_size if pd.notna(original) and desc_size else np.nan
            if pd.notna(ratio) and ratio > 0 and any(abs(ratio / p - 1) < 0.05 for p in [10, 100, 1000]):
                corrected, note = round(desc_size), f"digit-shift artefact corrected ({original} -> {desc_size:.0f})"
            else:
                corrected, note = original, f"description confirms original value ({original}), no change"
        frame.loc[idx, 'Property Size'] = corrected
        print(f"Ad List {int(frame.loc[idx, 'Ad List'])} in {split_name}: {note}")

"""
Bedroom: rule-based re-check, but scoped to Bedroom >= 8 only - NOT the full
Z-score outlier list (fig07 flags 37 rows in X_train alone, most of them
ordinary 5-6 bedroom units). A blanket regex against the full list was tried
and rejected here after producing false positives of exactly the kind
already documented dataset-wide in notes/bedroom_bathroom_regex_false_
positives.csv: (a) "SQFT : 3000\\nBEDROOMS : 6" - the earlier [\\s-]* pattern
crossed the newline and grabbed the SQFT figure instead; (b) "4+1 bedroom"
(a common Malaysian listing convention meaning 5 total) - the regex can only
ever capture the "1" immediately before "bedroom", silently corrupting a
correct value of 5. Restricting to >=8 keeps the scope to genuinely extreme,
individually-checkable rows (2 in X_train: Ad List 96822478 and 102236931),
and [^\\S\\n]* (whitespace but not newline) replaces the old \\s* so a match
can no longer span two lines. Run against X_train AND X_test even though
X_test's >=8 list may be empty - the loop still executes against it.
"""
BEDROOM_DESC_PATTERN = re.compile(r'(\d+)[^\S\n]*-?[^\S\n]*bed[^\S\n]*-?[^\S\n]*rooms?\b', re.IGNORECASE)

def extract_bedroom_from_description(text):
    if pd.isna(text):
        return np.nan
    m = BEDROOM_DESC_PATTERN.search(str(text))
    return float(m.group(1)) if m else np.nan

print("\n--- Bedroom: description-based rule applied to Bedroom >= 8 ---")
for split_name, frame in [('X_train', X_train), ('X_test', X_test)]:
    flagged_idx = frame.index[frame['Bedroom'] >= 8]
    if len(flagged_idx) == 0:
        print(f"{split_name}: no Bedroom >= 8 rows - rule has no rows to apply to")
        continue
    corrected_count = 0
    for idx in flagged_idx:
        original = frame.loc[idx, 'Bedroom']
        desc_val = extract_bedroom_from_description(frame.loc[idx, 'description'])
        adlist = int(frame.loc[idx, 'Ad List'])
        if pd.isna(desc_val):
            print(f"Ad List {adlist} in {split_name}: no bedroom count found in description, left unchanged ({original})")
        elif desc_val != original:
            print(f"Ad List {adlist} in {split_name}: Bedroom {original} -> {desc_val:.0f} (description states '{desc_val:.0f}-Bedrooms')")
            frame.loc[idx, 'Bedroom'] = desc_val
            corrected_count += 1
        else:
            print(f"Ad List {adlist} in {split_name}: description confirms original value ({original}), no change")
    print(f"{split_name}: {corrected_count}/{len(flagged_idx)} Bedroom >= 8 rows corrected via description match")

apply_confirmed_correction(103207012, 'Bathroom', 2,
                            "description states '2 Bathrooms' in both unit configurations listed")

"""
Parking Lot: second-layer domain-knowledge filter on top of the Mahalanobis
candidates above - many parking lots is only suspicious paired with a price
far below what that many parking lots would normally command (Parking Lot
and price are positively correlated, r=0.42 on X_train). ">=4" is the first
integer past Parking Lot's own IQR upper bound (3.5, from the boxplot
above); "50% of median" marks a price far enough under the X_train typical
price that ordinary variation doesn't explain it. No independent source
confirms the true count, so these become NaN (like Total Units==1 in 3.4),
not a corrected value - deferred to 3.11 imputation. The threshold itself is
computed from X_train's price median only and re-used as-is for X_test, so
no test-set statistic leaks into the rule.
"""
parking_lot_price_threshold = price_train_raw.median() * 0.5
print(f"\nParking Lot low-price threshold (50% of X_train median price): RM{parking_lot_price_threshold:,.0f}")
for split_name, frame, price_series in [
    ('X_train', X_train, price_train_raw),
    ('X_test', X_test, np.exp(y_test)),
]:
    mask = (frame['Parking Lot'] >= 4) & (price_series < parking_lot_price_threshold)
    if mask.any():
        print(f"Parking Lot -> NaN in {split_name} (>=4 lots, price below threshold):")
        review = frame.loc[mask, ['Ad List', 'Parking Lot']].copy()
        review['price'] = price_series.loc[mask]
        print(review.to_string(index=False))
        frame.loc[mask, 'Parking Lot'] = np.nan
    else:
        print(f"Parking Lot -> NaN in {split_name}: no matching rows")

# ============================================================
# 3.8 Feature Engineering (moved after 3.11's split - every rule
# here is a fixed transformation with no data-dependent fitting,
# so it is applied to X_train and X_test independently)
# ============================================================
print("\n" + "="*60)
print("STEP 3.8: FEATURE ENGINEERING")
print("="*60)

print("\n--- State (extracted from Address) ---")

"""
Address format is inconsistent across rows - not just varying prefixes, but
the state isn't always the last comma-separated segment. Some listings write
"..., State, City" instead of "..., City, State" (e.g. "Puchong, Selangor,
Puchong"), which would make a last-segment-only extraction miss the state
entirely (confirmed: all 152 addresses where the naive last-segment approach
failed turned out to have the state one or more segments earlier, not
missing). So every segment is checked, scanning from the end backwards, and
matched against a fixed list of real Malaysian states - exact whole-segment
equality, not substring search, so a place name that merely contains a
state's name (e.g. "Kampung Melaka") is never mistaken for the state
"Melaka".

Matching is case-insensitive (e.g. "SABAH", "putrajaya") - two rows in this
dataset have a wrong-case state segment, currently masked only by a
correctly-cased duplicate elsewhere in the same address; without case-
insensitivity a future row with no such backup would be wrongly marked NaN.

This is a fixed rule with nothing fitted from data, so it's applied to
X_train and X_test independently below - unlike the rare-category merge and
one-hot encoding built from 'State' in 3.9, which must be fit on X_train only.
"""
VALID_STATES = {'Selangor', 'Penang', 'Kuala Lumpur', 'Johor', 'Sabah', 'Sarawak',
                 'Perak', 'Kedah', 'Pahang', 'Negeri Sembilan', 'Melaka',
                 'Terengganu', 'Kelantan', 'Perlis', 'Putrajaya', 'Labuan'}
VALID_STATES_LOWER = {s.lower(): s for s in VALID_STATES}

def extract_state_from_address(address):
    if pd.isna(address):
        return np.nan
    segments = [s.strip() for s in address.split(',')]
    for segment in reversed(segments):
        if segment.lower() in VALID_STATES_LOWER:
            return VALID_STATES_LOWER[segment.lower()]
    return np.nan

print("\n--- Property Age (from Completion Year) ---")

"""
Property Age is computed against a fixed reference year, not whatever year
the script happens to run in - Completion Year is a fixed fact about the
property, and this dataset was collected in September 2023 (per the Kaggle
source), so ages must be measured from then, not from today's calendar year.
A negative age (Completion Year after REFERENCE_YEAR) isn't invalid data -
it means the unit was still under construction / sold off-plan at the time
of collection, so Property Age is correctly left NaN for these rows.
"""
REFERENCE_YEAR = 2023

def compute_property_age(completion_year):
    if pd.isna(completion_year):
        return np.nan
    age = REFERENCE_YEAR - completion_year
    return age if age >= 0 else np.nan

print("\n--- Listed_Facility_Count (from Facilities) ---")

"""
Purely numeric items are dropped before counting - Ad List 95706905's
Facilities value ends in a stray "10" that isn't a real facility name (a
scraping artefact), which would otherwise silently inflate its count by 1.
"""
def count_facilities(facilities_text):
    if pd.isna(facilities_text):
        return 0
    items = [item.strip() for item in facilities_text.split(',')]
    return len([item for item in items if item and not item.isdigit()])

print("\n--- Has_X flags (from 'nearby amenity' text columns) ---")

"""
Bus Stop / Mall / Park / School / Hospital / Highway / Railway Station each
list a specific nearby amenity name as free text, with missing rates
between 76% and 96% (see notes_missing_value_decision_audit.md). The
specific name has too little repeat structure to be usable directly, but
whether the field was filled in at all is itself a signal. Converting to a
presence flag also removes the missing-value problem entirely - every row
gets 0 or 1. Note this flag means "information was recorded", not "amenity
confirmed absent" - a 0 does not prove there is no mall nearby, only that it
wasn't written.
"""
NEARBY_COLUMNS = ['Bus Stop', 'Mall', 'Park', 'School', 'Hospital', 'Highway', 'Railway Station']

def engineer_features(X, label):
    X['State'] = X['Address'].apply(extract_state_from_address)
    address_missing = X['Address'].isna().sum()
    no_state_in_any_segment = X['State'].isna().sum() - address_missing
    print(f"[{label}] Address missing (-> State=NaN): {address_missing} | "
          f"no segment matched a state: {no_state_in_any_segment} | "
          f"total State missing: {X['State'].isna().sum()}")

    X['Property Age'] = X['Completion Year'].apply(compute_property_age)
    completion_missing = X['Completion Year'].isna().sum()
    off_plan_count = (X['Completion Year'] > REFERENCE_YEAR).sum()
    print(f"[{label}] Completion Year missing (-> Property Age=NaN): {completion_missing} | "
          f"off-plan (Completion Year > {REFERENCE_YEAR}): {off_plan_count} | "
          f"Property Age missing: {X['Property Age'].isna().sum()}")

    X['Listed_Facility_Count'] = X['Facilities'].apply(count_facilities)
    print(f"[{label}] Facilities missing (-> Listed_Facility_Count=0): {X['Facilities'].isna().sum()} | "
          f"Listed_Facility_Count range: {X['Listed_Facility_Count'].min()}-{X['Listed_Facility_Count'].max()}")

    for col in NEARBY_COLUMNS:
        flag_col = 'Has_' + col.replace(' ', '_')
        X[flag_col] = X[col].notna().astype(int)
    print(f"[{label}] Has_X flags created: {[('Has_' + c.replace(' ', '_')) for c in NEARBY_COLUMNS]}")

    return X

X_train = engineer_features(X_train, "X_train")
X_test = engineer_features(X_test, "X_test")

"""
State's rare-category merge and one-hot encoding, and Property Type's
rare-category merge and one-hot encoding, are both statistics computed from
data (which categories are common enough to keep), so they are fit on
X_train only and applied to X_test - see 3.9 below, not here. 'State' is
kept here as plain extracted text for that reason.
"""

"""
--- Features tried in 3.8 and dropped (kept here as a record, not in X) ---

Is_Off_Plan: 1 if Completion Year > REFERENCE_YEAR (still under construction
at collection time), 0 if completed, NaN if Completion Year itself missing -
tried so the off-plan signal wouldn't collapse into the same NaN as a
genuinely unknown Completion Year. Dropped: only 22/3793 (0.6%) positive
cases, and r=-0.0004, p=0.986 against price - not significant.

Facilities_Recorded: a presence flag paired with Listed_Facility_Count.
Dropped after verifying Listed_Facility_Count == 0 for exactly the same 607
rows where Facilities is missing, with zero exceptions - pure redundancy.

Nearby_Amenity_Count: sum of the 7 Has_X flags into a single 0-7 score.
Dropped - a pure linear combination of columns already in X, and
empirically low-value too: r=-0.019, p=0.242 against price (n=3793), not
statistically significant.

Units per Floor: Total Units / # of Floors. Dropped because several rows
produce physically impossible values (e.g. Ad List 100502825, Total
Units=7810 / # of Floors=5 = 1562 units on one floor). Left out until that
upstream data quality issue is resolved with evidence.
"""

print(f"\nX_train shape after 3.8: {X_train.shape} | X_test shape after 3.8: {X_test.shape}")

# ============================================================
# 3.9 Categorical Encoding (executed after 3.8, before 3.7's
# drop - fixed-rule encodings are applied to X_train/X_test
# independently; State and Property Type's rare-category merge
# and one-hot encoding are fit on X_train only, then applied to
# X_test, since the merge threshold is a statistic of the data)
# ============================================================
print("\n" + "="*60)
print("STEP 3.9: CATEGORICAL ENCODING")
print("="*60)

print("\n--- Land Title: rare-category merge, then binary encoding ---")

"""
Land Title has 3 categories: Non Bumi Lot, Bumi Lot, Malay Reserved. Malay
Reserved is under 0.2% of records - the same sparse-category problem as
Property Type's rare types. Merged into Bumi Lot rather than one-hot as its
own column: both Bumi Lot and Malay Reserved represent land with restricted
purchase eligibility (Bumiputera-only), in contrast to Non Bumi Lot's
unrestricted eligibility - a real, domain-based grouping rather than a
statistic computed from data, so it's applied to X_train and X_test
independently rather than fit on X_train only.

That merge leaves only 2 categories, so a full one-hot would just produce a
second column that's the exact logical negation of the first - a binary
flag is more direct.

Uses .map() rather than a `== 'Non Bumi Lot'` comparison - `==` against NaN
silently returns False (misclassified as Bumi Lot) instead of staying NaN.
"""
for _X, _label in [(X_train, "X_train"), (X_test, "X_test")]:
    _X['Land Title'] = _X['Land Title'].replace('Malay Reserved', 'Bumi Lot')
    _X['Is_Non_Bumi_Lot'] = _X['Land Title'].map({'Non Bumi Lot': 1, 'Bumi Lot': 0})
    print(f"[{_label}] 'Is_Non_Bumi_Lot' value counts:\n{_X['Is_Non_Bumi_Lot'].value_counts(dropna=False)}")
print("\n" + "-"*60)

print("\n--- Tenure Type: binary encoding ---")

"""
Only 2 categories (Freehold / Leasehold, 0 missing) - a full one-hot would
produce a second column that's the exact logical negation of the first (the
"dummy variable trap"), so a single binary indicator carries the same
information with one fewer redundant column. Fixed rule, applied to
X_train/X_test independently.

Uses .map() rather than a `== 'Freehold'` comparison, for the same
NaN-safety reason as Land Title above.
"""
for _X, _label in [(X_train, "X_train"), (X_test, "X_test")]:
    _X['Freehold Indicator'] = _X['Tenure Type'].map({'Freehold': 1, 'Leasehold': 0})
    print(f"[{_label}] 'Freehold Indicator' value counts:\n{_X['Freehold Indicator'].value_counts(dropna=False)}")
print("\n" + "-"*60)

print("\n--- Floor Range: ordinal encoding ---")

"""
Low/Medium/High have a real order (higher floor is a meaningfully different
attribute, not an arbitrary label), so ordinal encoding preserves that
monotonic relationship instead of discarding it into unordered one-hot
columns. "Unknown" (filled in back in 3.5) doesn't fit that order, so it
maps to NaN here rather than being forced into a fake rank. Fixed mapping
rule, applied to X_train/X_test independently.

Filling Floor_Range_Ordinal's NaN is deliberately NOT done here - it happens
in the imputation step below, using only X_train's median, for the same
leakage reason the other numeric imputations are deferred there.
"""
for _X, _label in [(X_train, "X_train"), (X_test, "X_test")]:
    _X['Floor_Range_Ordinal'] = _X['Floor Range'].map({'Low': 1, 'Medium': 2, 'High': 3})
    print(f"[{_label}] 'Floor_Range_Ordinal' value counts (NaN=Unknown, imputed later):\n"
          f"{_X['Floor_Range_Ordinal'].value_counts(dropna=False)}")
print("\n" + "-"*60)

print("\n--- Facilities: text cleanup + multi-hot encoding (vocabulary fit on X_train only) ---")

"""
Cleanup pipeline, applied before binarising: split on comma, strip
whitespace, title-case for consistency, drop purely numeric junk (the same
"10" scraping artefact filtered in Listed_Facility_Count's Ad List
95706905).

The full facility vocabulary is fixed (14 distinct names, verified earlier
across the whole dataset, no synonyms/case variants), so it does not
normally differ between X_train and X_test - but it is still, technically, a
statistic read off the data, so it's fit (learned) on X_train's facility
lists only via MultiLabelBinarizer. X_test is then built by reindexing
against exactly the columns X_train produced: any facility name seen only in
X_test (not seen in X_train) is silently ignored rather than creating a new
column - same reindex-and-fill-0 pattern used for State/Property Type's
one-hot below.
"""
def clean_facility_list(facilities_text):
    if pd.isna(facilities_text):
        return []
    items = [item.strip().title() for item in facilities_text.split(',')]
    return [item for item in items if item and not item.isdigit()]

train_facility_lists = X_train['Facilities'].apply(clean_facility_list)
test_facility_lists = X_test['Facilities'].apply(clean_facility_list)

mlb = MultiLabelBinarizer()
facility_encoded_train = pd.DataFrame(
    mlb.fit_transform(train_facility_lists),
    columns=['Has_' + c.replace(' ', '_') for c in mlb.classes_],
    index=X_train.index
).astype(int)

facility_encoded_test = pd.DataFrame(0, index=X_test.index, columns=facility_encoded_train.columns)
unseen_test_facilities = set()
for idx, items in zip(X_test.index, test_facility_lists):
    for item in items:
        col = 'Has_' + item.replace(' ', '_')
        if col in facility_encoded_test.columns:
            facility_encoded_test.loc[idx, col] = 1
        else:
            unseen_test_facilities.add(item)

X_train = pd.concat([X_train, facility_encoded_train], axis=1)
X_test = pd.concat([X_test, facility_encoded_test], axis=1)

print(f"Distinct facility types found in X_train: {len(mlb.classes_)}")
print(f"Columns created: {list(facility_encoded_train.columns)}")
print(f"Facility names seen in X_test but not X_train (dropped, not encoded): {unseen_test_facilities or 'none'}")
print(f"\nX_train column totals:\n{facility_encoded_train.sum().sort_values(ascending=False)}")
print("\n" + "-"*60)

print("\n--- State: rare-category merge + one-hot encoding (fit on X_train only) ---")

"""
State has no inherent order between categories (Selangor isn't "more" or
"less" than Penang), so one-hot is the right encoding - unlike Floor Range,
which gets ordinal encoding because Low/Medium/High has a real order.

NaN is turned into an explicit "Unknown" category before encoding, not left
as NaN - get_dummies() would otherwise silently mark all dummy columns as 0
for missing rows, identical to "known and simply not any of these states"
rather than "state genuinely not on record". This fill is a constant, not a
fitted statistic, so it's safe to apply to X_train and X_test independently.

The <10-listings rare-category threshold IS a fitted statistic (unlike the
fill above, it depends on which categories are common enough to keep), so
it's computed from X_train's value_counts() ONLY, then the same rare-category
list is applied to X_test's State column - the same leakage rule used for
the imputation medians below and 3.10's scaler.

X_test's State is cast to a Categorical using X_train's post-merge category
list (not encoded independently) - this guarantees get_dummies drops the
same reference category on both sides. Any category X_train never saw at all
becomes NaN in X_test and gets an all-zero dummy row, the same as a proper
unseen-category fallback in a deployed model. The reindex below is then just
a safety net for column order/coverage.
"""
X_train['State'] = X_train['State'].fillna('Unknown')
X_test['State'] = X_test['State'].fillna('Unknown')

RARE_STATE_THRESHOLD = 10
train_state_counts = X_train['State'].value_counts()
rare_states = train_state_counts[train_state_counts < RARE_STATE_THRESHOLD].index.tolist()
print(f"Rare states in X_train (<{RARE_STATE_THRESHOLD} listings), merged into 'Other': {rare_states}")

X_train['State'] = X_train['State'].replace(rare_states, 'Other')
X_test['State'] = X_test['State'].replace(rare_states, 'Other')

state_categories = sorted(X_train['State'].unique())
X_train['State'] = pd.Categorical(X_train['State'], categories=state_categories)
# Any X_test value absent from X_train entirely (e.g. a singleton state that
# landed only in the test split) isn't in state_categories and isn't in
# rare_states either (rare_states only lists values X_train actually saw) -
# cleared to NaN explicitly before the Categorical cast, since pandas now
# deprecates letting the cast do that implicitly.
X_test['State'] = X_test['State'].where(X_test['State'].isin(state_categories))
X_test['State'] = pd.Categorical(X_test['State'], categories=state_categories)

state_dummies_train = pd.get_dummies(X_train['State'], prefix='State', drop_first=True).astype(int)
state_dummies_train.columns = [c.replace(' ', '_') for c in state_dummies_train.columns]
state_dummies_test = pd.get_dummies(X_test['State'], prefix='State', drop_first=True).astype(int)
state_dummies_test.columns = [c.replace(' ', '_') for c in state_dummies_test.columns]
state_dummies_test = state_dummies_test.reindex(columns=state_dummies_train.columns, fill_value=0)

X_train = pd.concat([X_train, state_dummies_train], axis=1)
X_test = pd.concat([X_test, state_dummies_test], axis=1)

print(f"'State' one-hot columns created ({len(state_dummies_train.columns)}): {list(state_dummies_train.columns)}")
print(state_dummies_train.sum().sort_values(ascending=False))
print("\n" + "-"*60)

print("\n--- Property Type: rare-category merge + one-hot encoding (fit on X_train only) ---")

"""
Same fit-on-X_train-only reasoning as State above, with a <20-listings
threshold. Property Type has no missing values (unlike State), so no
"Unknown" fill is needed first. Duplex / Townhouse Condo / Studio / Others
each have fewer than 20 listings out of 3793 - one-hot on these individually
would produce columns that are almost entirely 0.
"""
RARE_PROPERTY_TYPE_THRESHOLD = 20
train_property_type_counts = X_train['Property Type'].value_counts()
rare_property_types = train_property_type_counts[train_property_type_counts < RARE_PROPERTY_TYPE_THRESHOLD].index.tolist()
print(f"'Property Type' counts in X_train before merge:\n{train_property_type_counts}\n")
print(f"Rare Property Types in X_train (<{RARE_PROPERTY_TYPE_THRESHOLD} listings), merged into 'Other': {rare_property_types}")

X_train['Property Type'] = X_train['Property Type'].replace(rare_property_types, 'Other')
X_test['Property Type'] = X_test['Property Type'].replace(rare_property_types, 'Other')

property_type_categories = sorted(X_train['Property Type'].unique())
X_train['Property Type'] = pd.Categorical(X_train['Property Type'], categories=property_type_categories)
X_test['Property Type'] = X_test['Property Type'].where(X_test['Property Type'].isin(property_type_categories))
X_test['Property Type'] = pd.Categorical(X_test['Property Type'], categories=property_type_categories)

property_type_dummies_train = pd.get_dummies(X_train['Property Type'], prefix='PropertyType', drop_first=True).astype(int)
property_type_dummies_train.columns = [c.replace(' ', '_') for c in property_type_dummies_train.columns]
property_type_dummies_test = pd.get_dummies(X_test['Property Type'], prefix='PropertyType', drop_first=True).astype(int)
property_type_dummies_test.columns = [c.replace(' ', '_') for c in property_type_dummies_test.columns]
property_type_dummies_test = property_type_dummies_test.reindex(columns=property_type_dummies_train.columns, fill_value=0)

X_train = pd.concat([X_train, property_type_dummies_train], axis=1)
X_test = pd.concat([X_test, property_type_dummies_test], axis=1)

print(f"\n'Property Type' counts in X_train after merge:\n{X_train['Property Type'].value_counts()}")
print(f"\nOne-hot columns created ({len(property_type_dummies_train.columns)}): {list(property_type_dummies_train.columns)}")
print("\n" + "-"*60)

print(f"\nX_train shape after 3.9: {X_train.shape} | X_test shape after 3.9: {X_test.shape}")

# ============================================================
# 3.7 Feature Selection (executed after 3.8/3.9, since it drops
# columns that 3.8/3.9 needed to still exist while extracting
# from them - applied to X_train/X_test independently using the
# same fixed column lists on each side)
# ============================================================
print("\n" + "="*60)
print("STEP 3.7: FEATURE SELECTION")
print("="*60)

print("\n--- Stage A: columns with no predictive value, unrelated to any 3.8/3.9 engineering ---")

"""
These were never going to be used, encoded or otherwise - dropped purely on
their own merits (unstructured text, unique identifier, zero variance,
agent/listing-firm metadata unrelated to the property, or high cardinality
with no extractable pattern), independent of anything built in 3.8/3.9.
"""
cols_no_engineering = [
    'description', 'Ad List',
    'Nearby School', 'Nearby Mall', 'Nearby Railway Station',
    'Category',
    'Firm Type', 'Firm Number', 'REN Number',
    'Building Name', 'Developer',
]

print("Non-null counts before drop (X_train):")
for c in cols_no_engineering:
    print(f"  {c:25s} {X_train[c].notna().sum()} non-null")

print("\n--- Stage B: raw columns already superseded by a 3.8/3.9 engineered feature ---")

"""
Each of these was kept alive specifically so 3.8/3.9 could extract from it
(Has_X flags from the 7 nearby-amenity columns, Freehold Indicator from
Tenure Type, Is_Non_Bumi_Lot from Land Title, Floor_Range_Ordinal from
Floor Range, multi-hot from Facilities, one-hot from State/Property Type -
both fit on X_train and applied to X_test already, in 3.9 above). Now that
every extraction/encoding is done, the raw source column is redundant with
its own derived feature(s) already in X.

Completion Year is dropped for a different, stronger reason than the
others: verified Property Age == REFERENCE_YEAR - Completion Year exactly,
for every non-off-plan row, with zero exceptions. This isn't approximate
overlap - it's the same variable in different units, so keeping both would
hand a linear model perfectly collinear inputs for no benefit.
"""
cols_replaced_by_engineering = [
    'Address',
    'Completion Year',
    'Bus Stop', 'Mall', 'Park', 'School', 'Hospital', 'Highway', 'Railway Station',
    'Tenure Type',
    'Land Title',
    'Floor Range',
    'Facilities',
    'State',
    'Property Type',
]

print("Non-null counts before drop (X_train):")
for c in cols_replaced_by_engineering:
    print(f"  {c:25s} {X_train[c].notna().sum()} non-null")

shape_before_drop_train = X_train.shape
shape_before_drop_test = X_test.shape
X_train = X_train.drop(columns=cols_no_engineering + cols_replaced_by_engineering)
X_test = X_test.drop(columns=cols_no_engineering + cols_replaced_by_engineering)

print(f"\nX_train shape before 3.7 drop: {shape_before_drop_train} -> after: {X_train.shape}")
print(f"X_test shape before 3.7 drop:  {shape_before_drop_test} -> after: {X_test.shape}")
print(f"Columns dropped: {len(cols_no_engineering) + len(cols_replaced_by_engineering)}")
print("\n" + "-"*60)

# ============================================================
# Missing-value imputation (median fill, fit on X_train only and
# applied to X_test - executed after 3.7's drop)
# ============================================================
print("\n" + "="*60)
print("MISSING-VALUE IMPUTATION")
print("="*60)

"""
These columns still carry real NaN, deliberately left unfilled since 3.5 /
3.8's Property Age computation, since imputing before the split existed
would compute a statistic from data the model shouldn't have seen yet. The
median itself comes from X_train.median() (skipna=True by default, so
already-observed values compute it correctly), then the SAME value is used
to fill X_test - X_test's own median is never touched, which is what avoids
leaking test-set information into the imputation.

Property Age / # of Floors / Total Units / Parking Lot each get a
missing-indicator flag before the median fill - regression testing (R² and
a t-test on each flag's coefficient) confirmed missingness itself is
statistically significant against price for all four (p < 1e-16), so an
imputed median and a genuinely-observed value are NOT the same information
here, and collapsing that distinction would hide a real signal from the
model.

Property Size is excluded from flagging - it has only 1 missing value in
the raw data, too small a sample for the flag's coefficient to support any
meaningful statistical inference (same reasoning already used in 3.8 to
drop Is_Off_Plan for having too few positive cases). Floor_Range_Ordinal is
likewise median-filled only, with no flag added.
"""
IMPUTE_FLAG_COLS = {
    'Property Age': 'Property_Age_Missing',
    '# of Floors': 'Num_Floors_Missing',
    'Total Units': 'Total_Units_Missing',
    'Parking Lot': 'Parking_Lot_Missing',
}
IMPUTE_NO_FLAG_COLS = ['Property Size', 'Floor_Range_Ordinal']

for col, flag_col in IMPUTE_FLAG_COLS.items():
    X_train[flag_col] = X_train[col].isna().astype(int)
    X_test[flag_col] = X_test[col].isna().astype(int)

    train_median = X_train[col].median()
    X_train[col] = X_train[col].fillna(train_median)
    X_test[col] = X_test[col].fillna(train_median)

    print(f"{col:20s} train median={train_median:>8.2f} | train missing={X_train[flag_col].sum():>4d} | test missing={X_test[flag_col].sum():>4d} | flag={flag_col}")

for col in IMPUTE_NO_FLAG_COLS:
    train_missing = X_train[col].isna().sum()
    test_missing = X_test[col].isna().sum()

    train_median = X_train[col].median()
    X_train[col] = X_train[col].fillna(train_median)
    X_test[col] = X_test[col].fillna(train_median)

    print(f"{col:20s} train median={train_median:>8.2f} | train missing={train_missing:>4d} | test missing={test_missing:>4d} | no flag")

print(f"\nRemaining NaN in X_train: {X_train.isna().sum().sum()}")
print(f"Remaining NaN in X_test:  {X_test.isna().sum().sum()}")
assert X_train.isna().sum().sum() == 0
assert X_test.isna().sum().sum() == 0

X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)
joblib.dump((X_train, X_test, y_train, y_test), os.path.join(PROCESSED_DIR, "train_test_split.pkl"))
print(f"\nSaved X_train/X_test/y_train/y_test to {PROCESSED_DIR}")

# ============================================================
# 3.10 Feature Scaling (executed after imputation, using only
# X_train to fit, so no test-set information leaks into the
# scaling statistics)
# ============================================================
print("\n" + "="*60)
print("STEP 3.10: FEATURE SCALING")
print("="*60)

print("\n--- StandardScaler, fit on X_train only ---")

"""
StandardScaler (not MinMaxScaler) - this dataset has genuine extreme values
(e.g. Total Units up to several thousand), and min-max scaling would let a
single outlier compress the entire rest of the distribution into a tiny
range. StandardScaler's mean/std are less distorted by a few extreme points.

Only genuinely continuous/count numeric columns are scaled, plus
Floor_Range_Ordinal - price is the target, not a feature, so it's excluded
entirely. One-hot columns (State_*/PropertyType_*) and binary flags
(Has_*/Is_Non_Bumi_Lot/Freehold Indicator/the imputation *_Missing flags)
are left unscaled (already bounded 0/1, scaling a dummy isn't meaningful).
Floor_Range_Ordinal is included in scaling, though it
only has 3 levels - it has a real magnitude and order (1<2<3), not just
presence/absence, so leaving it on a raw 1-3 scale while every other numeric
feature is standardised to mean=0/std=1 would let it disproportionately
dominate or shrink in a distance-based model (KNN/SVR) purely from a unit
mismatch, unrelated to its actual importance.

scaler.fit() is called on X_train ONLY, then the same fitted scaler
transforms both X_train and X_test - X_test's own mean/std are never
computed or used, which is what avoids leaking test-set information into
the model's input scale.
"""
SCALE_COLS = ['Bedroom', 'Bathroom', 'Property Size', '# of Floors',
              'Total Units', 'Parking Lot', 'Property Age', 'Listed_Facility_Count',
              'Floor_Range_Ordinal']

scaler = StandardScaler()
X_train[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])
X_test[SCALE_COLS] = scaler.transform(X_test[SCALE_COLS])

print(f"Columns scaled ({len(SCALE_COLS)}): {SCALE_COLS}")
print(f"\nX_train[SCALE_COLS] post-scaling summary (mean should be ~0, std ~1):")
print(X_train[SCALE_COLS].describe().loc[['mean', 'std']])
print(f"\nX_test[SCALE_COLS] post-scaling summary (mean/std need NOT be exactly 0/1 -")
print(f"it's transformed with X_train's scaler, not its own):")
print(X_test[SCALE_COLS].describe().loc[['mean', 'std']])

"""
Saved under _scaled filenames, not overwriting the imputation step's
X_train.csv/X_test.csv - tree-based models (Decision Tree/Random Forest/
Gradient Boosting) don't need scaling and can use that unscaled-but-imputed
version directly, without having to invert this transform to get back the
original values.
"""
X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train_scaled.csv"), index=False)
X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test_scaled.csv"), index=False)
joblib.dump((X_train, X_test, y_train, y_test), os.path.join(PROCESSED_DIR, "train_test_split_scaled.pkl"))
joblib.dump(scaler, os.path.join(PROCESSED_DIR, "scaler.pkl"))
print(f"\nSaved scaled X_train_scaled/X_test_scaled (unscaled X_train.csv/X_test.csv left untouched) to {PROCESSED_DIR}")

# ============================================================
# 3.12 Final Dataset Structure Summary
# ============================================================
print("\n" + "="*60)
print("STEP 3.12: FINAL DATASET STRUCTURE SUMMARY")
print("="*60)

"""
Feature engineering (3.8), encoding (3.9) and selection (3.7) now run on
X_train/X_test after the split, not on `df` beforehand, so `df` only
reflects the cleaned dataset through 3.6 (pre-engineering, still including
'price' and every raw column). The "final" feature set this summary
describes is therefore read from X_train's columns, not df's - and
cols_no_engineering / cols_replaced_by_engineering (built in 3.7 above) are
reused rather than retyped, so this can't silently drift out of sync with
what the pipeline actually dropped.
"""
dropped_cols = cols_no_engineering + cols_replaced_by_engineering
retained_raw_cols = [c for c in RAW_COLUMNS if c in X_train.columns]
engineered_cols = [c for c in X_train.columns if c not in RAW_COLUMNS]

print(f"\nCleaned dataset entering the split (3.1-3.6, pre-engineering): {df.shape[0]} rows x {df.shape[1]} columns")
print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

print(f"\n--- Raw features retained as-is ({len(retained_raw_cols)}) ---")
print(retained_raw_cols)

print(f"\n--- Raw features dropped in 3.7 ({len(dropped_cols)}) ---")
print(dropped_cols)

print(f"\n--- Engineered/encoded features created (3.8+3.9) ({len(engineered_cols)}) ---")
print(engineered_cols)

print(f"\n--- Missing value confirmation ---")
print(f"Remaining NaN in X_train: {X_train.isna().sum().sum()}")
print(f"Remaining NaN in X_test:  {X_test.isna().sum().sum()}")

print("\n--- Summary table ---")

"""
"Row-identical groups split across train/test" - an earlier version of this
pipeline used a GroupShuffleSplit specifically to force this count to 0.
3.11 now uses a plain train_test_split for simplicity instead, so this
number is expected to be non-zero: it counts feature-identical rows
(different real listings that happen to agree on every retained column,
once identifying columns are dropped in 3.7) that landed on opposite sides
of the split. This is documented as an accepted limitation of the
simplified split, not a bug - see 3.11's docstring above.
"""
_train_tagged = X_train.assign(price=y_train.values, _src='train')
_test_tagged = X_test.assign(price=y_test.values, _src='test')
_combined = pd.concat([_train_tagged, _test_tagged], ignore_index=True)
_feature_cols = [c for c in _combined.columns if c != '_src']
_cross_dup = _combined[_combined.duplicated(subset=_feature_cols, keep=False)]
_sources_per_group = _cross_dup.groupby(_feature_cols, dropna=False)['_src'].apply(set)
cross_split_leaked_groups = int((_sources_per_group.apply(len) > 1).sum())

state_onehot_added = len(state_dummies_train.columns)
property_type_onehot_added = len(property_type_dummies_train.columns)
missing_indicator_flags_added = len(IMPUTE_FLAG_COLS)

summary_table = pd.DataFrame({
    "Item": [
        "Cleaned rows entering split (pre-engineering, 3.1-3.6)",
        "Raw features retained as-is",
        "Raw features dropped (Section 3.7)",
        "Engineered/encoded features created (3.8+3.9)",
        "State one-hot columns added (3.9, fit on X_train)",
        "Property Type one-hot columns added (3.9, fit on X_train)",
        "Missing-value indicator flags added",
        "Final number of features in X_train/X_test",
        "Training set shape",
        "Testing set shape",
        "Remaining missing values (X_train + X_test)",
        "Row-identical groups split across train/test (known limitation, see 3.11)",
    ],
    "Result": [
        df.shape[0],
        len(retained_raw_cols),
        len(dropped_cols),
        len(engineered_cols),
        state_onehot_added,
        property_type_onehot_added,
        missing_indicator_flags_added,
        X_train.shape[1],
        str(X_train.shape),
        str(X_test.shape),
        int(X_train.isna().sum().sum() + X_test.isna().sum().sum()),
        cross_split_leaked_groups,
    ]
})
print(summary_table.to_string(index=False))
