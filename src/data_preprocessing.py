import numpy as np
import pandas as pd
import os
import re
import joblib

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from scipy.stats import chi2

pd.set_option('display.max_columns', None)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, "data", "raw", "houses.csv")
df = pd.read_csv(csv_path)
RAW_COLUMNS = list(df.columns)  # snapshot for 3.12's summary - kept vs dropped vs engineered

# CLEANING_FIGURES_DIR: diagnostic plots from the cleaning pipeline.
# EDA_DIR: Section 4's own output (train_for_eda.csv + its figures).
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
EXTRA_DIR = os.path.join(BASE_DIR, "data", "extra")
MODELLING_DIR = os.path.join(BASE_DIR, "data", "modelling")
EDA_DIR = os.path.join(BASE_DIR, "data", "eda")
CLEANING_FIGURES_DIR = os.path.join(BASE_DIR, "data", "cleaning_figures")
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(EXTRA_DIR, exist_ok=True)
os.makedirs(MODELLING_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)
os.makedirs(CLEANING_FIGURES_DIR, exist_ok=True)

# ============================================================
# Dataset Review
# ============================================================
# Observe-only pass over the raw dataset: shape, dtypes, sample rows,
# uniqueness, and exact duplicate count. No changes made here.
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
# Detects non-standard missing-value placeholders (e.g. '-') and converts
# them all to a single NaN representation.
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
plt.savefig(os.path.join(CLEANING_FIGURES_DIR, "fig01_missing_bar.png"), dpi=150, bbox_inches="tight")

# ============================================================
# 3.2.1 Duplicate Removal
# ============================================================
"""
Removes exact duplicate rows, then resolves duplicated Ad List groups by
keeping the most complete record per group. Also reports (without
removing) near-duplicate re-listings that share every field except Ad
List/description.
"""
print("\n" + "="*60)
print("STEP 3.2.1: DUPLICATE HANDLING")
print("="*60)

shape_before = df.shape[0]

# Stage 1 - exact duplicates
exact_dup_count = df.duplicated().sum()
df = df.drop_duplicates()
shape_after_exact = df.shape[0]

# Stage 2 - duplicated Ad List: a repeated ID means the same ad was captured twice.
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
comparison_df.to_csv(os.path.join(EXTRA_DIR, "adlist_merge_before_after.csv"), index=False)
print(f"\nBefore/after merge comparison saved to {EXTRA_DIR}\\adlist_merge_before_after.csv")

# Stage 3 - near-duplicate re-listings: same content, different Ad List.
print("\n--- Supplementary check: re-listings ignoring Ad List / description ---")
ignore_cols = ['Ad List', 'description']
check_cols = [c for c in df.columns if c not in ignore_cols]
near_dup_count = df.duplicated(subset=check_cols).sum()
print(f"Rows identical except Ad List/description: {near_dup_count}")
print("Interpretation: different Ad List values confirm these are independent")
print("listing events, not duplicate scrapes of the same ad. Likely represents")
print("re-listing of the same property over time. Retained (not removed) as")
print("there is no evidence this is a data collection error.")

near_dup_mask = df.duplicated(subset=check_cols, keep=False)
near_dup_df = df.loc[near_dup_mask].copy()
group_ids = near_dup_df.groupby(check_cols, dropna=False, sort=False).ngroup()
near_dup_df.insert(0, 'group_id', group_ids)

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
"""
Removes rows with Property Type == 'Others' - not a real property
category, out of scope for this project.
"""
print("\n" + "="*60)
print("STEP 3.2.2: UNRELATED ROW REMOVAL")
print("="*60)

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
# 3.2.3 Rare Property Type Removal
# ============================================================
"""
Removes Duplex/Studio/Townhouse Condo rows - each has too few listings
for a stable one-hot category once encoded.
"""
print("\n" + "="*60)
print("STEP 3.2.3: RARE PROPERTY TYPE REMOVAL")
print("="*60)

RARE_PROPERTY_TYPES_TO_REMOVE = ['Duplex', 'Studio', 'Townhouse Condo']
shape_before_rare_removal = df.shape[0]
rare_type_count = df['Property Type'].isin(RARE_PROPERTY_TYPES_TO_REMOVE).sum()

df = df[~df['Property Type'].isin(RARE_PROPERTY_TYPES_TO_REMOVE)]
shape_after_rare_removal = df.shape[0]

print(f"\nRows with Property Type in {RARE_PROPERTY_TYPES_TO_REMOVE}: {rare_type_count}")
print(f"Rows before removal: {shape_before_rare_removal}")
print(f"Rows after removing rare Property Types: {shape_after_rare_removal}")

df.to_csv(os.path.join(PROCESSED_DIR, "houses_cleaned.csv"), index=False)
joblib.dump(df, os.path.join(PROCESSED_DIR, "houses_cleaned.pkl"))
print(f"\nSaved to {PROCESSED_DIR}: {df.shape}")

# ============================================================
# 3.3 Data Type Conversion
# ============================================================
"""
Converts price/Property Size (text with units/currency symbols) and the
remaining numeric-looking columns into proper numeric dtypes.
"""
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

# Extract the number preceding "sq.ft."
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
"""
Corrects/flags physically implausible values: Property Size cross-checked
against the listing description, a scraping artefact removed from
Facilities, and unrealistic # of Floors values set to NaN.
"""
print("\n" + "="*60)
print("STEP 3.4: INVALID VALUE CORRECTION")
print("="*60)

df['Property Size'] = df['Property Size'].astype(float)  # a correction may add a decimal or NaN

print("\n--- Property Size: bottom 1% before correction ---")
size_p1_threshold = df['Property Size'].quantile(0.01)
print(f"1st percentile threshold: {size_p1_threshold:.1f} sq.ft.")
print(df.loc[df['Property Size'] <= size_p1_threshold, ['Ad List', 'Property Size']].to_string(index=False))

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

FACILITIES_FIX_ADLIST = 95706905
old_facilities = df.loc[df['Ad List'] == FACILITIES_FIX_ADLIST, 'Facilities'].iloc[0]
new_facilities = re.sub(r',\s*10\s*$', '', old_facilities)
df.loc[df['Ad List'] == FACILITIES_FIX_ADLIST, 'Facilities'] = new_facilities
print(f"Ad List {FACILITIES_FIX_ADLIST}")
print(f"  Before: {old_facilities}")
print(f"  After:  {new_facilities}")
print("\n" + "-"*60)

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
"""
No direct action taken here now (the one case this section used to
handle no longer occurs after 3.2.2 runs first). Remaining missing
values are deferred to 3.11.1's fit-on-X_train imputation.
"""
print("\n" + "="*60)
print("STEP 3.5: MISSING-VALUE HANDLING")
print("="*60)

df.to_csv(os.path.join(PROCESSED_DIR, "houses_cleaned.csv"), index=False)
joblib.dump(df, os.path.join(PROCESSED_DIR, "houses_cleaned.pkl"))
print(f"\nSaved to {PROCESSED_DIR}: {df.shape}")

# ============================================================
# 3.6 Train-Test Split
# ============================================================
"""
Splits into X_train/X_test/y_train/y_test before any fit-dependent step
(outlier stats, feature engineering, encoding, selection, imputation),
so every such step downstream can be fit on X_train only.
"""
print("\n" + "="*60)
print("STEP 3.6: TRAIN-TEST SPLIT")
print("="*60)

print("\n--- Split ---")

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

# Known limitation: a plain random split (not group-aware) can leave a
# small number of feature-identical rows split across train/test once
# identifier columns are dropped in 3.10 - accepted as a simplicity
# trade-off, measured and reported in 3.12's summary table.

# ============================================================
# 3.7 Outlier Treatment
# ============================================================
"""
Detects outliers in Property Size/Bedroom/Bathroom/Parking Lot (boxplot +
Z-score) and in Parking Lot vs price (Mahalanobis distance), all fit on
X_train only. Detection/visualisation only; confirmed corrections below
are applied solely where the listing description provides evidence.
"""
print("\n" + "="*60)
print("STEP 3.7: OUTLIER TREATMENT")
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
    plt.savefig(os.path.join(CLEANING_FIGURES_DIR, filename), dpi=150, bbox_inches="tight")

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
    pair = pair_df[[xcol, ycol]].dropna()
    mean_vec = pair.mean().values
    cov = np.cov(pair.values, rowvar=False)
    inv_cov = np.linalg.inv(cov)
    diff = pair.values - mean_vec
    mahal_sq = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
    threshold = chi2.ppf(0.975, df=2)
    outlier = mahal_sq > threshold

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
    plt.savefig(os.path.join(CLEANING_FIGURES_DIR, filename), dpi=150, bbox_inches="tight")

    print(f"Mahalanobis threshold (chi2, df=2, 97.5%): {threshold:.2f}")
    print(f"Outliers flagged: {outlier.sum()} / {len(pair)}")
    return pair.index[outlier]

print("\n--- Parking Lot vs price: Mahalanobis Distance ---")
price_train_raw = np.exp(y_train)  # y_train is log(price); back-transformed for interpretability
pl_pair = pd.DataFrame({'Parking Lot': X_train['Parking Lot'], 'price': price_train_raw})
pl_outlier_idx = mahalanobis_figure(
    pl_pair, 'Parking Lot', 'price', "Parking Lot vs price", "fig10_parking_price_mahalanobis.png"
)
print("Outlier Ad List (for manual review, no correction applied yet):")
pl_review = X_train.loc[pl_outlier_idx, ['Ad List', 'Parking Lot']].copy()
pl_review['price'] = price_train_raw.loc[pl_outlier_idx]
print(pl_review.to_string(index=False))

print("\n--- Confirmed corrections (manually reviewed against description) ---")

# Property Size: every row the Z-score screen flags is re-checked against
# its description (Land Area preferred over Built Up; digit-shift
# artefacts corrected; no evidence -> NaN; otherwise left unchanged).
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

train_size_mean = X_train['Property Size'].dropna().mean()
train_size_std = X_train['Property Size'].dropna().std()

for split_name, frame in [('X_train', X_train), ('X_test', X_test)]:
    s = frame['Property Size'].dropna()
    z = (s - train_size_mean) / train_size_std
    flagged_idx = z.index[z.abs() > Z_THRESHOLD]
    if len(flagged_idx) == 0:
        print(f"{split_name}: no Property Size Z-score outliers - rule has no rows to apply to")
        continue
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

# Bedroom/Bathroom: correction scoped to >= 8 only (not the full Z-score
# list), to avoid false positives from ambiguous listing phrasing; each
# flagged row is checked against its own description.
BEDROOM_DESC_PATTERN = re.compile(r'(\d+)[^\S\n]*-?[^\S\n]*bed[^\S\n]*-?[^\S\n]*rooms?\b', re.IGNORECASE)
BATHROOM_DESC_PATTERN = re.compile(r'(\d+)[^\S\n]*-?[^\S\n]*bath[^\S\n]*-?[^\S\n]*rooms?\b', re.IGNORECASE)

def extract_room_count_from_description(text, pattern):
    if pd.isna(text):
        return np.nan
    m = pattern.search(str(text))
    return float(m.group(1)) if m else np.nan

for col, pattern in [('Bedroom', BEDROOM_DESC_PATTERN), ('Bathroom', BATHROOM_DESC_PATTERN)]:
    print(f"\n--- {col}: description-based rule applied to {col} >= 8 ---")
    for split_name, frame in [('X_train', X_train), ('X_test', X_test)]:
        flagged_idx = frame.index[frame[col] >= 8]
        if len(flagged_idx) == 0:
            print(f"{split_name}: no {col} >= 8 rows - rule has no rows to apply to")
            continue
        corrected_count = 0
        for idx in flagged_idx:
            original = frame.loc[idx, col]
            desc_val = extract_room_count_from_description(frame.loc[idx, 'description'], pattern)
            adlist = int(frame.loc[idx, 'Ad List'])
            if pd.isna(desc_val):
                print(f"Ad List {adlist} in {split_name}: no {col.lower()} count found in description, left unchanged ({original})")
            elif desc_val != original:
                print(f"Ad List {adlist} in {split_name}: {col} {original} -> {desc_val:.0f} (description states '{desc_val:.0f}-{col}s')")
                frame.loc[idx, col] = desc_val
                corrected_count += 1
            else:
                print(f"Ad List {adlist} in {split_name}: description confirms original value ({original}), no change")
        print(f"{split_name}: {corrected_count}/{len(flagged_idx)} {col} >= 8 rows corrected via description match")

# Parking Lot: domain-knowledge filter on top of the Mahalanobis candidates
# - >=4 lots paired with price far below the X_train median has no
# plausible explanation and no independent source to verify, so it's set
# to NaN for later imputation.
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
# 3.8 Feature Engineering
# ============================================================
"""
Derives State (from Address), Property Age (from Completion Year),
Listed_Facility_Count and Has_X amenity flags (from Facilities/nearby
columns). All fixed transformations, applied to X_train/X_test independently.
"""
print("\n" + "="*60)
print("STEP 3.8: FEATURE ENGINEERING")
print("="*60)

print("\n--- 3.8.1 State (extracted from Address) ---")

# Scans every comma-separated Address segment (not just the last) against a
# fixed list of Malaysian states, case-insensitive.
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

print("\n--- 3.8.2 Property Age (from Completion Year) ---")

# Property Age = fixed reference year (dataset collection year) minus
# Completion Year; a negative result (off-plan) is left as NaN.
REFERENCE_YEAR = 2023

def compute_property_age(completion_year):
    if pd.isna(completion_year):
        return np.nan
    age = REFERENCE_YEAR - completion_year
    return age if age >= 0 else np.nan

print("\n--- 3.8.3 Listed_Facility_Count (from Facilities) ---")

# Counts valid facility items in the Facilities text, dropping purely
# numeric (non-facility) entries.
def count_facilities(facilities_text):
    if pd.isna(facilities_text):
        return 0
    items = [item.strip() for item in facilities_text.split(',')]
    return len([item for item in items if item and not item.isdigit()])

print("\n--- 3.8.4 Has_X flags (from 'nearby amenity' text columns) ---")

# Converts each nearby-amenity text column into a presence flag (1 =
# information recorded, 0 = not recorded - not a confirmed absence).
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

# State/Property Type's rare-category merge and one-hot encoding are fit
# on X_train only - see 3.9 below, not here.

# Features considered and dropped (not in X): Is_Off_Plan (not significant
# against price), Facilities_Recorded (redundant with
# Listed_Facility_Count == 0), Nearby_Amenity_Count (linear combination of
# existing Has_X flags, not significant), Units per Floor (produces
# physically impossible values for some rows).

print(f"\nX_train shape after 3.8: {X_train.shape} | X_test shape after 3.8: {X_test.shape}")

# ============================================================
# 3.9 Categorical Encoding
# ============================================================
"""
Encodes Land Title, Tenure Type, Floor Range, Facilities (fixed rules,
applied to X_train/X_test independently), then State and Property Type
(rare-category merge + one-hot, fit on X_train only).
"""
print("\n" + "="*60)
print("STEP 3.9: CATEGORICAL ENCODING")
print("="*60)

print("\n--- 3.9.1 Land Title: rare-category merge, then binary encoding ---")

# Merges the rare 'Malay Reserved' category into 'Bumi Lot' (same
# purchase-restriction status), then encodes the two remaining categories
# as a single binary flag.
for _X, _label in [(X_train, "X_train"), (X_test, "X_test")]:
    _X['Land Title'] = _X['Land Title'].replace('Malay Reserved', 'Bumi Lot')
    _X['Is_Non_Bumi_Lot'] = _X['Land Title'].map({'Non Bumi Lot': 1, 'Bumi Lot': 0})
    print(f"[{_label}] 'Is_Non_Bumi_Lot' value counts:\n{_X['Is_Non_Bumi_Lot'].value_counts(dropna=False)}")
print("\n" + "-"*60)

print("\n--- 3.9.2 Tenure Type: binary encoding ---")

# Encodes Tenure Type (2 categories, no missing values) as a single binary
# flag instead of one-hot.
for _X, _label in [(X_train, "X_train"), (X_test, "X_test")]:
    _X['Freehold Indicator'] = _X['Tenure Type'].map({'Freehold': 1, 'Leasehold': 0})
    print(f"[{_label}] 'Freehold Indicator' value counts:\n{_X['Freehold Indicator'].value_counts(dropna=False)}")
print("\n" + "-"*60)

print("\n--- 3.9.3 Floor Range: ordinal encoding ---")

# Encodes Floor Range as an ordinal (Low/Medium/High have a real order);
# Unknown maps to NaN, imputed later in 3.11.1.
for _X, _label in [(X_train, "X_train"), (X_test, "X_test")]:
    _X['Floor_Range_Ordinal'] = _X['Floor Range'].map({'Low': 1, 'Medium': 2, 'High': 3})
    print(f"[{_label}] 'Floor_Range_Ordinal' value counts (NaN=Unknown, imputed later):\n"
          f"{_X['Floor_Range_Ordinal'].value_counts(dropna=False)}")
print("\n" + "-"*60)

print("\n--- 3.9.4 Facilities: text cleanup + multi-hot encoding (vocabulary fit on X_train only) ---")

# Cleans the Facilities text (split/strip/title-case, drop numeric junk),
# then multi-hot encodes it; the vocabulary is fit on X_train only, and any
# facility name unseen in X_train is dropped (not encoded) in X_test.
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

# ============================================================
# EDA snapshot (train-only, post fixed-rule encoding, pre one-hot)
# ============================================================
"""
Saves the Section 4 EDA input: X_train only (no test-set leakage), with
every 3.8/3.9 fixed-rule feature already built, before State/Property
Type's one-hot encoding runs.
"""
print("\n--- EDA snapshot (train-only, engineered, pre one-hot) ---")
eda_cols_to_drop = [
    'description', 'Ad List',
    'Nearby School', 'Nearby Mall', 'Nearby Railway Station',
    'Category', 'Firm Type', 'Firm Number', 'REN Number',
    'Building Name', 'Developer',
    'Address', 'Completion Year',
    'Bus Stop', 'Mall', 'Park', 'School', 'Hospital', 'Highway', 'Railway Station',
    'Facilities',
]
eda_train = X_train.drop(columns=eda_cols_to_drop).copy()
# price is always a whole RM amount in the source data - round() clears the
# float noise np.exp(log(x)) round-tripping introduces.
eda_train['price'] = np.exp(y_train).round().astype(int)
eda_train.to_csv(os.path.join(EDA_DIR, "train_for_eda.csv"), index=False)
print(f"train_for_eda.csv saved: {eda_train.shape}")

print("\n--- 3.9.5 State: rare-category merge + one-hot encoding (fit on X_train only) ---")

# State has no inherent order, so one-hot is used (unlike Floor Range's
# ordinal encoding). NaN becomes an explicit 'Unknown' category; the rare-
# category threshold and category list are fit on X_train only and applied
# to X_test.
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

print("\n--- 3.9.6 Property Type: one-hot encoding (categories fit on X_train only) ---")

# No rare-category merge needed here - 3.2.3 already removed the rare
# types, so every remaining Property Type has hundreds+ listings. Category
# list is fit on X_train only and applied to X_test.
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

print(f"\n'Property Type' counts in X_train:\n{X_train['Property Type'].value_counts()}")
print(f"\nOne-hot columns created ({len(property_type_dummies_train.columns)}): {list(property_type_dummies_train.columns)}")
print("\n" + "-"*60)

print(f"\nX_train shape after 3.9: {X_train.shape} | X_test shape after 3.9: {X_test.shape}")

# ============================================================
# 3.10 Feature Selection
# ============================================================
"""
Drops raw columns with no predictive value on their own merits (3.10.1),
and raw columns already superseded by a 3.8/3.9 engineered feature (3.10.2).
"""
print("\n" + "="*60)
print("STEP 3.10: FEATURE SELECTION")
print("="*60)

print("\n--- 3.10.1 Stage A: columns with no predictive value, unrelated to any 3.8/3.9 engineering ---")

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

print("\n--- 3.10.2 Stage B: raw columns already superseded by a 3.8/3.9 engineered feature ---")

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

print(f"\nX_train shape before 3.10 drop: {shape_before_drop_train} -> after: {X_train.shape}")
print(f"X_test shape before 3.10 drop:  {shape_before_drop_test} -> after: {X_test.shape}")
print(f"Columns dropped: {len(cols_no_engineering) + len(cols_replaced_by_engineering)}")
print("\n" + "-"*60)

# ============================================================
# 3.11.1 Missing-value Imputation
# ============================================================
"""
Median-imputes the columns still carrying real NaN, fit on X_train only
and applied to X_test. Property Age/# of Floors/Total Units/Parking Lot
also get a missing-indicator flag (missingness itself is significant
against price for these); Property Size and Floor_Range_Ordinal are
median-filled only, no flag.
"""
print("\n" + "="*60)
print("STEP 3.11.1: MISSING-VALUE IMPUTATION")
print("="*60)

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

X_train.to_csv(os.path.join(MODELLING_DIR, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(MODELLING_DIR, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(MODELLING_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(MODELLING_DIR, "y_test.csv"), index=False)
joblib.dump((X_train, X_test, y_train, y_test), os.path.join(MODELLING_DIR, "train_test_split.pkl"))
print(f"\nSaved X_train/X_test/y_train/y_test to {MODELLING_DIR}")

# ============================================================
# 3.11.2 Feature Scaling
# ============================================================
"""
Log-transforms (log1p) the continuous/count numeric columns, applied
identically to X_train and X_test (log1p has no fitted parameters, so
no train-only fit step is needed). Saved separately as the "_scaled"
files; tree-based models can use the unscaled X_train.csv/X_test.csv instead.
"""
print("\n" + "="*60)
print("STEP 3.11.2: FEATURE SCALING")
print("="*60)

print("\n--- Log transform (log1p), applied identically to X_train and X_test ---")

SCALE_COLS = ['Bedroom', 'Bathroom', 'Property Size', '# of Floors',
              'Total Units', 'Parking Lot', 'Property Age', 'Listed_Facility_Count',
              'Floor_Range_Ordinal']

X_train[SCALE_COLS] = np.log1p(X_train[SCALE_COLS])
X_test[SCALE_COLS] = np.log1p(X_test[SCALE_COLS])

print(f"Columns log-transformed ({len(SCALE_COLS)}): {SCALE_COLS}")
print(f"\nX_train[SCALE_COLS] post-transform summary:")
print(X_train[SCALE_COLS].describe().loc[['mean', 'std']])
print(f"\nX_test[SCALE_COLS] post-transform summary:")
print(X_test[SCALE_COLS].describe().loc[['mean', 'std']])

X_train.to_csv(os.path.join(MODELLING_DIR, "X_train_scaled.csv"), index=False)
X_test.to_csv(os.path.join(MODELLING_DIR, "X_test_scaled.csv"), index=False)
joblib.dump((X_train, X_test, y_train, y_test), os.path.join(MODELLING_DIR, "train_test_split_scaled.pkl"))
print(f"\nSaved log-transformed X_train_scaled/X_test_scaled (untransformed X_train.csv/X_test.csv left untouched) to {MODELLING_DIR}")

# ============================================================
# 3.12 Final Dataset Structure Summary
# ============================================================
"""
Reports the final feature set (retained raw / dropped / engineered),
read from X_train's columns since 3.8-3.10 operate on X_train/X_test,
not on `df`.
"""
print("\n" + "="*60)
print("STEP 3.12: FINAL DATASET STRUCTURE SUMMARY")
print("="*60)

dropped_cols = cols_no_engineering + cols_replaced_by_engineering
retained_raw_cols = [c for c in RAW_COLUMNS if c in X_train.columns]
engineered_cols = [c for c in X_train.columns if c not in RAW_COLUMNS]

print(f"\nCleaned dataset entering the split (3.1-3.5, pre-engineering): {df.shape[0]} rows x {df.shape[1]} columns")
print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

print(f"\n--- Raw features retained as-is ({len(retained_raw_cols)}) ---")
print(retained_raw_cols)

print(f"\n--- Raw features dropped in 3.10 ({len(dropped_cols)}) ---")
print(dropped_cols)

print(f"\n--- Engineered/encoded features created (3.8+3.9) ({len(engineered_cols)}) ---")
print(engineered_cols)

print(f"\n--- Missing value confirmation ---")
print(f"Remaining NaN in X_train: {X_train.isna().sum().sum()}")
print(f"Remaining NaN in X_test:  {X_test.isna().sum().sum()}")

print("\n--- Summary table ---")

# "Row-identical groups split across train/test": counts feature-identical
# rows (different real listings agreeing on every retained column, once
# identifiers are dropped in 3.10) split across train/test - an accepted
# limitation of 3.6's plain random split, not a bug.
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
        "Cleaned rows entering split (pre-engineering, 3.1-3.5)",
        "Raw features retained as-is",
        "Raw features dropped (Section 3.10)",
        "Engineered/encoded features created (3.8+3.9)",
        "State one-hot columns added (3.9.5, fit on X_train)",
        "Property Type one-hot columns added (3.9.6, fit on X_train)",
        "Missing-value indicator flags added",
        "Final number of features in X_train/X_test",
        "Training set shape",
        "Testing set shape",
        "Remaining missing values (X_train + X_test)",
        "Row-identical groups split across train/test (known limitation, see 3.6)",
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

print(df['Property Type'].value_counts())
