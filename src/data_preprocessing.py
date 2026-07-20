import numpy as np
import pandas as pd
import os
import re
import joblib

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
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
print("STEP 3.2: DUPLICATE HANDLING")
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

"""
Property Size is cross-checked against the 'sq.ft' figure independently
stated in the description. Two rules: (a) below a physically possible size
is invalid; (b) ~10x/100x/1000x the description value is a digit-shift
extraction error. Values matching neither (e.g. 9,376/9,800 sq.ft.) are
deferred to Section 3.6 as genuine extreme values.
"""
df['Property Size'] = df['Property Size'].astype(float)  # a correction may add a decimal or NaN

SIZE_PATTERN = re.compile(
    r'(\d[\d,]*\.?\d*)\s*(?:sq\.?\s*ft\.?|sqft|square\s*feet|sf)(?!\w)',
    re.IGNORECASE
)

def extract_size_from_description(text):
    if pd.isna(text):
        return np.nan
    match = SIZE_PATTERN.search(str(text))
    return float(match.group(1).replace(',', '')) if match else np.nan

desc_size = df['description'].apply(extract_size_from_description)

IMPOSSIBLE_MIN_SIZE = 100  # smallest genuine studio/SoHo units run ~200+ sq.ft.
too_small_mask = df['Property Size'] < IMPOSSIBLE_MIN_SIZE


"""
Ratio check only applies above 8,000 sq.ft.: without this guard, a genuine
1,213 sq.ft. unit (Ad List 103788229) would get "corrected" by an unrelated
typo in its own description ("1.213 sqft"), since the ratio also lands on ~1000x.
"""
LARGE_SUSPECT_MIN = 8000
large_suspect_mask = df['Property Size'] > LARGE_SUSPECT_MIN

ratio = df['Property Size'] / desc_size
digit_shift_mask = large_suspect_mask & desc_size.notna() & ratio.apply(
    lambda r: pd.notna(r) and r > 0 and any(abs(r / p - 1) < 0.05 for p in [10, 100, 1000])
)

size_flag_mask = too_small_mask | digit_shift_mask
print(f"Property Size flagged below screening threshold (< {IMPOSSIBLE_MIN_SIZE} sq.ft.): {too_small_mask.sum()}")
print(f"Property Size flagged by digit-shift artefact (~10x/100x/1000x description):  {digit_shift_mask.sum()}")

size_log = []
for idx in df.index[size_flag_mask]:
    old_val = df.loc[idx, 'Property Size']
    dval = desc_size.loc[idx]
    if pd.notna(dval):
        df.loc[idx, 'Property Size'] = dval
        final_val = round(dval)
    else:
        df.loc[idx, 'Property Size'] = np.nan
        final_val = np.nan
    size_log.append({
        "Ad List": df.loc[idx, 'Ad List'],
        "Original Size": old_val,
        "Description Size": dval,
        "Final Size": final_val,
    })

size_log_df = pd.DataFrame(size_log)
print("\n--- Property Size Correction Log ---")
print(size_log_df.to_string(index=False))

large_deferred = df.loc[(df['Property Size'] > 8000) & (~size_flag_mask), ['Ad List', 'Property Size']]

# Property Size is integer sq.ft. in this dataset; round the one decimal the
# digit-shift correction introduced, then cast back to int64 (still 0-missing).
df['Property Size'] = df['Property Size'].round().astype(int)

# Blanket regex cross-check against description was tested but rejected: 878
# mismatches dataset-wide, mostly false positives (see notes/bedroom_bathroom_
# regex_false_positives.csv). Instead, only singleton values (value_counts()==1)
# were individually verified against description.
room_corrections = [
    {"Ad List": 102236931, "Column": "Bedroom", "Original": 10, "Description evidence": 3,
     "Note": "Description states '3-Bedrooms' (and '3-Bathroom', matching the existing Bathroom value)."},
    {"Ad List": 103207012, "Column": "Bathroom", "Original": 8, "Description evidence": 2,
     "Note": "Description states '2 Bathrooms' in both unit configurations listed; 8 does not appear."},
    {"Ad List": 96822478, "Column": "Bedroom", "Original": 8, "Description evidence": 8,
     "Note": "Description confirms '8 Bedrooms'; retained as genuine, not a data error."},
]
for corr in room_corrections:
    df.loc[df['Ad List'] == corr["Ad List"], corr["Column"]] = corr["Description evidence"]
room_log_df = pd.DataFrame(room_corrections).rename(columns={
    "Column": "Variable",
    "Original": "Original Value",
    "Description evidence": "Description Value",
})
room_log_df["Final Value"] = room_log_df["Description Value"]
print("\n--- Bedroom / Bathroom Correction Log ---")
print(room_log_df[["Ad List", "Variable", "Original Value", "Description Value", "Final Value"]].to_string(index=False))

print(f"\nRemaining large Property Size values deferred to Section 3.6 (not modified in this stage): {len(large_deferred)}")
print(large_deferred.to_string(index=False))

# Total Units == 1 doesn't match genuine multi-unit developments (confirmed via
# description); treated as a scraping placeholder rather than a real count.
total_units_sentinel_mask = df['Total Units'] == 1
print(f"\n'Total Units' == 1 (suspected placeholder, not a genuine value): {total_units_sentinel_mask.sum()}")
df.loc[total_units_sentinel_mask, 'Total Units'] = np.nan

# No independent field verifies floor count; values above Malaysia's tallest
# building (Merdeka 118, 118 storeys) are physically impossible and flagged.
FLOOR_MAX = 118
floor_invalid_mask = df['# of Floors'] > FLOOR_MAX
print(f"\n'# of Floors' > {FLOOR_MAX} (unlikely): {floor_invalid_mask.sum()}")
print(df.loc[floor_invalid_mask, ['Ad List', '# of Floors']].to_string(index=False))
df.loc[floor_invalid_mask, '# of Floors'] = np.nan

# Parking Lot values of 9-10 paired with low-cost listings look suspicious, but
# unlike Property Size there's no independent field to confirm they're wrong,
# so they're left unchanged here and revisited in Section 3.6.
parking_suspect_count = (df['Parking Lot'] >= 9).sum()
print(f"\n'Parking Lot' >= 9 (context looks suspicious, no evidence to confirm error): {parking_suspect_count} - left unchanged, revisited in 3.6")

print(f"\nProperty Size range after 3.4: {df['Property Size'].min()} - {df['Property Size'].max()}")
print(f"Total Units range after 3.4:   {df['Total Units'].min()} - {df['Total Units'].max()}")
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

# "Unknown" category instead of mode, which would inflate the most common range.
floor_range_missing = df['Floor Range'].isna().sum()
df['Floor Range'] = df['Floor Range'].fillna('Unknown')
print(f"\n'Floor Range' missing values filled with 'Unknown': {floor_range_missing}")

# Completion Year / # of Floors / Total Units / Parking Lot: median + missing-
# indicator imputation is deferred to the 3.11 modelling pipeline (fit on
# X_train only) to avoid test-set leakage. Left as real NaN here for EDA.

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
# 3.6 Outlier Treatment
# ============================================================
print("\n" + "="*60)
print("STEP 3.6: OUTLIER TREATMENT")
print("="*60)

# Price: raw vs log boxplot - log compresses the right-skewed scale so the
# bulk of listings aren't dominated by the high tail (outliers stay visible).
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# original price
sns.boxplot(x=df['price'], ax=axes[0]) #right graph, draw box, whisker, outliers
axes[0].set_title("Raw Listing Price")
axes[0].set_xlabel("Listing Price (RM)")
axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}')) # convert 1ef to 1,000,000

#log price
sns.boxplot(x=np.log(df['price']), ax=axes[1])
axes[1].set_title("Log-transformed Price")
axes[1].set_xlabel("ln(Listing Price)")
plt.tight_layout()
plt.savefig(os.path.join(DOWNLOADS_DIR, "fig02_price_boxplot.png"), dpi=150, bbox_inches="tight")
plt.close()

price_q1, price_q3 = df['price'].quantile([0.25, 0.75])
# 没有算lower bound是因为得出来的是negative
price_upper = price_q3 + 1.5 * (price_q3 - price_q1) # print colsole output mention有多少 > upper bound
print(f"Price IQR upper bound: RM{price_upper:,.0f}; values above: {(df['price'] > price_upper).sum()}")
print(f"Price skewness: raw = {df['price'].skew():.2f}, log = {np.log(df['price']).skew():.2f}")

# Bedroom/Bathroom IQR is degenerate (Q1==Q3) - not usable as a rule; their
# extreme values were already corrected in 3.4 using description evidence.
print(f"\nBedroom IQR: Q1={df['Bedroom'].quantile(.25)}, Q3={df['Bedroom'].quantile(.75)} (degenerate, not used as a rule)")
print(f"Bathroom IQR: Q1={df['Bathroom'].quantile(.25)}, Q3={df['Bathroom'].quantile(.75)} (degenerate, not used as a rule)")

# Property Size vs Price: candidates are `large_deferred` from 3.4 (>8000 sq.ft,
# not caught by digit-shift) - reused, not re-picked, so results stay reproducible.
LARGE_SIZE_REVIEW_ADLIST = large_deferred['Ad List'].tolist()
review_check = df.loc[df['Ad List'].isin(LARGE_SIZE_REVIEW_ADLIST),
                       ['Ad List', 'Property Size', 'price']].copy()
review_check['Ad List'] = review_check['Ad List'].astype(int)
review_check['price_per_sqft'] = (review_check['price'] / review_check['Property Size']).round(1)

fig, ax = plt.subplots(figsize=(9, 6)) # Create a 9x6 inch blank canvas and a drawing area (ax)
sns.scatterplot(data=df, x='Property Size', y='price', hue='Property Type', alpha=0.5, ax=ax) # first layer
ax.scatter(review_check['Property Size'], review_check['price'], color='red', s=150,
           marker='X', label='Large size under review', zorder=5) # second layer
for _, r in review_check.iterrows():
    ax.annotate(
        f"Ad List {int(r['Ad List'])}\n{r['Property Size']:,.0f} sq.ft.\nRM{r['price_per_sqft']:,.1f}/sqft",
        xy=(r['Property Size'], r['price']), xytext=(10, 10), textcoords='offset points',
        fontsize=8, color='red'
    )
ax.set_title("Property Size vs Price")
ax.set_ylabel("Listing Price (RM)")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(DOWNLOADS_DIR, "fig03_size_vs_price.png"), dpi=150, bbox_inches="tight")
plt.close()

# Price per sq.ft.: a size-price consistency check - unusually low means size
# is large relative to price. The 3 review records are marked with vlines.
price_per_sqft = df['price'] / df['Property Size']
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(price_per_sqft, bins=50, ax=ax)
# Stagger annotation heights - two values (15.3/26.7) are close and would overlap.
annotation_heights = [0.95, 0.75, 0.55]
for (_, r), height in zip(review_check.sort_values('price_per_sqft').iterrows(), annotation_heights):
    ax.axvline(r['price_per_sqft'], color='red', linestyle='--', linewidth=1)
    ax.annotate(f"Ad List {int(r['Ad List'])} (RM{r['price_per_sqft']:,.1f}/sqft)",
                xy=(r['price_per_sqft'], ax.get_ylim()[1] * height),
                fontsize=8, color='red', ha='left', va='center',
                xytext=(5, 0), textcoords='offset points')
ax.set_title("Distribution of Listing Price per Square Foot")
ax.set_xlabel("Price per sq.ft. (RM/sq.ft.)")
plt.tight_layout()
plt.savefig(os.path.join(DOWNLOADS_DIR, "fig04_price_per_sqft.png"), dpi=150, bbox_inches="tight")
plt.close()

print("\n--- Price per sq.ft. for the 3 large Property Size records under review ---")
print(review_check.to_string(index=False))

# Parking Lot: IQR flags 33 candidates, but most pair with higher-priced
# properties (not suspicious). A genuine candidate needs all three: high
# (>IQR upper), rare (<=2 occurrences), AND price far below the median.
parking_q1, parking_q3 = df['Parking Lot'].quantile([0.25, 0.75])
parking_upper = parking_q3 + 1.5 * (parking_q3 - parking_q1)
print(f"\nParking Lot IQR upper bound: {parking_upper}; candidate values above: {(df['Parking Lot'] > parking_upper).sum()}")

parking_value_counts = df['Parking Lot'].value_counts()
rare_parking_values = parking_value_counts[parking_value_counts <= 2].index
overall_median_price = df['price'].median()
parking_candidate_mask = (
    (df['Parking Lot'] > parking_upper)
    & df['Parking Lot'].isin(rare_parking_values)
    & (df['price'] < overall_median_price * 0.5)
)
PARKING_REVIEW_ADLIST = df.loc[parking_candidate_mask, 'Ad List'].tolist()
# Property Size/Bedroom/Bathroom aren't part of the filter - shown here as
# supporting context (all 3 are compact units, well below the 902 sq.ft. median).
print(df.loc[parking_candidate_mask, ['Ad List', 'Parking Lot', 'price', 'Property Size', 'Bedroom', 'Bathroom']].to_string(index=False))

# Parking Lot vs Price: a count plot alone can't show if a high count is
# plausible - only price cross-check does. The 3 severe candidates are highlighted.
fig, ax = plt.subplots(figsize=(9, 6))
sns.scatterplot(data=df, x='Parking Lot', y='price', alpha=0.4, ax=ax)
parking_review = df.loc[df['Ad List'].isin(PARKING_REVIEW_ADLIST), ['Ad List', 'Parking Lot', 'price', 'Property Size']].copy()
parking_review['Ad List'] = parking_review['Ad List'].astype(int)
ax.scatter(parking_review['Parking Lot'], parking_review['price'], color='red', s=150,
           marker='X', label='High lot count under review', zorder=5)
# The two 10-lot records (78k/88k) are close together - offset labels to avoid overlap.
annotation_offsets = [(10, 10), (10, 65), (-100, 10)]
for (_, r), offset in zip(parking_review.sort_values('price').iterrows(), annotation_offsets):
    ax.annotate(f"Ad List {int(r['Ad List'])}\n{r['Parking Lot']:.0f} lots\nRM{r['price']:,.0f}\n{r['Property Size']:,.0f} sq.ft.",
                xy=(r['Parking Lot'], r['price']), xytext=offset, textcoords='offset points',
                fontsize=8, color='red')
ax.set_title("Parking Lot vs Price")
ax.set_ylabel("Listing Price (RM)")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(DOWNLOADS_DIR, "fig05_parking_vs_price.png"), dpi=150, bbox_inches="tight")
plt.close()

# --- Final treatment (Table 3.6) ---
# Property Size: 103729938 (17,611 sq.ft.) has no source support and is
# inconsistent with price/attributes -> converted to NaN. 96973074 (9,376) and
# 103792765 (9,800) are retained - both have source-confirmed sizes in their
# description, even though 96973074's price/sqft looks unusual.
df.loc[df['Ad List'] == 103729938, 'Property Size'] = np.nan
print("\nProperty Size converted to NaN: Ad List 103729938 (no source support, inconsistent)")
print("Property Size retained: Ad List 96973074, 103792765 (source-confirmed size)")

# Parking Lot: all 3 candidates lack independent evidence and are inconsistent
# with price/property attributes -> converted to NaN.
df.loc[df['Ad List'].isin(PARKING_REVIEW_ADLIST), 'Parking Lot'] = np.nan
print(f"Parking Lot converted to NaN: Ad List {PARKING_REVIEW_ADLIST}")

df.to_csv(os.path.join(PROCESSED_DIR, "houses_cleaned.csv"), index=False)
joblib.dump(df, os.path.join(PROCESSED_DIR, "houses_cleaned.pkl"))
print(f"\nSaved to {PROCESSED_DIR}: {df.shape}")

print(df.info())