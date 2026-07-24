import numpy as np
import pandas as pd
import os
import re
import joblib

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import GroupShuffleSplit

pd.set_option('display.max_columns', None)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, "data", "raw", "houses.csv")
df = pd.read_csv(csv_path)
RAW_COLUMNS = list(df.columns)  # snapshot for 3.12's summary - kept vs dropped vs engineered

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

# ============================================================
# 3.8 Feature Engineering (executed before 3.7 Feature Selection,
# since several dropped columns are the source of engineered features below)
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
equality, not substring search, so a place name that merely contains a state's
name (e.g. "Kampung Melaka") is never mistaken for the state "Melaka".

Matching is case-insensitive (e.g. "SABAH", "putrajaya") - two rows in this
dataset have a wrong-case state segment, currently masked only by a correctly-
cased duplicate elsewhere in the same address; without case-insensitivity a
future row with no such backup would be wrongly marked NaN.
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

df['State'] = df['Address'].apply(extract_state_from_address)

address_missing = df['Address'].isna().sum()
no_state_in_any_segment = df['State'].isna().sum() - address_missing
print(f"Address missing (-> State = NaN):                          {address_missing}")
print(f"Address present, no segment matched a real state (-> NaN): {no_state_in_any_segment}")
print(f"Total 'State' missing after extraction:                    {df['State'].isna().sum()}")

"""
Rare-category merge (states with fewer than 10 listings -> 'Other') and
one-hot encoding are deferred to Part 3 of the pipeline, AFTER the train/test
split - the <10 threshold is a statistic computed from data, so it must be
fit on X_train only and then applied to X_test, the same leakage reasoning
already used for Floor_Range_Ordinal's median fill. 'State' is kept here as
plain extracted text; row_group_ids (built right after 3.7 below) is
deliberately computed on this pre-merge, pre-encoding text value.
"""
print("\n--- 'State' value counts after extraction (rare-category merge + encoding deferred to Part 3, post-split) ---")
print(df['State'].value_counts(dropna=False))
print("\n" + "-"*60)

print("\n--- Property Age (from Completion Year) ---")

"""
Property Age is computed against a fixed reference year, not whatever year
the script happens to run in - Completion Year is a fixed fact about the
property, and this dataset was collected in September 2023 (per the Kaggle
source), so ages must be measured from then, not from today's calendar year.
A dynamic "current year" would silently shift every Property Age value (and
any report numbers built on it) further every year this script is re-run.
"""
REFERENCE_YEAR = 2023

def compute_property_age(completion_year):
    if pd.isna(completion_year):
        return np.nan
    age = REFERENCE_YEAR - completion_year
    return age if age >= 0 else np.nan

df['Property Age'] = df['Completion Year'].apply(compute_property_age)

"""
A negative age (Completion Year after REFERENCE_YEAR) isn't invalid data - it
means the unit was still under construction / sold off-plan at the time of
collection, a normal, common scenario in Malaysian property listings. "Age"
simply doesn't apply yet to an unbuilt property, so Property Age is correctly
left NaN for these rows (Is_Off_Plan, which would have preserved that as its
own signal, was tried and dropped - see the record further below).
"""
completion_missing = df['Completion Year'].isna().sum()
off_plan_count = (df['Completion Year'] > REFERENCE_YEAR).sum()
print(f"Completion Year missing (-> Property Age = NaN):          {completion_missing}")
print(f"Completion Year > {REFERENCE_YEAR} (-> off-plan, Property Age = NaN): {off_plan_count}")
print(f"Total 'Property Age' missing:                             {df['Property Age'].isna().sum()}")
print(f"\n'Property Age' range: {df['Property Age'].min()} - {df['Property Age'].max()}")
print("\n" + "-"*60)

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

df['Listed_Facility_Count'] = df['Facilities'].apply(count_facilities)

facilities_missing = df['Facilities'].isna().sum()
print(f"Facilities missing (Listed_Facility_Count set to 0):    {facilities_missing}")
print(f"\n'Listed_Facility_Count' range: {df['Listed_Facility_Count'].min()} - {df['Listed_Facility_Count'].max()}")
print(df['Listed_Facility_Count'].describe())
print("\n" + "-"*60)

print("\n--- Has_X flags (from 'nearby amenity' text columns) ---")

"""
Bus Stop / Mall / Park / School / Hospital / Highway / Railway Station each
list a specific nearby amenity name as free text (e.g. "Mid Valley Megamall"),
with missing rates between 76% and 96% (see notes_missing_value_decision_audit.md).
The specific name has too little repeat structure to be usable directly, but
whether the field was filled in at all is itself a signal. Converting to a
presence flag also removes the missing-value problem entirely - every row
gets 0 or 1, there is no NaN left over to defer to a later imputation step.
Note this flag means "information was recorded", not "amenity confirmed
absent" - a 0 does not prove there is no mall nearby, only that it wasn't written.
"""
NEARBY_COLUMNS = ['Bus Stop', 'Mall', 'Park', 'School', 'Hospital', 'Highway', 'Railway Station']

has_flag_log = []
for col in NEARBY_COLUMNS:
    flag_col = 'Has_' + col.replace(' ', '_')
    df[flag_col] = df[col].notna().astype(int)
    has_flag_log.append({
        "Source column": col,
        "New flag": flag_col,
        "Missing in source (-> flag=0)": df[col].isna().sum(),
        "Present in source (-> flag=1)": df[col].notna().sum(),
    })

has_flag_df = pd.DataFrame(has_flag_log)
print(has_flag_df.to_string(index=False))
print("\n" + "-"*60)

"""
--- Features tried in 3.8 and dropped (kept here as a record, not in df) ---

Is_Off_Plan: 1 if Completion Year > REFERENCE_YEAR (still under construction
at collection time), 0 if completed, NaN if Completion Year itself missing -
tried so the off-plan signal wouldn't collapse into the same NaN as a
genuinely unknown Completion Year. Dropped: only 22/3793 (0.6%) positive
cases, and r=-0.0004, p=0.986 against price - not significant. The domain
rationale (new-build premium) is still sound; the sample is simply too small
to detect it either way, which is a different and stronger reason to drop
than "correlation happened to be low" alone.

Facilities_Recorded: a presence flag paired with Listed_Facility_Count,
mirroring the Is_Off_Plan idea above. Dropped after verifying
Listed_Facility_Count == 0 for exactly the same 607 rows where Facilities is
missing, with zero exceptions - no row has Facilities filled in yet produces
a count of 0 after cleaning. The two carry identical information in this
dataset, so keeping both is pure redundancy.

Nearby_Amenity_Count: sum of the 7 Has_X flags into a single 0-7 score.
Dropped for two reasons - it's a pure linear combination of columns already
in df (the same redundancy problem as Total_Rooms = Bedroom + Bathroom), and
it's empirically low-value too: r=-0.019, p=0.242 against price (n=3793),
not statistically significant.

Units per Floor: Total Units / # of Floors. Dropped because several rows
produce physically impossible values (e.g. Ad List 100502825, Total
Units=7810 / # of Floors=5 = 1562 units on one floor). Neither source column
was flagged as invalid by their own individual range checks in 3.4/3.6 (each
looks reasonable in isolation), so this cross-column ratio surfaced a data
quality issue in Total Units / # of Floors that univariate outlier checks
can't catch. Left out until that upstream issue is resolved with evidence.
"""

df.to_csv(os.path.join(PROCESSED_DIR, "houses_cleaned.csv"), index=False)
joblib.dump(df, os.path.join(PROCESSED_DIR, "houses_cleaned.pkl"))
print(f"\nSaved to {PROCESSED_DIR}: {df.shape}")

# ============================================================
# 3.9 Categorical Encoding
# ============================================================
print("\n" + "="*60)
print("STEP 3.9: CATEGORICAL ENCODING")
print("="*60)

"""
State and Property Type one-hot encoding (including their rare-category
merges) are deferred to Part 3 of the pipeline, after the train/test split -
both merge thresholds (State <10, Property Type <20 listings) are statistics
computed from data, so per the same leakage rule applied everywhere else in
this pipeline (Floor_Range_Ordinal's median, the 3.11 imputation medians,
3.10's scaler), they must be fit on X_train only and then applied to X_test.
See Part 3 (after the 3.11 split) for both encodings.
"""

print("\n--- Land Title: rare-category merge, then binary encoding ---")

"""
Land Title has 3 categories: Non Bumi Lot (3179), Bumi Lot (607), Malay
Reserved (7). Malay Reserved is under 0.2% of records - the same sparse-
category problem as Property Type's rare types. Merged into Bumi Lot rather
than one-hot as its own column: both Bumi Lot and Malay Reserved represent
land with restricted purchase eligibility (Bumiputera-only), in contrast to
Non Bumi Lot's unrestricted eligibility - a real, not arbitrary, grouping.

That merge leaves only 2 categories, so a full one-hot would just produce a
second column that's the exact logical negation of the first - a binary flag
(same treatment as Tenure Type's Freehold/Leasehold) is more direct.

Uses .map() rather than a `== 'Non Bumi Lot'` comparison, for the same reason
as Freehold Indicator - Land Title has 0 missing today, but `==` against NaN
silently returns False (misclassified as Bumi Lot) instead of staying NaN, so
this stays correct even if a future data pull or a reordered pipeline
introduces missing values here.
"""
df['Land Title'] = df['Land Title'].replace('Malay Reserved', 'Bumi Lot')
print(f"'Land Title' counts after merging Malay Reserved into Bumi Lot:\n{df['Land Title'].value_counts()}")

df['Is_Non_Bumi_Lot'] = df['Land Title'].map({'Non Bumi Lot': 1, 'Bumi Lot': 0})
print(f"\n'Is_Non_Bumi_Lot' value counts:\n{df['Is_Non_Bumi_Lot'].value_counts()}")
print("\n" + "-"*60)

print("\n--- Tenure Type: binary encoding ---")

"""
Only 2 categories (Freehold 2311, Leasehold 1482, 0 missing) - a full one-hot
would produce a second column that's the exact logical negation of the
first (the "dummy variable trap"), so a single binary indicator carries the
same information with one fewer redundant column.

Uses .map() rather than a `== 'Freehold'` comparison - `==` against NaN
silently returns False, which would misclassify a genuinely unknown tenure
type as Leasehold/0 rather than leaving it NaN. .map() preserves NaN as NaN
if this column ever does have missing values in a future data pull.
"""
df['Freehold Indicator'] = df['Tenure Type'].map({'Freehold': 1, 'Leasehold': 0})
print(f"'Freehold Indicator' value counts:\n{df['Freehold Indicator'].value_counts(dropna=False)}")
print("\n" + "-"*60)

print("\n--- Floor Range: ordinal encoding ---")

"""
Low/Medium/High have a real order (higher floor is a meaningfully different
attribute, not an arbitrary label), so ordinal encoding preserves that
monotonic relationship instead of discarding it into unordered one-hot
columns. "Unknown" (filled in back in 3.5) doesn't fit that order, so it maps
to NaN here rather than being forced into a fake rank.

Floor_Range_Known keeps the Unknown/observed distinction visible as its own
signal, the same pairing already used for Property Age/Is_Off_Plan.

Filling Floor_Range_Ordinal's NaN (with the train-set median) is deliberately
NOT done here - it happens in 3.11, after the train/test split, using only
X_train's median, for the same leakage reason Completion Year / # of Floors /
Total Units / Parking Lot's imputation is deferred there. Left as real NaN
for now so 3.9's output isn't quietly contaminated by a full-dataset statistic.

Floor_Range_Known uses .map() rather than `!= 'Unknown'` - Floor Range has 0
missing today (filled with the string "Unknown" back in 3.5), but a `!=`
comparison would silently treat a future genuine NaN as "known" (1) instead
of leaving it NaN, if this step is ever reordered to run before 3.5's fill.
"""
df['Floor_Range_Ordinal'] = df['Floor Range'].map({'Low': 1, 'Medium': 2, 'High': 3})
df['Floor_Range_Known'] = df['Floor Range'].map(lambda x: np.nan if pd.isna(x) else int(x != 'Unknown'))

print(f"'Floor_Range_Ordinal' value counts (NaN = Unknown, imputed later in 3.11):\n{df['Floor_Range_Ordinal'].value_counts(dropna=False)}")
print(f"\n'Floor_Range_Known' value counts:\n{df['Floor_Range_Known'].value_counts()}")
print("\n" + "-"*60)

print("\n--- Facilities: text cleanup + multi-hot encoding ---")

"""
Cleanup pipeline, applied before binarising: split on comma, strip whitespace,
title-case for consistency, drop purely numeric junk (the same "10" artefact
filtered in Listed_Facility_Count's Ad List 95706905).

"Merge near-duplicate wording" (e.g. "Badminton" vs "Badminton Court") was
planned but turns out to have nothing to merge: the full vocabulary across
all 3793 rows was checked earlier and is exactly 14 distinct, non-overlapping
facility names (Parking, Security, Playground, Swimming Pool, Lift,
Gymnasium, Minimart, Barbeque area, Jogging Track, Multipurpose hall, Sauna,
Tennis Court, Club house, Squash Court) plus the one junk "10" - no synonyms,
no case variants. The merge step is a no-op here, kept in the pipeline in
case a future data pull introduces messier wording.
"""
def clean_facility_list(facilities_text):
    if pd.isna(facilities_text):
        return []
    items = [item.strip().title() for item in facilities_text.split(',')]
    return [item for item in items if item and not item.isdigit()]

facility_lists = df['Facilities'].apply(clean_facility_list)

mlb = MultiLabelBinarizer()
facility_encoded = pd.DataFrame(
    mlb.fit_transform(facility_lists),
    columns=['Has_' + c.replace(' ', '_') for c in mlb.classes_],
    index=df.index
).astype(int)  # MultiLabelBinarizer already outputs int64, but made explicit rather than relied on
df = pd.concat([df, facility_encoded], axis=1)

print(f"Distinct facility types found: {len(mlb.classes_)}")
print(f"Columns created: {list(facility_encoded.columns)}")
print(f"\nColumn totals:\n{facility_encoded.sum().sort_values(ascending=False)}")
print("\n" + "-"*60)

df.to_csv(os.path.join(PROCESSED_DIR, "houses_cleaned.csv"), index=False)
joblib.dump(df, os.path.join(PROCESSED_DIR, "houses_cleaned.pkl"))
print(f"\nSaved to {PROCESSED_DIR}: {df.shape}")

# ============================================================
# 3.7 Feature Selection (executed after 3.8/3.9, since it drops
# columns that 3.8/3.9 needed to still exist while extracting from them)
# ============================================================
print("\n" + "="*60)
print("STEP 3.7: FEATURE SELECTION")
print("="*60)

print("\n--- Stage A: columns with no predictive value, unrelated to any 3.8/3.9 engineering ---")

"""
These were never going to be used, encoded or otherwise - dropped purely on
their own merits (unstructured text, unique identifier, zero variance, agent/
listing-firm metadata unrelated to the property, or high cardinality with no
extractable pattern), independent of anything built in 3.8/3.9.
"""
cols_no_engineering = [
    'description', 'Ad List',
    'Nearby School', 'Nearby Mall', 'Nearby Railway Station',
    'Category',
    'Firm Type', 'Firm Number', 'REN Number',
    'Building Name', 'Developer',
]

print("Non-null counts before drop:")
for c in cols_no_engineering:
    print(f"  {c:25s} {df[c].notna().sum()} non-null")

print("\n--- Stage B: raw columns already superseded by a 3.8/3.9 engineered feature ---")

"""
Each of these was kept alive specifically so 3.8/3.9 could extract from it
(Has_X flags from the 7 nearby-amenity columns, Freehold Indicator from
Tenure Type, Is_Non_Bumi_Lot from Land Title, Floor_Range_Ordinal/Known from
Floor Range, multi-hot from Facilities). Now that every extraction is done,
the raw source column is redundant with its own derived feature(s) already
in df.

State and Property Type are deliberately NOT dropped here, unlike the other
engineered-from columns. Their rare-category merge + one-hot encoding is
fit-dependent (the merge threshold is a statistic of the data) and is
deferred to Part 3, after the 3.11 split, so both raw text columns have to
stay alive in df past this point - they're dropped later, from X_train/
X_test individually, right after that encoding runs.

Completion Year is dropped for a different, stronger reason than the others:
verified Property Age == REFERENCE_YEAR - Completion Year exactly, for every
non-off-plan row, with zero exceptions. This isn't approximate overlap like
Total_Rooms - it's the same variable in different units, so keeping both
would hand a linear model perfectly collinear inputs for no benefit.
"""
cols_replaced_by_engineering = [
    'Address',
    'Completion Year',
    'Bus Stop', 'Mall', 'Park', 'School', 'Hospital', 'Highway', 'Railway Station',
    'Tenure Type',
    'Land Title',
    'Floor Range',
    'Facilities',
]

print("Non-null counts before drop:")
for c in cols_replaced_by_engineering:
    print(f"  {c:25s} {df[c].notna().sum()} non-null")

shape_before_drop = df.shape
df = df.drop(columns=cols_no_engineering + cols_replaced_by_engineering)

print(f"\nShape before 3.7 drop: {shape_before_drop}")
print(f"Shape after 3.7 drop:  {df.shape}")
print(f"Columns dropped: {len(cols_no_engineering) + len(cols_replaced_by_engineering)}")
print("\n" + "-"*60)

print("\n--- Post-drop duplicate re-check ---")

"""
3.2 de-duplicated the original 32-column dataset, but that check can't see
duplicates that only become identical once identifying columns (Ad List,
Address, Building Name, description, etc.) are gone. Computed fresh below on
whatever df looks like after this section's Stage A/B drop - counts are
printed dynamically rather than hardcoded, since the column set dropped here
(24 columns; State and Property Type are deliberately kept, see Stage B's
docstring above) differs from earlier pipeline iterations.

These rows are NOT removed - unlike the 3.2 exact duplicates (which really
were the same scrape captured twice), these are genuinely different real
listings (different Ad List, Address, description) that simply can't be
told apart once the identifying columns are gone. Deleting them would throw
away real observations for no data-quality reason.

The actual risk is narrower: if one of these groups gets split across the
train/test split, the test set ends up containing a row the model already
saw in training, silently inflating any evaluation metric. The fix belongs
in 3.11 (a group-aware split, keeping every group on one side), not here -
this matches the same principle already recorded for the Stage-3 near-
duplicate relistings in notes_near_duplicate_relistings.md: "use a group-
aware train/test split... this avoids leakage independent of the dedup
decision."

row_group_ids is computed HERE, right after this section's drop, and BEFORE
State/Property Type are merged/encoded in Part 3 - grouping on the still-raw,
unmerged text values gives the same group structure as grouping after the
merge would (verified: rare-category merging only collapses distinct labels
within a column into 'Other', it never changes which rows already agreed or
disagreed with each other on that column), so there is no need to wait for
Part 3 to compute this.
"""
row_group_ids = df.groupby(list(df.columns), dropna=False).ngroup()
group_sizes = row_group_ids.value_counts()
print(f"Row-identical groups on the post-drop {df.shape[1]} columns: {(group_sizes > 1).sum()}")
print(f"Rows involved in those groups:                                {(group_sizes[group_sizes > 1]).sum()}")
print("Not removed - see docstring. Handled instead via a group-aware split in 3.11.")
print("\n" + "-"*60)

df.to_csv(os.path.join(PROCESSED_DIR, "houses_cleaned.csv"), index=False)
joblib.dump(df, os.path.join(PROCESSED_DIR, "houses_cleaned.pkl"))
print(f"\nSaved to {PROCESSED_DIR}: {df.shape}")

# ============================================================
# 3.11 Train-Test Split
# ============================================================
print("\n" + "="*60)
print("STEP 3.11: TRAIN-TEST SPLIT")
print("="*60)

print("\n--- Split ---")

"""
Split happens before any statistic-fitting step (imputation medians, scaler
mean/std) - those must be computed from X_train only, so the split has to
exist first. X and y are split in a single call, not two separate calls, so
X_train/y_train are guaranteed to be the same rows (two separate calls risk
misaligned rows even with the same random_state).

Group-aware split (GroupShuffleSplit), not a plain train_test_split - 3.7's
post-drop duplicate check (printed above) found a handful of groups of
row-identical listings that a plain random split could tear apart, landing
some rows of the same group in train and others in test (test would then
contain a row identical to one the model trained on, inflating evaluation
metrics). row_group_ids has one ID per unique row-content group - mostly
singleton groups (a listing with no row-identical twin) plus the small
number of multi-row groups reported above, so the 80:20 ratio is barely
affected in practice (each multi-row group only forces a couple of rows to
move together, out of thousands).

Regression target (price, continuous), not classification, so no stratify=.
"""
X = df.drop(columns=['price'])
y = np.log(df['price'])

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=row_group_ids))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape} | y_test:  {y_test.shape}")
print(f"Actual test proportion: {len(X_test) / (len(X_train) + len(X_test)):.3f} (target was 0.2 - group constraint causes minor rounding)")
assert len(X_train) == len(y_train)
assert len(X_test) == len(y_test)

# ============================================================
# Part 3 - statistic-fitting steps (fit on X_train only, applied to X_test)
# ============================================================
X_train = X_train.copy()
X_test = X_test.copy()

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
list is applied to X_test's State column - the same leakage rule already
used for the 3.11 imputation medians and 3.10's scaler.

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
would produce columns that are almost entirely 0, and the group-aware split
could easily leave one of them with zero rows in X_train.
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

print("\n--- Dropping raw State / Property Type text columns (superseded by the one-hot columns above) ---")
X_train = X_train.drop(columns=['State', 'Property Type'])
X_test = X_test.drop(columns=['State', 'Property Type'])
print("\n" + "-"*60)

print("\n--- Missing-value imputation (median + indicator, fit on X_train only) ---")

"""
These 5 columns still carry real NaN, deliberately left unfilled since 3.5
specifically to avoid computing a statistic from the full dataset before the
split existed. Each gets a missing-indicator flag before the median fill -
same reasoning as Facilities_Recorded/Floor_Range_Known elsewhere in this
pipeline: an imputed median and a genuinely-observed value are not the same
information, and collapsing that distinction silently would hide it from the
model. Floor_Range_Ordinal is excluded here - it already has that indicator
(Floor_Range_Known, built in 3.9), so adding a second one would be redundant.

The median itself comes from X_train.median() (skipna=True by default, so
already-observed values compute it correctly), then the SAME value is used to
fill X_test - X_test's own median is never touched, which is what avoids
leaking test-set information into the imputation.
"""
IMPUTE_COLS = {
    'Property Age': 'Property_Age_Missing',
    '# of Floors': 'Num_Floors_Missing',
    'Total Units': 'Total_Units_Missing',
    'Parking Lot': 'Parking_Lot_Missing',
    'Property Size': 'Property_Size_Missing',
}

for col, flag_col in IMPUTE_COLS.items():
    X_train[flag_col] = X_train[col].isna().astype(int)
    X_test[flag_col] = X_test[col].isna().astype(int)

    train_median = X_train[col].median()
    X_train[col] = X_train[col].fillna(train_median)
    X_test[col] = X_test[col].fillna(train_median)

    print(f"{col:15s} train median={train_median:>8.2f} | train missing={X_train[flag_col].sum():>4d} | test missing={X_test[flag_col].sum():>4d}")

# Floor_Range_Ordinal: no new indicator (Floor_Range_Known already covers it), median fill only.
floor_range_median = X_train['Floor_Range_Ordinal'].median()
X_train['Floor_Range_Ordinal'] = X_train['Floor_Range_Ordinal'].fillna(floor_range_median)
X_test['Floor_Range_Ordinal'] = X_test['Floor_Range_Ordinal'].fillna(floor_range_median)
print(f"{'Floor_Range_Ordinal':15s} train median={floor_range_median:>8.2f} | (indicator already exists: Floor_Range_Known)")

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
# 3.10 Feature Scaling (executed after 3.11's split, using only X_train
# to fit, so no test-set information leaks into the scaling statistics)
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
(Has_*/Is_Non_Bumi_Lot/Freehold Indicator/Floor_Range_Known/the imputation
*_Missing flags) are left unscaled (already bounded 0/1, scaling a dummy
isn't meaningful). Floor_Range_Ordinal is included in scaling, though it
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
Saved under _scaled filenames, not overwriting 3.11's X_train.csv/X_test.csv -
tree-based models (Decision Tree/Random Forest/Gradient Boosting) don't need
scaling and can use 3.11's unscaled-but-imputed version directly, without
having to invert this transform to get back the original values.
"""
X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train_scaled.csv"), index=False)
X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test_scaled.csv"), index=False)
joblib.dump((X_train, X_test, y_train, y_test), os.path.join(PROCESSED_DIR, "train_test_split_scaled.pkl"))
joblib.dump(scaler, os.path.join(PROCESSED_DIR, "scaler.pkl"))
print(f"\nSaved scaled X_train_scaled/X_test_scaled (3.11's unscaled X_train.csv/X_test.csv left untouched) to {PROCESSED_DIR}")

# ============================================================
# 3.12 Final Dataset Structure Summary
# ============================================================
print("\n" + "="*60)
print("STEP 3.12: FINAL DATASET STRUCTURE SUMMARY")
print("="*60)

"""
Every list below is derived from variables already in memory (RAW_COLUMNS,
cols_no_engineering, cols_replaced_by_engineering, df.columns) rather than
typed out by hand, so this section can't silently drift out of sync with
what the pipeline actually did further up the script.
"""
retained_raw_cols = [c for c in RAW_COLUMNS if c in df.columns]
dropped_cols = [c for c in RAW_COLUMNS if c not in df.columns]
engineered_cols = [c for c in df.columns if c not in RAW_COLUMNS]

print(f"\nFinal cleaned dataset (pre-split): {df.shape[0]} rows x {df.shape[1]} columns")
print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

print(f"\n--- Raw features retained as-is ({len(retained_raw_cols)}) ---")
print(retained_raw_cols)

print(f"\n--- Raw features dropped ({len(dropped_cols)}) ---")
print(dropped_cols)

print(f"\n--- Engineered/encoded features created ({len(engineered_cols)}) ---")
print(engineered_cols)

print(f"\n--- Missing value confirmation ---")
print(f"Remaining NaN in X_train: {X_train.isna().sum().sum()}")
print(f"Remaining NaN in X_test:  {X_test.isna().sum().sum()}")

remaining_nan_df = df.isna().sum()
remaining_nan_df = remaining_nan_df[remaining_nan_df > 0]
if len(remaining_nan_df) == 0:
    print("Remaining NaN in df (pre-split): 0")
else:
    print(f"Remaining NaN in df (pre-split): {remaining_nan_df.sum()} total, in {len(remaining_nan_df)} column(s):")
    print(remaining_nan_df)
    print("Explanation: these are the columns 3.11 imputes AFTER the train/test split")
    print("(median fit on X_train only). df itself is left with real NaN on purpose,")
    print("so EDA reads genuinely observed values, not a full-dataset statistic that")
    print("would leak into what should be an X_train-only computation.")

print("\n--- Summary table ---")

"""
Every value below is read from a variable already computed earlier in this
script (RAW_COLUMNS/cols_no_engineering/cols_replaced_by_engineering/
X_train/X_test/df), not re-typed by hand - so this table can't silently
drift out of sync with what the pipeline actually did.

"Row-identical groups" is intentionally NOT "remaining duplicate rows" -
df.duplicated().sum() would report a nonzero count here, which would
misleadingly read as "a cleanup step that didn't finish." It's the opposite:
these rows are different real listings that were deliberately kept (not
3.2's kind of duplicate - see the 3.7 post-drop re-check above), with the
actual train/test leakage risk handled by 3.11's group-aware split instead.
This row reports the group count as evidence that risk was addressed, not as
an outstanding cleanup item.
"""
_train_tagged = X_train.assign(price=y_train.values, _src='train')
_test_tagged = X_test.assign(price=y_test.values, _src='test')
_combined = pd.concat([_train_tagged, _test_tagged], ignore_index=True)
_feature_cols = [c for c in _combined.columns if c != '_src']
_cross_dup = _combined[_combined.duplicated(subset=_feature_cols, keep=False)]
_sources_per_group = _cross_dup.groupby(_feature_cols, dropna=False)['_src'].apply(set)
cross_split_leaked_groups = int((_sources_per_group.apply(len) > 1).sum())

"""
X_train/X_test' column count differs from df's pre-split feature count for
two reasons, both introduced in Part 3 (after the 3.11 split, fit on X_train
only): State/Property Type's one-hot columns are ADDED (their raw text
columns stay in df - only dropped from X_train/X_test, once the fit-dependent
merge+encoding is done - see the 3.7 Stage B docstring), and a *_Missing
indicator flag is ADDED per imputed column. Each count below is read from a
variable already computed earlier in Part 3 (state_dummies_train /
property_type_dummies_train / IMPUTE_COLS), not hardcoded, so this table
can't silently drift out of sync with what the pipeline actually did.
"""
state_onehot_added = len(state_dummies_train.columns)
property_type_onehot_added = len(property_type_dummies_train.columns)
missing_indicator_flags_added = len(IMPUTE_COLS)
raw_cols_dropped_post_split = 2  # State, Property Type - dropped from X_train/X_test only (Part 3), not from df

summary_table = pd.DataFrame({
    "Item": [
        "Final number of rows (pre-split df)",
        "Final number of features (X, pre-split)",
        "Numerical features",
        "Non-numeric features remaining",
        "Raw features retained as-is",
        "Raw features dropped (Section 3.7)",
        "Engineered features created (3.8+3.9 fixed rules)",
        "State one-hot columns added (Part 3, fit on X_train)",
        "Property Type one-hot columns added (Part 3, fit on X_train)",
        "Raw State/Property Type columns dropped post-split (Part 3)",
        "Missing-value indicator flags added (Part 3)",
        "Final number of features in X_train/X_test",
        "Training set shape",
        "Testing set shape",
        "Remaining missing values (X_train + X_test)",
        "Row-identical groups retained (not deleted - see 3.7 note)",
        "Row-identical groups split across train/test (should be 0)",
    ],
    "Result": [
        df.shape[0],
        df.drop(columns=['price']).shape[1],
        len(df.select_dtypes(include=['int64', 'float64']).columns),
        len(df.select_dtypes(exclude=['int64', 'float64']).columns),
        len(retained_raw_cols),
        len(dropped_cols),
        len(engineered_cols),
        state_onehot_added,
        property_type_onehot_added,
        raw_cols_dropped_post_split,
        missing_indicator_flags_added,
        X_train.shape[1],
        str(X_train.shape),
        str(X_test.shape),
        int(X_train.isna().sum().sum() + X_test.isna().sum().sum()),
        int((row_group_ids.value_counts() > 1).sum()),
        cross_split_leaked_groups,
    ]
})
print(summary_table.to_string(index=False))
