
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

# ============================================================
# 0. Load Data & Shared Directories
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EDA_DIR = os.path.join(BASE_DIR, "data", "eda")   # train_for_eda.csv, all EDA figures, and eda_results.xlsx all live here
os.makedirs(EDA_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(EDA_DIR, "train_for_eda.csv"))

EXCEL_PATH = os.path.join(EDA_DIR, "eda_results.xlsx")


def write_sheets(sheets: dict):
    """
    Write {sheet_name: dataframe} into the shared workbook. Only the sheets
    passed in are replaced; every other sheet already in the workbook
    (from other 4.2 subsections) is left untouched.
    """
    write_mode = 'a' if os.path.exists(EXCEL_PATH) else 'w'
    writer_kwargs = {'if_sheet_exists': 'replace'} if write_mode == 'a' else {}
    with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl', mode=write_mode, **writer_kwargs) as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)


# ============================================================
# 4.2.1 CONTINUOUS NUMERICAL VARIABLES - Property Size (sq.ft.)
# ============================================================
print("=" * 60)
print("SECTION 4.2.1: PROPERTY SIZE (sq.ft.)")
print("=" * 60)

size = df['Property Size'].dropna()

# --- Histogram ---
plt.figure(figsize=(8, 5))
sns.histplot(size, bins=50, kde=True, color='steelblue')
plt.title('Distribution of Property Size (sq.ft.)')
plt.xlabel('Property Size (sq.ft.)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "fig_4_2_1_property_size_hist.png"), dpi=150, bbox_inches="tight")
plt.close()

# --- Boxplot ---
plt.figure(figsize=(8, 3))
sns.boxplot(x=size, color='lightcoral')
plt.title('Boxplot of Property Size (sq.ft.)')
plt.xlabel('Property Size (sq.ft.)')
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "fig_4_2_1_property_size_box.png"), dpi=150, bbox_inches="tight")
plt.close()

# --- Summary statistics ---
summary_table = pd.DataFrame({
    'Statistic': ['Mean', 'Median', '25th Percentile', '75th Percentile', 'Min', 'Max'],
    'Value': [
        size.mean(), size.median(), size.quantile(0.25),
        size.quantile(0.75), size.min(), size.max()
    ]
})
print("\n--- Summary Statistics ---")
print(summary_table.to_string(index=False))

# --- Outlier detection (IQR method) ---
Q1 = size.quantile(0.25)
Q3 = size.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Property Size'] < lower_bound) | (df['Property Size'] > upper_bound)]
size_cols = [c for c in ['Property Size', 'Bedroom', 'Bathroom', 'price'] if c in df.columns]
outliers_top10 = outliers.sort_values('Property Size', ascending=False)[size_cols].head(10)

print("\n--- Outlier Detection (IQR method) ---")
print(f"Lower bound: {lower_bound:.1f} sq.ft.")
print(f"Upper bound: {upper_bound:.1f} sq.ft.")
print(f"Number of outliers: {len(outliers)} ({len(outliers) / len(df) * 100:.1f}% of data)")
print("\nTop 10 largest properties (potential outliers):")
print(outliers_top10.to_string(index=False))

outlier_meta = pd.DataFrame({
    'Metric': ['Lower Bound', 'Upper Bound', 'Num Outliers', 'Pct Outliers'],
    'Value': [lower_bound, upper_bound, len(outliers), f"{len(outliers) / len(df) * 100:.1f}%"]
})

# --- Log transform (to address right skew) ---
df['log_Property_Size'] = np.log1p(df['Property Size'])
log_size = df['log_Property_Size'].dropna()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(size, bins=50, kde=True, color='steelblue', ax=axes[0, 0])
axes[0, 0].set_title('Original: Property Size (sq.ft.)')
sns.histplot(log_size, bins=50, kde=True, color='seagreen', ax=axes[0, 1])
axes[0, 1].set_title('Log-Transformed: log(Property Size + 1)')
sns.boxplot(x=size, color='lightcoral', ax=axes[1, 0])
axes[1, 0].set_title('Original Boxplot')
sns.boxplot(x=log_size, color='lightgreen', ax=axes[1, 1])
axes[1, 1].set_title('Log-Transformed Boxplot')
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "fig_4_2_1_property_size_log_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()

skew_table = pd.DataFrame({
    'Version': ['Original', 'Log-Transformed'],
    'Skewness': [skew(size), skew(log_size)]
})
print("\n--- Skewness Comparison ---")
print(skew_table.to_string(index=False))

write_sheets({
    'PropSize_Summary': summary_table,
    'PropSize_Skewness': skew_table,
})
# Outliers sheet needs two tables stacked with a gap row (metadata, then
# top-10 table), so it's written directly rather than via write_sheets().
with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    outlier_meta.to_excel(writer, sheet_name='PropSize_Outliers', index=False, startrow=0)
    outliers_top10.to_excel(writer, sheet_name='PropSize_Outliers', index=False, startrow=len(outlier_meta) + 2)

print(f"\n[4.2.1] Figures saved to: {EDA_DIR}")
print("  fig_4_2_1_property_size_hist.png")
print("  fig_4_2_1_property_size_box.png")
print("  fig_4_2_1_property_size_log_comparison.png")
print(f"[4.2.1] Workbook sheets written: PropSize_Summary, PropSize_Outliers, PropSize_Skewness")


# ============================================================
# 4.2.2 DISCRETE NUMERICAL VARIABLES - Bedroom, Bathroom, Parking Lot
# ============================================================
print("\n" + "=" * 60)
print("SECTION 4.2.2: DISCRETE NUMERICAL VARIABLES")
print("=" * 60)

DISCRETE_VARS = ['Bedroom', 'Bathroom', 'Parking Lot']

# --- Vertical bar plots (count plots) ---
for var in DISCRETE_VARS:
    counts = df[var].value_counts().sort_index()

    plt.figure(figsize=(7, 5))
    sns.countplot(x=df[var].dropna(), color='steelblue', order=counts.index)
    plt.title(f'Count Plot of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.tight_layout()

    fname = f"fig_4_2_2_{var.replace(' ', '_').lower()}_countplot.png"
    plt.savefig(os.path.join(EDA_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")

# --- Mode identification ---
mode_rows = []
for var in DISCRETE_VARS:
    s = df[var].dropna()
    mode_val = s.mode().iloc[0]          # first mode if multiple exist
    mode_count = (s == mode_val).sum()
    mode_pct = mode_count / len(s) * 100
    mode_rows.append({
        'Variable': var,
        'Mode': mode_val,
        'Mode Count': mode_count,
        'Mode %': round(mode_pct, 1),
        'Missing': df[var].isna().sum()
    })

mode_table = pd.DataFrame(mode_rows)
print("\n--- Mode Summary (Bedroom / Bathroom / Parking Lot) ---")
print(mode_table.to_string(index=False))

write_sheets({'Discrete_Mode': mode_table})

print(f"\n[4.2.2] Figures saved to: {EDA_DIR}")
for var in DISCRETE_VARS:
    print(f"  fig_4_2_2_{var.replace(' ', '_').lower()}_countplot.png")
print(f"[4.2.2] Workbook sheet written: Discrete_Mode")

print(f"\nAll of Section 4.2 complete. Workbook: {EXCEL_PATH}")


# ============================================================
# 4.2.3 CATEGORICAL VARIABLE DISTRIBUTIONS - Tenure Type, Land Title, Property Type
# ============================================================
print("\n" + "=" * 60)
print("SECTION 4.2.3: CATEGORICAL VARIABLE DISTRIBUTIONS")
print("=" * 60)

CATEGORICAL_VARS = ['Tenure Type', 'Land Title', 'Property Type']

cat_freq_tables = {}   # kept so the proportions block below can reuse them

for var in CATEGORICAL_VARS:
    counts = df[var].value_counts(dropna=False)
    pct = (counts / len(df) * 100).round(1)
    freq_table = pd.DataFrame({
        var: counts.index.astype(str),
        'Count': counts.values,
        'Percentage': pct.values
    }).sort_values('Count', ascending=False).reset_index(drop=True)
    cat_freq_tables[var] = freq_table

    print(f"\n--- {var}: Frequency Table ---")
    print(freq_table.to_string(index=False))

    # --- Horizontal bar chart ---
    plt.figure(figsize=(7, max(3, 0.5 * len(freq_table))))
    order = freq_table[var]
    sns.barplot(
        y=freq_table[var].astype(str), x=freq_table['Count'],
        color='steelblue', order=order
    )
    plt.title(f'Distribution of {var}')
    plt.xlabel('Count')
    plt.ylabel(var)
    for i, (count, pct_val) in enumerate(zip(freq_table['Count'], freq_table['Percentage'])):
        plt.text(count, i, f"  {count} ({pct_val}%)", va='center', fontsize=9)
    plt.tight_layout()

    fname = f"fig_4_2_3_{var.replace(' ', '_').lower()}_barchart.png"
    plt.savefig(os.path.join(EDA_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")

# --- Proportions for major category splits (as report call-outs) ---
print("\n--- Major Category Splits ---")
major_split_rows = []

if set(['Freehold', 'Leasehold']).issubset(set(df['Tenure Type'].dropna().unique())):
    t = df['Tenure Type'].value_counts()
    major_split_rows.append({
        'Comparison': 'Freehold vs Leasehold',
        'Category A': 'Freehold', 'Count A': int(t.get('Freehold', 0)),
        'Category B': 'Leasehold', 'Count B': int(t.get('Leasehold', 0)),
        'A %': round(t.get('Freehold', 0) / t.sum() * 100, 1),
        'B %': round(t.get('Leasehold', 0) / t.sum() * 100, 1),
    })

if set(['Non Bumi Lot', 'Bumi Lot']).issubset(set(df['Land Title'].dropna().unique())):
    l = df['Land Title'].value_counts()
    major_split_rows.append({
        'Comparison': 'Non-Bumi vs Bumi Lot',
        'Category A': 'Non Bumi Lot', 'Count A': int(l.get('Non Bumi Lot', 0)),
        'Category B': 'Bumi Lot', 'Count B': int(l.get('Bumi Lot', 0)),
        'A %': round(l.get('Non Bumi Lot', 0) / l.sum() * 100, 1),
        'B %': round(l.get('Bumi Lot', 0) / l.sum() * 100, 1),
    })

major_split_table = pd.DataFrame(major_split_rows)
print(major_split_table.to_string(index=False))

# ============================================================
# Write calculated tables into the shared workbook
#    - Each variable's frequency table gets its own sheet
#    - Major-split proportions get one combined sheet
# ============================================================
cat_sheets = {f"{var.replace(' ', '')}_Freq": tbl for var, tbl in cat_freq_tables.items()}
cat_sheets['Categorical_MajorSplits'] = major_split_table
write_sheets(cat_sheets)

print(f"\n[4.2.3] Figures saved to: {EDA_DIR}")
for var in CATEGORICAL_VARS:
    print(f"  fig_4_2_3_{var.replace(' ', '_').lower()}_barchart.png")
print(f"[4.2.3] Workbook sheets written: {list(cat_sheets.keys())}")

print(f"\nAll of Section 4.2 complete (4.2.1-4.2.3). Workbook: {EXCEL_PATH}")