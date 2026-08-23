import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

pd.set_option('display.max_columns', None)


def show_full_numbers(ax, axis='x'):
    """Replace matplotlib's default 1e6-style offset notation with plain,
    comma-separated numbers (e.g. 1,000,000 instead of 1e6)."""
    formatter = mticker.FuncFormatter(lambda val, _: f"{val:,.0f}")
    (ax.xaxis if axis == 'x' else ax.yaxis).set_major_formatter(formatter)


# Ordinal one-hue ramp (blue, monotone lightness): bin order carries meaning
# (age bands, facility tiers), so color encodes that order, not identity.
ORDINAL_RAMP = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b']
CHART_SURFACE = '#fcfcfb'
TEXT_PRIMARY = '#0b0b0b'
TEXT_MUTED = '#898781'
GRIDLINE = '#e1e0d9'
BASELINE = '#c3c2b7'
ANOMALY_COLOR = '#eb6834'   # accent (categorical slot 2, orange) for flagged points


def plot_ordinal_line_trend(labels, values, title, xlabel, ylabel, filename, smooth_window=5,
                             median_bracket=None, range_band=None):
    """Line chart for a full ordered sequence (every raw value, not binned).
    The raw line is thin/faint (real but noisy); a rolling-mean line on top
    carries the actual trend shape.

    median_bracket: optional (label, y_value, text, side) - a dot + label at
    one representative point for a group's median. The line is allowed to
    pass behind the label (opaque background keeps the number readable).
    range_band: optional (label_start, label_end, y_low, y_high, text) -
    two dashed reference lines + one label for a plateau range."""
    n = len(labels)
    raw_color = ORDINAL_RAMP[1]
    smooth_color = ORDINAL_RAMP[-1]

    smoothed = pd.Series(values).rolling(smooth_window, center=True, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    x = np.arange(n)
    ax.fill_between(x, smoothed, 0, color=smooth_color, alpha=0.08, zorder=1)
    ax.plot(x, values, color=raw_color, linewidth=1.2, alpha=0.45, zorder=2, label="Actual")
    ax.plot(x, smoothed, color=smooth_color, linewidth=2.5, solid_capstyle='round',
            solid_joinstyle='round', zorder=3, label=f"Smoothed ({smooth_window}-point avg)")

    label_box = dict(facecolor=CHART_SURFACE, edgecolor='none', pad=2)

    if median_bracket:
        label, y_value, text, side = median_bracket
        xi = labels.index(label)
        ax.scatter(xi, y_value, s=130, color=ORDINAL_RAMP[0], edgecolor=CHART_SURFACE,
                    linewidth=2, zorder=5)
        offset = max(values) * 0.03 * (1 if side == 'above' else -1)
        ax.text(xi, y_value + offset, text, ha='left', va='bottom' if side == 'above' else 'top',
                 fontsize=10, color=TEXT_PRIMARY, zorder=5, bbox=label_box)

    if range_band:
        label_start, label_end, y_low, y_high, text = range_band
        xi_start, xi_end = labels.index(label_start), labels.index(label_end)
        for y in (y_low, y_high):
            ax.hlines(y, xi_start, xi_end, color=TEXT_MUTED, linewidth=1.2,
                       linestyle=(0, (5, 3)), zorder=4)
        ax.text((xi_start + xi_end) / 2, y_high + max(values) * 0.03, text, ha='center', va='bottom',
                 fontsize=10, color=TEXT_PRIMARY, zorder=5, bbox=label_box)

    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TEXT_MUTED, fontsize=8,
                        rotation=90 if n > 20 else 0)
    ax.set_xlim(-0.5, n - 0.5)

    ax.set_title(title, color=TEXT_PRIMARY, fontsize=13, pad=32)
    ax.set_xlabel(xlabel, color=TEXT_MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT_MUTED, fontsize=10)
    show_full_numbers(ax, 'y')
    ax.tick_params(axis='y', colors=TEXT_MUTED, labelsize=9)
    ax.set_ylim(0, max(values) * 1.15)
    ax.legend(loc='lower right', bbox_to_anchor=(1, 1.01), ncol=2,
              frameon=False, labelcolor=TEXT_MUTED, fontsize=9)

    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(BASELINE)
    ax.tick_params(axis='x', length=0)

    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, filename), dpi=150, facecolor=CHART_SURFACE, bbox_inches="tight")
    print(f"\nSaved: {filename}")


def plot_emphasis_scatter(data, x_col, y_col, is_anomaly, xlabel, ylabel, title, filename,
                           line_x=None, line_y=None):
    """Scatter with all listings de-emphasized in gray, flagged anomalies
    highlighted in the accent hue, and an optional expected-value line."""
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    ax.scatter(data[x_col], data[y_col], s=14, color=BASELINE, alpha=0.5, zorder=1, label="All listings")
    if line_x is not None:
        ax.plot(line_x, line_y, color=ORDINAL_RAMP[1], linewidth=2, zorder=2, label="Expected (trend)")
    flagged = data[is_anomaly]
    ax.scatter(flagged[x_col], flagged[y_col], s=70, color=ANOMALY_COLOR,
               edgecolor=CHART_SURFACE, linewidth=1, zorder=3, label="Flagged anomaly")

    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=13, pad=32)
    ax.set_xlabel(xlabel, color=TEXT_MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT_MUTED, fontsize=10)
    show_full_numbers(ax, 'y')
    ax.tick_params(axis='y', colors=TEXT_MUTED, labelsize=9)
    ax.tick_params(axis='x', colors=TEXT_MUTED, labelsize=9)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(BASELINE)
    ax.legend(loc='lower right', bbox_to_anchor=(1, 1.01), ncol=3,
              frameon=False, labelcolor=TEXT_MUTED, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, filename), dpi=150, facecolor=CHART_SURFACE, bbox_inches="tight")
    print(f"\nSaved: {filename}")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDA_DIR = os.path.join(BASE_DIR, "data", "eda")   # train_for_eda.csv and all EDA figures live here
os.makedirs(EDA_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(EDA_DIR, "train_for_eda.csv"))
price = df['price']

# ============================================================
# 4.1.1 Price Distribution and Central Tendency
# ============================================================
print("=" * 60)
print("4.1.1: PRICE DISTRIBUTION AND CENTRAL TENDENCY")
print("=" * 60)

mean_p, median_p = price.mean(), price.median()
mode_p = price.mode().iloc[0]
std_p, min_p, max_p = price.std(), price.min(), price.max()

print(f"Mean:   RM {mean_p:,.2f}")
print(f"Median: RM {median_p:,.2f}")
print(f"Mode:   RM {mode_p:,.0f}  ({(price == mode_p).sum()} listings share this price)")
print(f"Std:    RM {std_p:,.2f}")
print(f"Min:    RM {min_p:,.0f}")
print(f"Max:    RM {max_p:,.0f}")

skew_raw = stats.skew(price)
if skew_raw > 0.5:
    direction = "right-skewed (long tail toward high prices)"
elif skew_raw < -0.5:
    direction = "left-skewed (long tail toward low prices)"
else:
    direction = "approximately symmetrical"
print(f"Skewness: {skew_raw:.3f} -> {direction}")

fig, ax = plt.subplots(figsize=(9, 6))
sns.histplot(price, kde=True, bins=50, color="#4C72B0", ax=ax)
ax.axvline(mean_p, color="#C44E52", linestyle="--", label=f"Mean = RM {mean_p:,.0f}")
ax.axvline(median_p, color="#55A868", linestyle="--", label=f"Median = RM {median_p:,.0f}")
ax.set_xlabel("Price (RM)")
ax.set_ylabel("Count")
ax.set_title("Price Distribution (raw, train set)")
ax.legend()
show_full_numbers(ax, 'x')
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "fig_411_price_hist_kde.png"), dpi=150, bbox_inches="tight")
print("Saved: fig_411_price_hist_kde.png")

# ============================================================
# 4.1.2 Skewness and Normality Assessment
# ============================================================
print("\n" + "=" * 60)
print("4.1.2: SKEWNESS AND NORMALITY ASSESSMENT")
print("=" * 60)

log_price = np.log10(price)
skew_log = stats.skew(log_price)

print(f"Raw price skewness:      {skew_raw:.3f}")
print(f"log10(price) skewness:   {skew_log:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(price, kde=True, bins=50, color="#4C72B0", ax=axes[0])
axes[0].set_title(f"Raw Price (skew = {skew_raw:.2f})")
axes[0].set_xlabel("Price (RM)")
show_full_numbers(axes[0], 'x')

sns.histplot(log_price, kde=True, bins=50, color="#DD8452", ax=axes[1])
axes[1].set_title(f"log10(Price) (skew = {skew_log:.2f})")
axes[1].set_xlabel("log10(Price)")

plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "fig_412_price_log_comparison.png"), dpi=150, bbox_inches="tight")
print("Saved: fig_412_price_log_comparison.png")

print(f"""
Why log transformation is required for ML models:
- Raw price is heavily right-skewed (skew = {skew_raw:.2f}) with a long tail of
  high-value outliers (max RM {max_p:,.0f} vs median RM {median_p:,.0f}). Models
  that assume normally distributed residuals (Linear/Ridge Regression) get
  their error term dominated by these few expensive properties, biasing
  predictions for the far more common mid-priced properties.
- log10 transform compresses that long tail, pulling skewness down to
  {skew_log:.2f} (near-symmetrical), stabilizing variance across the price
  range and letting errors be reasoned about in relative/percentage terms
  rather than absolute RM - which is also how this pipeline's actual model
  target is built (np.log(price) in data_preprocessing.py, natural log
  instead of log10, same purpose).
""")

# ============================================================
# 4.1.3 Price Distribution across Property Categories
# ============================================================
print("\n" + "=" * 60)
print("4.1.3: PRICE DISTRIBUTION ACROSS PROPERTY CATEGORIES")
print("=" * 60)

major_categories = ['Condominium', 'Apartment', 'Service Residence', 'Flat']
is_major = df['Property Type'].isin(major_categories)
sub = df[is_major].copy()
excluded_counts = df.loc[~is_major, 'Property Type'].value_counts().to_dict()
print(f"Rows in major 4 categories: {len(sub)} / {len(df)} "
      f"({len(df) - len(sub)} excluded as minor categories: {excluded_counts})")

summary = sub.groupby('Property Type')['price'].agg(
    Count='count', Mean='mean', Median='median',
    Q1=lambda s: s.quantile(0.25), Q3=lambda s: s.quantile(0.75),
    Min='min', Max='max', Std='std'
)
summary['IQR'] = summary['Q3'] - summary['Q1']
summary = summary[['Count', 'Mean', 'Median', 'IQR', 'Min', 'Max', 'Std']]
summary = summary.sort_values('Median', ascending=False)
print(summary.round(0))

highest_median_cat = summary['Median'].idxmax()
highest_var_cat = summary['Std'].idxmax()
print(f"\nHighest median price: {highest_median_cat} (RM {summary.loc[highest_median_cat, 'Median']:,.0f})")
print(f"Most price variance:  {highest_var_cat} (std = RM {summary.loc[highest_var_cat, 'Std']:,.0f})")

fig, ax = plt.subplots(figsize=(9, 6))
order = summary.index.tolist()
sns.boxplot(data=sub, x='Property Type', y='price', order=order, ax=ax)
ax.set_ylabel("Price (RM)")
ax.set_title("Price Distribution by Property Type (major categories)")
show_full_numbers(ax, 'y')
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "fig_413_price_by_property_type.png"), dpi=150, bbox_inches="tight")
print("Saved: fig_413_price_by_property_type.png")


# ============================================================
# 4.2.1 CONTINUOUS NUMERICAL VARIABLES - Property Size (sq.ft.)
# ============================================================
print("\n" + "=" * 60)
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
plt.savefig(os.path.join(EDA_DIR, "fig_421_property_size_hist.png"), dpi=150, bbox_inches="tight")

# --- Boxplot ---
plt.figure(figsize=(8, 3))
sns.boxplot(x=size, color='lightcoral')
plt.title('Boxplot of Property Size (sq.ft.)')
plt.xlabel('Property Size (sq.ft.)')
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "fig_421_property_size_box.png"), dpi=150, bbox_inches="tight")

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
plt.savefig(os.path.join(EDA_DIR, "fig_421_property_size_log_comparison.png"), dpi=150, bbox_inches="tight")

skew_table = pd.DataFrame({
    'Version': ['Original', 'Log-Transformed'],
    'Skewness': [stats.skew(size), stats.skew(log_size)]
})
print("\n--- Skewness Comparison ---")
print(skew_table.to_string(index=False))

print(f"\n[4.2.1] Figures saved to: {EDA_DIR}")
print("  fig_421_property_size_hist.png")
print("  fig_421_property_size_box.png")
print("  fig_421_property_size_log_comparison.png")


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

    fname = f"fig_422_{var.replace(' ', '_').lower()}_countplot.png"
    plt.savefig(os.path.join(EDA_DIR, fname), dpi=150, bbox_inches="tight")
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

print(f"\n[4.2.2] Figures saved to: {EDA_DIR}")
for var in DISCRETE_VARS:
    print(f"  fig_422_{var.replace(' ', '_').lower()}_countplot.png")


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

    fname = f"fig_423_{var.replace(' ', '_').lower()}_barchart.png"
    plt.savefig(os.path.join(EDA_DIR, fname), dpi=150, bbox_inches="tight")
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

print(f"\n[4.2.3] Figures saved to: {EDA_DIR}")
for var in CATEGORICAL_VARS:
    print(f"  fig_423_{var.replace(' ', '_').lower()}_barchart.png")

print(f"\nAll of Section 4.2 complete (4.2.1-4.2.3).")


# ============================================================
# 4.3.1 Correlation Matrix and Heatmap
# ============================================================
df['log10_price'] = np.log10(df['price'])

print("\n" + "=" * 60)
print("STEP 4.3.1: CORRELATION MATRIX AND HEATMAP")
print("=" * 60)

CORE_COLS = ['Bedroom', 'Bathroom', 'Property Size', '# of Floors', 'Total Units',
             'Parking Lot', 'Property Age', 'Listed_Facility_Count',
             'Floor_Range_Ordinal', 'Is_Non_Bumi_Lot', 'Freehold Indicator',
             'log10_price']

core_corr = df[CORE_COLS].corr(method='pearson')
core_corr_vs_price = core_corr['log10_price'].drop('log10_price').sort_values(ascending=False)

print("\n--- Core feature correlation with log10_price ---")
print(core_corr_vs_price)

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(core_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Pearson Correlation Matrix (Core Numeric Features)")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "fig_431_correlation_heatmap.png"), dpi=150, bbox_inches="tight")
print(f"\nSaved: fig_431_correlation_heatmap.png")

AMENITY_COLS = [c for c in df.columns if c.startswith('Has_')]

amenity_corr = df[AMENITY_COLS + ['log10_price']].corr()['log10_price'].drop('log10_price').sort_values(ascending=False)

print("\n--- Amenity correlation with log10_price ---")
print(amenity_corr)

fig, ax = plt.subplots(figsize=(8, 8))
colors = ['crimson' if v < 0 else 'steelblue' for v in amenity_corr]
amenity_corr.plot(kind='barh', ax=ax, color=colors)
ax.set_title("Amenity Correlation with log10_price")
ax.set_xlabel("Pearson r")
ax.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "fig_431_amenity_correlation.png"), dpi=150, bbox_inches="tight")
print(f"\nSaved: fig_431_amenity_correlation.png")

# ============================================================
# 4.3.2 Price vs. Property Size Interaction
# ============================================================
print("\n" + "=" * 60)
print("STEP 4.3.2: PRICE VS PROPERTY SIZE INTERACTION")
print("=" * 60)

df['Bedroom_Group'] = df['Bedroom'].apply(lambda x: '5+' if x >= 5 else str(int(x)))

SIZE_LIMIT = 3000
PRICE_LIMIT = 2_000_000

mask = (df['Property Size'] < SIZE_LIMIT) & (df['price'] < PRICE_LIMIT)
filtered = df[mask]

n_excluded = len(df) - len(filtered)
print(f"\nTotal rows: {len(df)}")
print(f"Rows excluded by filter (>= {SIZE_LIMIT} sqft OR >= RM{PRICE_LIMIT:,}): {n_excluded} "
      f"({n_excluded/len(df)*100:.2f}%)")

bedroom_order = sorted(filtered['Bedroom_Group'].unique(),
                        key=lambda x: 99 if x == '5+' else int(x))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

axes[0].scatter(df['Property Size'], df['price'],
                 s=15, alpha=0.4, color='steelblue')
axes[0].set_title("Full Data (Unfiltered)")
axes[0].set_xlabel("Property Size (sqft)")
axes[0].set_ylabel("Price (RM)")

palette = plt.cm.viridis(np.linspace(0, 1, len(bedroom_order)))
for color, grp in zip(palette, bedroom_order):
    subset = filtered[filtered['Bedroom_Group'] == grp]
    axes[1].scatter(subset['Property Size'], subset['price'],
                     s=15, alpha=0.6, color=color, label=f"{grp} Bedroom")

axes[1].set_title(f"Filtered (<{SIZE_LIMIT} sqft, <RM{PRICE_LIMIT:,}), colored by Bedroom")
axes[1].set_xlabel("Property Size (sqft)")
axes[1].set_ylabel("Price (RM)")
axes[1].legend(title="Bedroom", loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "fig_432_price_vs_property_size.png"), dpi=150, bbox_inches="tight")
print(f"\nSaved: fig_432_price_vs_property_size.png")

# ------------------------------------------------------------
# 4.3.2b Price vs Property Size — Linear Trend
# ------------------------------------------------------------
slope, intercept = np.polyfit(filtered['Property Size'], filtered['price'], 1)
r_value = filtered['Property Size'].corr(filtered['price'])

print(f"\nLinear fit: price = {slope:.2f} * Property_Size + ({intercept:.2f})")
print(f"Pearson r (filtered subset): {r_value:.3f}")

fig, ax = plt.subplots(figsize=(9, 7))
sns.regplot(data=filtered, x='Property Size', y='price',
            scatter_kws={'alpha': 0.3, 's': 15, 'color': 'steelblue'},
            line_kws={'color': 'red', 'linewidth': 2},
            ax=ax)
ax.set_title("Price vs Property Size with Linear Trend (Filtered)")
ax.set_xlabel("Property Size (sqft)")
ax.set_ylabel("Price (RM)")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "fig_432_trend_line.png"), dpi=150, bbox_inches="tight")
print(f"\nSaved: fig_432_trend_line.png")

# ============================================================
# 4.3.3 Price vs. Room Features
# ============================================================
print("\n" + "=" * 60)
print("STEP 4.3.3: PRICE VS ROOM FEATURES")
print("=" * 60)

df['Bathroom_Group'] = df['Bathroom'].apply(lambda x: '4+' if x >= 4 else str(int(x)))
df['Bedroom_Group4'] = df['Bedroom'].apply(lambda x: '4+' if x >= 4 else str(int(x)))

group_order = ['1', '2', '3', '4+']

median_by_bedroom = df.groupby('Bedroom_Group4')['price'].median().reindex(group_order)
median_by_bathroom = df.groupby('Bathroom_Group')['price'].median().reindex(group_order)

print("\n--- Median price by Bedroom group ---")
print(median_by_bedroom)

print("\n--- Median price by Bathroom group ---")
print(median_by_bathroom)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

sns.boxplot(data=df, x='Bedroom_Group4', y='price', order=group_order,
            ax=axes[0], color='steelblue', showfliers=False)
axes[0].set_title("Price by Bedroom Count")
axes[0].set_xlabel("Bedroom")
axes[0].set_ylabel("Price (RM)")

sns.boxplot(data=df, x='Bathroom_Group', y='price', order=group_order,
            ax=axes[1], color='darkorange', showfliers=False)
axes[1].set_title("Price by Bathroom Count")
axes[1].set_xlabel("Bathroom")
axes[1].set_ylabel("Price (RM)")

plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "fig_433_price_vs_room_features.png"), dpi=150, bbox_inches="tight")
print(f"\nSaved: fig_433_price_vs_room_features.png")

# ============================================================
# 4.3.4 Price vs Parking Lot Count
# ============================================================
print("\n" + "=" * 60)
print("STEP 4.3.4: PRICE VS PARKING LOT COUNT")
print("=" * 60)

n_missing = df['Parking Lot'].isna().sum()
print(f"\nParking Lot missing: {n_missing} ({n_missing/len(df)*100:.1f}%)")

parking_plot_df = df.dropna(subset=['Parking Lot']).copy()
parking_plot_df['Parking_Group'] = parking_plot_df['Parking Lot'].apply(
    lambda x: '3+' if x >= 3 else str(int(x)))

parking_order = ['1', '2', '3+']

median_by_parking = parking_plot_df.groupby('Parking_Group')['price'].median().reindex(parking_order)
print("\n--- Median price by Parking Lot group ---")
print(median_by_parking)

fig, ax = plt.subplots(figsize=(8, 7))
sns.boxplot(data=parking_plot_df, x='Parking_Group', y='price', order=parking_order,
            ax=ax, color='seagreen', showfliers=False)
ax.set_title(f"Price by Parking Lot Count (N={len(parking_plot_df)}, "
             f"{n_missing} missing excluded)")
ax.set_xlabel("Parking Lot")
ax.set_ylabel("Price (RM)")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "fig_434_price_vs_parking_lot.png"), dpi=150, bbox_inches="tight")
print(f"\nSaved: fig_434_price_vs_parking_lot.png")

print(f"\nAll of Section 4.3 complete.")


# ============================================================
# 4.4.1 Property Age Depreciation Pattern
# ============================================================
print("\n" + "=" * 60)
print("STEP 4.4.1: PROPERTY AGE DEPRECIATION PATTERN")
print("=" * 60)

n_age_missing = df['Property Age'].isna().sum()
print(f"Rows with missing Property Age (excluded from this pattern): {n_age_missing}")

age_trend = df.groupby('Property Age', observed=True)['price'].agg(
    Count='count', Median_Price='median')
print("\n--- Median Price by Property Age (every year) ---")
print(age_trend)

age_labels = [str(int(a)) for a in age_trend.index]
young_median = df.loc[df['Property Age'].between(0, 5), 'price'].median()

plot_ordinal_line_trend(
    labels=age_labels,
    values=age_trend['Median_Price'].tolist(),
    title="Median Price by Property Age (Depreciation Trend)",
    xlabel="Property Age (years)",
    ylabel="Median Price (RM)",
    filename="fig_441_price_by_property_age.png",
    median_bracket=("2", young_median, f"0-5 yrs median: RM {young_median:,.0f}", "above"),
    range_band=("20", age_labels[-1], 300000, 350000, "~RM 300,000 - 350,000"),
)

# ============================================================
# 4.4.2 Facility Count Pattern
# ============================================================
print("\n" + "=" * 60)
print("STEP 4.4.2: FACILITY COUNT PATTERN")
print("=" * 60)

facility_trend = df.groupby('Listed_Facility_Count', observed=True)['price'].agg(
    Count='count', Median_Price='median')
print("\n--- Median Price by Facility Count (every count) ---")
print(facility_trend)

plot_ordinal_line_trend(
    labels=[str(int(c)) for c in facility_trend.index],
    values=facility_trend['Median_Price'].tolist(),
    title="Median Price by Facility Count",
    xlabel="Listed Facility Count",
    ylabel="Median Price (RM)",
    filename="fig_442_price_by_facility_tier.png",
)

# ============================================================
# 4.4.3 Price vs Property Size Anomalies
# ============================================================
print("\n" + "=" * 60)
print("STEP 4.4.3: PRICE VS PROPERTY SIZE ANOMALIES")
print("=" * 60)

# Reuses the linear fit (slope, intercept) from 4.3.2. A listing far from
# that line is priced very differently than its size would predict -
# a bivariate anomaly, not the single-variable outliers already handled
# in Sections 3.6/3.7.
size_df = df.dropna(subset=['Property Size']).copy()
size_df['Predicted_Price'] = slope * size_df['Property Size'] + intercept
size_df['Residual'] = size_df['price'] - size_df['Predicted_Price']

is_anomaly_size = size_df['Residual'].abs().rank(ascending=False, method='first') <= 10
top_anomalies_size = size_df.loc[is_anomaly_size,
    ['Property Size', 'Bedroom', 'Bathroom', 'price', 'Predicted_Price', 'Residual']
].sort_values('Residual', key=abs, ascending=False)
print("\n--- Top 10 Price-vs-Size Anomalies (largest |actual - predicted|) ---")
print(top_anomalies_size.to_string(index=False))

fit_x = np.linspace(size_df['Property Size'].min(), size_df['Property Size'].max(), 100)
fit_y = slope * fit_x + intercept

plot_emphasis_scatter(
    data=size_df, x_col='Property Size', y_col='price', is_anomaly=is_anomaly_size,
    line_x=fit_x, line_y=fit_y,
    xlabel="Property Size (sqft)", ylabel="Price (RM)",
    title="Price vs Property Size - Flagged Anomalies",
    filename="fig_443_price_size_anomalies.png",
)

# ============================================================
# 4.4.4 Facility Tier Deviation Anomalies
# ============================================================
print("\n" + "=" * 60)
print("STEP 4.4.4: FACILITY TIER DEVIATION ANOMALIES")
print("=" * 60)

# Expected price = the 4.4.2 median for that listing's own facility count.
# A listing far from its own tier's median is anomalous relative to what
# similar-facility listings sell for.
df['Facility_Expected_Price'] = df['Listed_Facility_Count'].map(facility_trend['Median_Price'])
df['Facility_Residual'] = df['price'] - df['Facility_Expected_Price']

is_anomaly_fac = df['Facility_Residual'].abs().rank(ascending=False, method='first') <= 10
top_anomalies_fac = df.loc[is_anomaly_fac,
    ['Listed_Facility_Count', 'price', 'Facility_Expected_Price', 'Facility_Residual']
].sort_values('Facility_Residual', key=abs, ascending=False)
print("\n--- Top 10 Facility-Tier Anomalies (largest |actual - tier median|) ---")
print(top_anomalies_fac.to_string(index=False))

plot_emphasis_scatter(
    data=df, x_col='Listed_Facility_Count', y_col='price', is_anomaly=is_anomaly_fac,
    line_x=facility_trend.index.tolist(), line_y=facility_trend['Median_Price'].tolist(),
    xlabel="Listed Facility Count", ylabel="Price (RM)",
    title="Price vs Facility Count - Flagged Anomalies",
    filename="fig_444_facility_anomalies.png",
)

# ============================================================
# 4.4.5 Property Age Deviation Anomalies
# ============================================================
print("\n" + "=" * 60)
print("STEP 4.4.5: PROPERTY AGE DEVIATION ANOMALIES")
print("=" * 60)

# Expected price = the 4.4.1 median for that listing's own property age.
df['Age_Expected_Price'] = df['Property Age'].map(age_trend['Median_Price'])
df['Age_Residual'] = df['price'] - df['Age_Expected_Price']

is_anomaly_age = df['Age_Residual'].abs().rank(ascending=False, method='first') <= 10
top_anomalies_age = df.loc[is_anomaly_age,
    ['Property Age', 'price', 'Age_Expected_Price', 'Age_Residual']
].sort_values('Age_Residual', key=abs, ascending=False)
print("\n--- Top 10 Property-Age Anomalies (largest |actual - age median|) ---")
print(top_anomalies_age.to_string(index=False))

plot_emphasis_scatter(
    data=df.dropna(subset=['Property Age']), x_col='Property Age', y_col='price', is_anomaly=is_anomaly_age,
    line_x=age_trend.index.tolist(), line_y=age_trend['Median_Price'].tolist(),
    xlabel="Property Age (years)", ylabel="Price (RM)",
    title="Price vs Property Age - Flagged Anomalies",
    filename="fig_445_property_age_anomalies.png",
)

print(f"\nAll of Section 4.4 complete.")
