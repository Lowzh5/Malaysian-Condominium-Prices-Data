import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

pd.set_option('display.max_columns', None)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DOWNLOADS_DIR = os.path.join(os.path.expanduser("~"), "Downloads")

df = pd.read_csv(os.path.join(PROCESSED_DIR, "train_for_eda.csv"))
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
ax.set_title("4.1.1 Price Distribution (raw, train set)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(DOWNLOADS_DIR, "eda_411_price_hist_kde.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: eda_411_price_hist_kde.png")

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

sns.histplot(log_price, kde=True, bins=50, color="#DD8452", ax=axes[1])
axes[1].set_title(f"log10(Price) (skew = {skew_log:.2f})")
axes[1].set_xlabel("log10(Price)")

plt.tight_layout()
plt.savefig(os.path.join(DOWNLOADS_DIR, "eda_412_price_log_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: eda_412_price_log_comparison.png")

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
ax.set_title("4.1.3 Price Distribution by Property Type (major categories)")
plt.tight_layout()
plt.savefig(os.path.join(DOWNLOADS_DIR, "eda_413_price_by_property_type.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: eda_413_price_by_property_type.png")
