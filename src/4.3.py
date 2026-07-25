import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EDA_DIR = os.path.join(BASE_DIR, "data", "eda")   # train_for_eda.csv, all EDA figures, and eda_results.xlsx all live here
os.makedirs(EDA_DIR, exist_ok=True)

eda_train = pd.read_csv(os.path.join(EDA_DIR, "train_for_eda.csv"))
eda_train['log10_price'] = np.log10(eda_train['price'])

EXCEL_PATH = os.path.join(EDA_DIR, "eda_results.xlsx")


def write_sheets(sheets: dict):
    """
    Write {sheet_name: dataframe} into the shared workbook. Only the sheets
    passed in are replaced; every other sheet already in the workbook
    (from other 4.x subsections) is left untouched.
    """
    write_mode = 'a' if os.path.exists(EXCEL_PATH) else 'w'
    writer_kwargs = {'if_sheet_exists': 'replace'} if write_mode == 'a' else {}
    with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl', mode=write_mode, **writer_kwargs) as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)


# ============================================================
# 4.3.1 Correlation Matrix and Heatmap
# ============================================================
print("\n" + "="*60)
print("STEP 4.3.1: CORRELATION MATRIX AND HEATMAP")
print("="*60)

CORE_COLS = ['Bedroom', 'Bathroom', 'Property Size', '# of Floors', 'Total Units',
             'Parking Lot', 'Property Age', 'Listed_Facility_Count',
             'Floor_Range_Ordinal', 'Is_Non_Bumi_Lot', 'Freehold Indicator',
             'log10_price']

core_corr = eda_train[CORE_COLS].corr(method='pearson')
core_corr_vs_price = core_corr['log10_price'].drop('log10_price').sort_values(ascending=False)

print("\n--- Core feature correlation with log10_price ---")
print(core_corr_vs_price)

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(core_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Pearson Correlation Matrix (Core Numeric Features)")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "fig_4_3_1_correlation_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: fig_4_3_1_correlation_heatmap.png")

AMENITY_COLS = [c for c in eda_train.columns if c.startswith('Has_')]

amenity_corr = eda_train[AMENITY_COLS + ['log10_price']].corr()['log10_price'].drop('log10_price').sort_values(ascending=False)

print("\n--- Amenity correlation with log10_price ---")
print(amenity_corr)

fig, ax = plt.subplots(figsize=(8, 8))
colors = ['crimson' if v < 0 else 'steelblue' for v in amenity_corr]
amenity_corr.plot(kind='barh', ax=ax, color=colors)
ax.set_title("Amenity Correlation with log10_price")
ax.set_xlabel("Pearson r")
ax.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "fig_4_3_1_amenity_correlation.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: fig_4_3_1_amenity_correlation.png")

write_sheets({
    'CoreCorr_Matrix': core_corr.reset_index().rename(columns={'index': 'Feature'}),
    'CoreCorr_vs_Price': core_corr_vs_price.reset_index().rename(
        columns={'index': 'Feature', 'log10_price': 'Correlation_with_log10_price'}),
    'AmenityCorr_vs_Price': amenity_corr.reset_index().rename(
        columns={'index': 'Amenity', 'log10_price': 'Correlation_with_log10_price'}),
})
print(f"[4.3.1] Workbook sheets written: CoreCorr_Matrix, CoreCorr_vs_Price, AmenityCorr_vs_Price")

# ============================================================
# 4.3.2 Price vs. Property Size Interaction
# ============================================================
print("\n" + "="*60)
print("STEP 4.3.2: PRICE VS PROPERTY SIZE INTERACTION")
print("="*60)

eda_train['Bedroom_Group'] = eda_train['Bedroom'].apply(lambda x: '5+' if x >= 5 else str(int(x)))

SIZE_LIMIT = 3000
PRICE_LIMIT = 2_000_000

mask = (eda_train['Property Size'] < SIZE_LIMIT) & (eda_train['price'] < PRICE_LIMIT)
filtered = eda_train[mask]

n_excluded = len(eda_train) - len(filtered)
print(f"\nTotal rows: {len(eda_train)}")
print(f"Rows excluded by filter (>= {SIZE_LIMIT} sqft OR >= RM{PRICE_LIMIT:,}): {n_excluded} "
      f"({n_excluded/len(eda_train)*100:.2f}%)")

filter_summary = pd.DataFrame({
    'Metric': ['Total rows', 'Size limit (sqft)', 'Price limit (RM)',
               'Rows excluded', 'Pct excluded'],
    'Value': [len(eda_train), SIZE_LIMIT, PRICE_LIMIT,
              n_excluded, round(n_excluded / len(eda_train) * 100, 2)]
})

bedroom_order = sorted(filtered['Bedroom_Group'].unique(),
                        key=lambda x: 99 if x == '5+' else int(x))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

axes[0].scatter(eda_train['Property Size'], eda_train['price'],
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
plt.savefig(os.path.join(EDA_DIR, "fig_4_3_2_price_vs_property_size.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: fig_4_3_2_price_vs_property_size.png")

# ------------------------------------------------------------
# 4.3.2b Price vs Property Size — Linear Trend
# ------------------------------------------------------------
slope, intercept = np.polyfit(filtered['Property Size'], filtered['price'], 1)
r_value = filtered['Property Size'].corr(filtered['price'])

print(f"\nLinear fit: price = {slope:.2f} * Property_Size + ({intercept:.2f})")
print(f"Pearson r (filtered subset): {r_value:.3f}")

linear_fit_table = pd.DataFrame({
    'Metric': ['Slope', 'Intercept', 'Pearson r (filtered subset)'],
    'Value': [slope, intercept, r_value]
})

fig, ax = plt.subplots(figsize=(9, 7))
sns.regplot(data=filtered, x='Property Size', y='price',
            scatter_kws={'alpha': 0.3, 's': 15, 'color': 'steelblue'},
            line_kws={'color': 'red', 'linewidth': 2},
            ax=ax)
ax.set_title("Price vs Property Size with Linear Trend (Filtered)")
ax.set_xlabel("Property Size (sqft)")
ax.set_ylabel("Price (RM)")
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "fig_4_3_2_trend_line.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: fig_4_3_2_trend_line.png")

write_sheets({
    'PropSize_FilterSummary': filter_summary,
    'PropSize_LinearFit': linear_fit_table,
})
print(f"[4.3.2] Workbook sheets written: PropSize_FilterSummary, PropSize_LinearFit")

# ============================================================
# 4.3.3 Price vs. Room Features
# ============================================================
print("\n" + "="*60)
print("STEP 4.3.3: PRICE VS ROOM FEATURES")
print("="*60)

eda_train['Bathroom_Group'] = eda_train['Bathroom'].apply(lambda x: '4+' if x >= 4 else str(int(x)))
eda_train['Bedroom_Group4'] = eda_train['Bedroom'].apply(lambda x: '4+' if x >= 4 else str(int(x)))

group_order = ['1', '2', '3', '4+']

median_by_bedroom = eda_train.groupby('Bedroom_Group4')['price'].median().reindex(group_order)
median_by_bathroom = eda_train.groupby('Bathroom_Group')['price'].median().reindex(group_order)

print("\n--- Median price by Bedroom group ---")
print(median_by_bedroom)

print("\n--- Median price by Bathroom group ---")
print(median_by_bathroom)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

sns.boxplot(data=eda_train, x='Bedroom_Group4', y='price', order=group_order,
            ax=axes[0], color='steelblue', showfliers=False)
axes[0].set_title("Price by Bedroom Count")
axes[0].set_xlabel("Bedroom")
axes[0].set_ylabel("Price (RM)")

sns.boxplot(data=eda_train, x='Bathroom_Group', y='price', order=group_order,
            ax=axes[1], color='darkorange', showfliers=False)
axes[1].set_title("Price by Bathroom Count")
axes[1].set_xlabel("Bathroom")
axes[1].set_ylabel("Price (RM)")

plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "fig_4_3_3_price_vs_room_features.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: fig_4_3_3_price_vs_room_features.png")

write_sheets({
    'MedianPrice_Bedroom': median_by_bedroom.reset_index().rename(columns={'price': 'Median_Price'}),
    'MedianPrice_Bathroom': median_by_bathroom.reset_index().rename(columns={'price': 'Median_Price'}),
})
print(f"[4.3.3] Workbook sheets written: MedianPrice_Bedroom, MedianPrice_Bathroom")

# ============================================================
# 4.3.4 Price vs Parking Lot Count
# ============================================================
print("\n" + "="*60)
print("STEP 4.3.4: PRICE VS PARKING LOT COUNT")
print("="*60)

n_missing = eda_train['Parking Lot'].isna().sum()
print(f"\nParking Lot missing: {n_missing} ({n_missing/len(eda_train)*100:.1f}%)")

parking_plot_df = eda_train.dropna(subset=['Parking Lot']).copy()
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
plt.savefig(os.path.join(EDA_DIR, "fig_4_3_4_price_vs_parking_lot.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: fig_4_3_4_price_vs_parking_lot.png")

write_sheets({
    'MedianPrice_Parking': median_by_parking.reset_index().rename(columns={'price': 'Median_Price'}),
})
print(f"[4.3.4] Workbook sheet written: MedianPrice_Parking")

print(f"\nAll of Section 4.3 complete. Workbook: {EXCEL_PATH}")
