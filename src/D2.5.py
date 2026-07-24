import pandas as pd

file_path = r"C:\Users\limhu\OneDrive\Documents\Data Science\D2.5.xlsx"
sheet = "houses - Copy"
df = pd.read_excel(file_path, sheet_name=sheet)

def to_numeric_digits_only(series):
    # For fields with currency symbols/units/thousand-separators (e.g. "RM 340 000", "1000 sq.ft.")
    s = series.astype(str).str.replace(r'[^0-9]', '', regex=True)
    s = s.replace('', pd.NA)
    return pd.to_numeric(s, errors='coerce')

def to_numeric_decimal(series):
    # For plain numeric fields that may have real decimals
    s = series.astype(str).str.extract(r'(-?\d+\.?\d*)')[0]
    return pd.to_numeric(s, errors='coerce')

def compute_stats(clean):
    clean = clean.dropna()
    return {
        'Count': int(clean.count()),
        'Mean': round(clean.mean(), 4),
        'Median': clean.median(),
        'Standard deviation': round(clean.std(ddof=1), 4),
        'Minimum': clean.min(),
        '25th percentile': clean.quantile(0.25),
        '75th percentile': clean.quantile(0.75),
        'Maximum': clean.max(),
    }

col_map = {
    'Price': ('price', 'digits'),
    'Property Size': ('Property Size', 'digits'),
    'Bedroom': ('Bedroom', 'decimal'),
    'Bathroom': ('Bathroom', 'decimal'),
    'Completion Year': ('Completion Year', 'decimal'),
    'Floors': ('# of Floors', 'decimal'),
    'Total Units': ('Total Units', 'decimal'),
    'Parking Lot': ('Parking Lot', 'decimal'),
}

results = {}
for label, (actual_col, mode) in col_map.items():
    clean = to_numeric_digits_only(df[actual_col]) if mode == 'digits' else to_numeric_decimal(df[actual_col])
    results[label] = compute_stats(clean)

stats_df = pd.DataFrame(results)
stats_df.index.name = 'Statistic'

with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    stats_df.to_excel(writer, sheet_name="Statistics")

print(stats_df)