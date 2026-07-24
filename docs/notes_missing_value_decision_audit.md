# Section 3.5 — Missing-Value Decision Audit

Context: `src/data_preprocessing.py` Section 3.5 only directly imputes three things:
`Bedroom`/`Bathroom` (recovered from description text), and `Floor Range` (filled with
`"Unknown"`). Every other column that still has missing values after 3.4 is left
untouched on purpose — either because it gets dropped in 3.7 (Feature Selection) or
transformed into a different representation in 3.8 (Feature Engineering), so imputing
its raw form in 3.5 would be discarded work. This file records the reasoning so it
doesn't need to be re-derived when writing the report.

## Deferred to 3.7 Feature Selection (column will be dropped)

| Column | Missing count | Reason |
|---|---|---|
| Building Name | 85 | High cardinality and planned exclusion from the baseline model |
| Developer | 1,645 | High cardinality and planned exclusion from the baseline model |
| Firm Type | 695 | Describes the listing agent/firm, not the property itself |
| Firm Number | 695 | Describes the listing agent/firm, not the property itself |
| REN Number | 989 | Describes the listing agent/firm, not the property itself |
| Nearby School | 3,133 | Redundant with the more complete 'School' column |
| Nearby Mall | 3,445 | Redundant with the more complete 'Mall' column |
| Nearby Railway Station | 3,448 | Redundant with the more complete 'Railway Station' column |

## Deferred to 3.8 Feature Engineering (column will be transformed)

| Column | Missing count | Reason |
|---|---|---|
| Address | 85 | Used to extract a 'State' feature; raw text is not imputed directly |
| Bus Stop | 3,126 | Converted to an information-present flag, not imputed |
| Mall | 3,345 | Converted to an information-present flag, not imputed |
| Park | 3,031 | Converted to an information-present flag, not imputed |
| School | 2,895 | Converted to an information-present flag, not imputed |
| Hospital | 3,468 | Converted to an information-present flag, not imputed |
| Highway | 3,658 | Converted to an information-present flag, not imputed |
| Railway Station | 3,342 | Converted to an information-present flag, not imputed |
| Facilities | 607 | Converted to `Listed_Facility_Count` + `Facilities_Recorded`; a count of 0 means "no facility information listed", not "no facilities" |

## Deferred to 3.11 Train-test Split (modelling pipeline, not 3.5)

| Column | Missing count | Reason |
|---|---|---|
| Completion Year | 1,907 | Median + missing indicator fit on X_train only, to avoid test-set leakage |
| # of Floors | 1,659 | Median + missing indicator fit on X_train only, to avoid test-set leakage |
| Total Units | 1,805 | Median + missing indicator fit on X_train only, to avoid test-set leakage |
| Parking Lot | 1,159 | Median + missing indicator fit on X_train only, to avoid test-set leakage |

These four are left with their real NaNs in the Section 3.5 cleaned dataset, so
Section 4 EDA reads only genuinely observed values (`df[col].notna()`). The median
and missing-indicator are computed later, fit on `X_train` only, inside the
modelling pipeline built at 3.11.
