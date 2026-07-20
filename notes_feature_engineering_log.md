# Section 3.8 — Feature Engineering Log

Context: `src/data_preprocessing.py` Section 3.8 runs before Section 3.7 (Feature
Selection), because several columns dropped in 3.7 are the source material 3.8
extracts from (Address → State, the 7 nearby-amenity columns → Has_X flags).
Dropping first would destroy the information before it could be extracted.

Every feature below was tested against `price` with Pearson's r and a
significance test (or ANOVA for the one categorical case, State) before being
kept. This file records that evidence so it doesn't need to be re-derived when
writing the report.

## Kept

| Feature | Source | Formula / logic | Evidence |
|---|---|---|---|
| `State` | `Address` | Scans every comma-separated segment from the end backwards, case-insensitive, matched against a fixed list of 16 Malaysian states/federal territories (exact whole-segment match, not substring — so "Kampung Melaka" is never mistaken for the state "Melaka"). Rare states (<10 listings: Kedah, Kelantan, Labuan, Terengganu) merged into "Other". | ANOVA: F=36.83, p=2.98×10⁻⁹⁴ (n=3708, 15 groups). Median price ranges from RM1,200,000 (Kelantan) to RM170,000 (Labuan). |
| `Property Age` | `Completion Year` | `REFERENCE_YEAR (2023, the dataset's actual collection date per Kaggle, not the runtime year) − Completion Year`. Negative results (off-plan) set to NaN — see `Is_Off_Plan` below. | r=-0.224, p=1.27×10⁻²² (n=1864). Direction matches real-estate depreciation logic; monotonic decline across age buckets (RM505k at 0-5yrs → RM310k at 31+yrs). |
| `Is_Off_Plan` | `Completion Year` | 1 if `Completion Year > REFERENCE_YEAR` (still under construction at collection time), 0 if completed, NaN if `Completion Year` itself missing. Preserves the off-plan signal instead of letting it collapse into the same NaN as a genuinely unknown Completion Year. | r=-0.0004, p=0.986 (n=1886, only 22 positive cases) — not significant, but the domain rationale (new-build premium) is sound; the sample is simply too small (22/3793) to detect it either way. Kept in df, but recommended for drop in 3.7 with this reasoning documented, not because the theory is wrong. |
| `Listed_Facility_Count` | `Facilities` | Count of comma-separated items after stripping and dropping purely numeric junk (Ad List 95706905 has a stray "10" — a scraping artefact, not a facility). | r=0.291, p=6.73×10⁻⁷⁵ (n=3793). Median price rises monotonically from RM260k (0 facilities) to RM460k (10+ facilities), a 77% increase. |
| `Has_Bus_Stop`, `Has_Mall`, `Has_Park`, `Has_School`, `Has_Hospital`, `Has_Highway`, `Has_Railway_Station` | `Bus Stop`/`Mall`/`Park`/`School`/`Hospital`/`Highway`/`Railway Station` (not the `Nearby X` variants — verified subset, see `notes_missing_value_decision_audit.md`) | `notna().astype(int)` per column. Missing rates 76%-96%; converting to presence flags removes the missing-value problem entirely (every row becomes 0 or 1, no NaN deferred further). | Individually tested: only `Has_Park` (r=-0.041, p=0.011) and `Has_School` (r=-0.045, p=0.006) are significant, and both are very weak. The other 5 are not significant (p>0.1). Kept for now — flagged as low-value; final keep/drop decision deferred to model-based feature selection. |

## Tried, tested, and dropped (not in df)

| Feature | Why it was tried | Why it was dropped |
|---|---|---|
| `Facilities_Recorded` | Presence flag paired with `Listed_Facility_Count`, mirroring `Property Age`/`Is_Off_Plan`, in case count==0 conflated "no facilities" with "not filled in". | Verified empirically: `Listed_Facility_Count == 0` for exactly the same 607 rows where `Facilities` is missing, zero exceptions. The two carry identical information in this dataset — pure redundancy. |
| `Nearby_Amenity_Count` | Sum of the 7 `Has_X` flags into a single 0-7 "how much nearby-amenity info was recorded" score. | Two reasons: (1) it's a pure linear combination of columns already in df — the same redundancy as `Total_Rooms = Bedroom + Bathroom` — so a linear model gains nothing from it; (2) empirically r=-0.019, p=0.242 (n=3793) — not significant, and this is a full-sample result, not an underpowered one. |
| `Units per Floor` | `Total Units / # of Floors`, as a building-density proxy (lower = more spacious/high-end). | Several rows produce physically impossible values (e.g. Ad List 100502825: Total Units=7810, # of Floors=5 → 1562 units on one floor). Neither source column was flagged as invalid by its own individual range check in 3.4 (`Total Units==1`) or 3.6 (IQR on other columns didn't cover these two) — each looks reasonable in isolation, so this cross-column ratio surfaced an upstream data-quality issue that univariate outlier checks can't catch. Left out until that Total Units / # of Floors issue is resolved with evidence (description cross-check or similar). |

## Also considered, not built

- `Total_Rooms = Bedroom + Bathroom`: r=0.475, but *lower* than `Bathroom` alone
  (r=0.585) — the sum dilutes Bathroom's stronger individual signal because
  Bedroom is a weaker predictor (r=0.295). Discussed but not added; if used,
  should be documented as low marginal value rather than presented as a clear win.
- `Size per Bedroom = Property Size / Bedroom`: r=0.240. Not yet built into the
  pipeline; div-by-zero/NaN guard still needed if implemented.
