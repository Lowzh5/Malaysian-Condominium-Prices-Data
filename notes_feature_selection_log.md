# Section 3.7 — Feature Selection Log

Context: `src/data_preprocessing.py` Section 3.7 runs *after* 3.8 (Feature
Engineering) and 3.9 (Categorical Encoding), not before, even though the
report presents it earlier. Several columns dropped here were deliberately
kept alive through 3.8/3.9 because those sections needed to extract from or
encode them first — dropping in report order (3.7 before 3.8) would destroy
the raw material before extraction could happen.

25 columns dropped in total, in two stages, each printed with non-null counts
before the drop (same "evidence before deletion" habit used since 3.1).

## Stage A — never going to be used, unrelated to any 3.8/3.9 engineering

| Column | Non-null (of 3793) | Reason |
|---|---|---|
| `description` | 3793 | Unstructured free text, not used by any engineered feature |
| `Ad List` | 3793 | Unique identifier, already used for deduplication in 3.2, no predictive value |
| `Nearby School` | 660 | Verified 100% subset of `School` (see `notes_missing_value_decision_audit.md`) |
| `Nearby Mall` | 348 | Verified 100% subset of `Mall` |
| `Nearby Railway Station` | 345 | Verified 100% subset of `Railway Station` |
| `Category` | 3793 | Constant column — single value across all rows, zero variance |
| `Firm Type` | 3098 | Listing-agent/firm metadata, not a property attribute |
| `Firm Number` | 3098 | Same as above — verified to always co-occur with Firm Type/REN Number (identical non-null count, 100% mutual overlap) |
| `REN Number` | 2804 | Same as above |
| `Building Name` | 3708 | High cardinality (1937 unique values), no extractable pattern, impractical for one-hot |
| `Developer` | 2148 | High cardinality (580 unique values), same reasoning |

## Stage B — raw columns already superseded by a 3.8/3.9 engineered feature

| Column | Non-null (of 3793) | Superseded by |
|---|---|---|
| `Address` | 3708 | `State` (3.8) |
| `State` | 3708 | Two-step chain: `Address` → `State` (3.8) → `State_Selangor`/`State_Penang`/etc. one-hot columns (3.9). `State` itself is superseded once the one-hot columns exist, the same way `Property Type`/`Land Title` are. |
| `Bus Stop` | 667 | `Has_Bus_Stop` (3.8) |
| `Mall` | 448 | `Has_Mall` (3.8) |
| `Park` | 762 | `Has_Park` (3.8) |
| `School` | 898 | `Has_School` (3.8) |
| `Hospital` | 325 | `Has_Hospital` (3.8) |
| `Highway` | 135 | `Has_Highway` (3.8) |
| `Railway Station` | 451 | `Has_Railway_Station` (3.8) |
| `Tenure Type` | 3793 | `Freehold Indicator` (3.9) |
| `Property Type` | 3793 | 4 one-hot columns (3.9, after rare-category merge) |
| `Land Title` | 3793 | `Is_Non_Bumi_Lot` (3.9, after rare-category merge) |
| `Floor Range` | 3793 | `Floor_Range_Ordinal` + `Floor_Range_Known` (3.9) |
| `Facilities` | 3186 | 14 `Has_<Facility>` multi-hot columns (3.9) + `Listed_Facility_Count` (3.8) |

## Net effect

Shape before drop: (3793, 77) → after drop: (3793, 52).

## Features engineered but explicitly recommended for exclusion (documented in `notes_feature_engineering_log.md`, not re-added here)

- `Is_Off_Plan` — sound domain rationale, but only 22/3793 positive cases;
  r=-0.0004, p=0.986. Kept in df; final in/out decision deferred to modelling.
- `Has_Park` / `Has_School` (of the 7 Has_X flags) — technically significant
  but r≈-0.04 (negligible); the other 5 Has_X flags are not significant at
  all. All 7 currently kept in df; flagged here as low-value candidates for
  a later model-based feature-importance pass, not dropped outright yet.
