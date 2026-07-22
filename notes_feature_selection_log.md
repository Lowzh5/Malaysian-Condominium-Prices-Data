# Section 3.7 — Feature Selection Log

Context: `src/data_preprocessing.py` Section 3.7 runs *after* 3.8 (Feature
Engineering) and 3.9 (Categorical Encoding), not before, even though the
report presents it earlier. Several columns dropped here were deliberately
kept alive through 3.8/3.9 because those sections needed to extract from or
encode them first — dropping in report order (3.7 before 3.8) would destroy
the raw material before extraction could happen.

26 columns dropped in total, in two stages, each printed with non-null counts
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
| `Completion Year` | 1886 | Verified `Property Age == REFERENCE_YEAR - Completion Year` exactly, zero exceptions (not approximate overlap like `Total_Rooms` — the same variable in different units). Kept `Property Age`, dropped this. |
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

Shape before drop: (3793, 76) → after drop: (3793, 50). (`Is_Off_Plan` was
already removed earlier, directly in 3.8 — see `notes_feature_engineering_log.md`
— which is why the starting count here is 76, not 77.)

## Post-drop duplicate re-check — found real train/test leakage, fixed without deleting rows

3.2 de-duplicated the original 32-column dataset (0 duplicates remained), but
that check is blind to duplicates that only become identical once
identifying columns are gone. Re-checked on the 50-column post-drop df:

- **67 groups of row-identical listings (136 rows total)** on the remaining
  50 columns, including `price` matching too (not a features-match-but-
  price-differs conflict, which would have been a separate, harder problem).
- Verified this wasn't cosmetic: with a plain random split, **20 of these 67
  groups landed split across `X_train`/`X_test`**, meaning 26 test rows were
  exact duplicates of a training row - silently inflating any evaluation
  metric computed on that split.

**First attempt (reverted): `df.drop_duplicates()`** — removed 69 rows.
Wrong call: unlike 3.2's exact duplicates (genuinely the same scrape captured
twice), these 136 rows are different real listings (different Ad List,
Address, description) that only look row-identical because the identifying
columns were just dropped. Deleting them throws away real observations that
did nothing wrong - the actual problem is narrower than "these rows exist."

**Fix that was kept**: a group ID is computed per row-identical group
(`df.groupby(list(df.columns)).ngroup()`, right after the Stage A/B drop, no
rows removed), then 3.11 splits with `GroupShuffleSplit` using that grouping
instead of a plain `train_test_split` - every row in a group lands on the
same side of the split, so a duplicate group can never straddle train/test,
without discarding any data. This is the same principle already recorded for
the Stage-3 near-duplicate relistings in `notes_near_duplicate_relistings.md`
("use a group-aware train/test split... this avoids leakage independent of
the dedup decision"). All 3793 rows are retained; confirmed zero duplicate
groups split across train/test after the fix, and actual test proportion is
0.201 (vs the 0.2 target - the group constraint only nudges 2-3 rows at a
time, negligible effect on the split ratio).

## Features engineered, currently still in df, flagged as low-value candidates

- `Has_Park` / `Has_School` (of the 7 Has_X flags) — technically significant
  but r≈-0.04 (negligible); the other 5 Has_X flags are not significant at
  all. All 7 currently kept in df; flagged here as candidates for a later
  model-based feature-importance pass, not dropped outright yet.
