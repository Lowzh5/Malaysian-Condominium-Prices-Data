# Section 3.12 — Final Dataset Structure Summary Log

Context: `src/data_preprocessing.py` Section 3.12 produces no new data
processing — it's a pure accounting step. Every number in it is read from a
variable already computed earlier in the script (`RAW_COLUMNS`,
`cols_no_engineering`, `cols_replaced_by_engineering`, `X_train`, `X_test`,
`df`, `row_group_ids`, `state_dummies_train`, `property_type_dummies_train`,
`IMPUTE_COLS`), not re-typed by hand, so the summary can't silently drift out
of sync with what the pipeline actually did further up the script.

## Feature accounting

`RAW_COLUMNS` is snapshotted immediately after the raw CSV is loaded (before
any processing), so at the end of the script:

- **Raw features retained as-is** = `[c for c in RAW_COLUMNS if c in df.columns]`
- **Raw features dropped** = `[c for c in RAW_COLUMNS if c not in df.columns]`
  (should match `cols_no_engineering + cols_replaced_by_engineering`'s length)
- **Engineered features created** = `[c for c in df.columns if c not in RAW_COLUMNS]`

Result: 8 raw retained, 24 dropped, 28 engineered (8 + 24 = 32 original
columns; 28 + 8 = 36 final `df` columns — both check out against the earlier
section logs).

`Property Type` counts as "raw retained" here, not "dropped" — unlike every
other Section 3.7 Stage-B column, its encoding is fit-dependent and deferred
to Part 3 (post-split), so it's still sitting in `df` as plain raw text at
this point (see `notes_feature_selection_log.md`). `State` counts as
"engineered" (it's derived from `Address`, so it was never in `RAW_COLUMNS`
to begin with) even though it's likewise still unmerged/unencoded text in
`df` — "engineered" here just means "not literally the original raw column",
not "fully encoded".

## Missing values — verified, not assumed

`X_train.isna().sum().sum() + X_test.isna().sum().sum()` = 0, checked after
Part 3's imputation (in 3.11), not assumed because a `fillna()` line was
written earlier in the script. `df` itself (pre-split) still legitimately
carries real NaN in 7 columns (`Property Size`, `# of Floors`, `Total Units`,
`Parking Lot`, `State`, `Property Age`, `Floor_Range_Ordinal`) — this is
intentional, not a bug: `State`'s rare-merge/encoding and the other 6 columns'
median imputation all happen *after* the split (fit on `X_train` only), so
`df` itself is left honest for EDA purposes rather than silently filled with
a full-dataset statistic that would amount to a leakage shortcut.

## Duplicate rows — re-verified on the final df, not assumed clean from 3.2

3.2 de-duplicated the *original* 32-column dataset; that result doesn't
automatically carry forward once 24 columns are later dropped in 3.7. Re-
checked on the final df and found 67 groups of row-identical listings (136
rows) — see `notes_feature_selection_log.md` and `notes_train_test_split_log.md`
for the full investigation and the group-aware-split fix.

The summary table reports this distinction explicitly rather than a single
`df.duplicated().sum()` number, because that number alone would read as "an
unfinished cleanup step" when it's the opposite — deliberately retained,
real data:

| Item | Result | What it means |
|---|---|---|
| Row-identical groups retained (not deleted) | 67 | Evidence the rows were kept on purpose, per the 3.7 note - not a leftover bug |
| Row-identical groups split across train/test | 0 | Evidence the actual leakage risk was resolved (via `GroupShuffleSplit` in 3.11), computed by tagging `X_train`/`X_test` rows with their source and checking whether any row-identical group has both sources present |

## Reconciling 35 vs. 54 — where every added/removed column comes from

`df`'s pre-split feature count (35) doesn't match `X_train.shape[1]` (54).
Unlike a plain "5 missing-indicator flags" gap, this pipeline now has State/
Property Type's fit-dependent encoding sitting between the two counts as
well, all happening in Part 3, after the 3.11 split:

- **+12**: `State` one-hot columns (`state_dummies_train`, rare-merge + one-hot
  fit on `X_train`).
- **+4**: `Property Type` one-hot columns (`property_type_dummies_train`,
  same fit-on-`X_train` treatment).
- **−2**: raw `State`/`Property Type` text columns, dropped from
  `X_train`/`X_test` only (once their one-hot columns above exist) — `df`
  itself keeps both, so this is a subtraction that only applies past the
  split.
- **+5**: missing-value indicator flags (`Property_Age_Missing`,
  `Num_Floors_Missing`, `Total_Units_Missing`, `Parking_Lot_Missing`,
  `Property_Size_Missing`), one per column in `IMPUTE_COLS`.

35 + 12 + 4 − 2 + 5 = 54. Each of these four numbers is read from a variable
already computed earlier in Part 3 (`len(state_dummies_train.columns)`,
`len(property_type_dummies_train.columns)`, `len(IMPUTE_COLS)`), not
hardcoded, so the table can't go stale if a future edit changes the rare-
category thresholds or which columns get imputed.

## Final summary table (as printed by the script)

| Item | Result |
|---|---|
| Final number of rows (pre-split df) | 3793 |
| Final number of features (X, pre-split) | 35 |
| Numerical features | 34 |
| Non-numeric features remaining | 2 |
| Raw features retained as-is | 8 |
| Raw features dropped (Section 3.7) | 24 |
| Engineered features created (3.8+3.9 fixed rules) | 28 |
| State one-hot columns added (Part 3, fit on X_train) | 12 |
| Property Type one-hot columns added (Part 3, fit on X_train) | 4 |
| Raw State/Property Type columns dropped post-split (Part 3) | 2 |
| Missing-value indicator flags added (Part 3) | 5 |
| Final number of features in X_train/X_test | 54 |
| Training set shape | (3028, 54) |
| Testing set shape | (765, 54) |
| Remaining missing values (X_train + X_test) | 0 |
| Row-identical groups retained (not deleted - see 3.7 note) | 67 |
| Row-identical groups split across train/test (should be 0) | 0 |

("Non-numeric features remaining" = 2 is `State` and `Property Type` — the
only two `df` columns still sitting as raw text at this point, since their
encoding is deferred to Part 3, post-split.)
