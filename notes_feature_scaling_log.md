# Section 3.10 — Feature Scaling Log

Context: `src/data_preprocessing.py` Section 3.10 runs after 3.11's split,
using only `X_train` to fit the scaler — no test-set information leaks into
the scaling statistics.

## Why StandardScaler, not MinMaxScaler

This dataset has genuine extreme values (e.g. `Total Units` up to several
thousand). Min-max scaling would let a single outlier compress the entire
rest of the distribution into a tiny range; `StandardScaler`'s mean/std are
far less distorted by a handful of extreme points.

## Which models need scaling, and why

Scaling matters for models that rely on distance calculations or gradient-
based optimisation, where a feature on a larger raw numerical scale would
dominate the result regardless of its actual importance:

- **KNN** — classifies/predicts by distance between points; an unscaled
  large-range feature would disproportionately dictate that distance.
- **SVR** — similarly relies on distance-based margins in its optimisation.
- **Linear Regression** — if fit via gradient descent, unscaled features can
  slow convergence. (Note: for plain OLS solved via the normal equations,
  scaling doesn't change predictions, R², or the model's actual fit at all —
  it only affects how easy the coefficients are to compare/interpret. "Biased
  coefficient magnitudes" is not the right way to describe this — *bias* is a
  specific statistical term for systematic estimation error, which this
  isn't.)

Tree-based models (**Decision Tree**, **Random Forest**, **Gradient
Boosting**) split via sequential thresholds ("is Property Size > 1000?"),
which are unaffected by a feature's absolute scale — a split point works
identically scaled or not. Scaling offers them no benefit and is skipped.

## What gets scaled

```
SCALE_COLS = ['Bedroom', 'Bathroom', 'Property Size', '# of Floors',
              'Total Units', 'Parking Lot', 'Property Age',
              'Listed_Facility_Count', 'Floor_Range_Ordinal']
```

`price` is the target, not a feature, so it's excluded entirely. One-hot
columns (`State_*`, `PropertyType_*`) and binary flags (`Has_*`,
`Is_Non_Bumi_Lot`, `Freehold Indicator`, `Floor_Range_Known`, the imputation
`*_Missing` flags) are left unscaled — already bounded 0/1, scaling a dummy
isn't meaningful.

`Floor_Range_Ordinal` **is** included, even though it only has 3 levels —
this was a live decision point, not obvious either way. It has a real
magnitude and order (1<2<3), not just presence/absence like the flags above,
so leaving it on a raw 1-3 scale while every other numeric feature is
standardised to mean≈0/std≈1 would let it disproportionately dominate or
shrink in a distance-based model purely from a unit mismatch, unrelated to
its actual importance. Decision: scale it alongside the genuinely continuous
columns.

## Fit on `X_train` only

```python
scaler = StandardScaler()
X_train[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])
X_test[SCALE_COLS] = scaler.transform(X_test[SCALE_COLS])
```

`fit_transform()` is only ever called on `X_train`. `X_test` only ever gets
`.transform()` — reusing `X_train`'s already-computed mean/std, never
computing its own. Verified: `X_train[SCALE_COLS]` post-scaling has mean≈0,
std≈1 (as expected for the data it was fit on); `X_test[SCALE_COLS]`'s
mean/std are close but *not* exactly 0/1 — that's the correct, expected
signature of "transformed with `X_train`'s scaler, not its own." If
`X_test`'s mean came out at exactly 0, that would be the warning sign that
leakage had occurred (test had been fit on its own statistics).

## Saved under separate filenames — doesn't overwrite 3.11's unscaled version

```
X_train_scaled.csv / X_test_scaled.csv / train_test_split_scaled.pkl / scaler.pkl
```

3.11 already saved `X_train.csv`/`X_test.csv` (imputed but unscaled) before
this section ran. Those are left untouched — tree-based models don't need
scaling and can use that version directly, without having to invert this
transform to recover the original values.
