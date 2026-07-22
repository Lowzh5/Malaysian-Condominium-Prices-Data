# Section 3.9 — Categorical Encoding Log

Context: `src/data_preprocessing.py` Section 3.9 encodes every categorical
column into a model-ready numeric form. Runs after 3.8 (Feature Engineering)
and before 3.7 (Feature Selection) — the raw text columns encoded here are
only dropped once their encoded replacements exist in df.

**State and Property Type are NOT encoded in this section**, unlike Land
Title/Tenure Type/Floor Range/Facilities below. Both have a rare-category
merge threshold (State <10, Property Type <20 listings) that is a statistic
computed from data, so per the pipeline's leakage rule it must be fit on
`X_train` only, after the 3.11 split — not on the full pre-split `df` the
way this section operates. Their rare-merge + one-hot encoding is documented
in `notes_train_test_split_log.md` (Part 3), not here.

## Land Title — rare-category merge, then binary encoding

- Raw distribution: Non Bumi Lot (3179), Bumi Lot (607), Malay Reserved (7).
- Malay Reserved (<0.2%) merged into Bumi Lot rather than kept as its own
  one-hot column — both represent land with restricted (Bumiputera-only)
  purchase eligibility, unlike Non Bumi Lot's unrestricted eligibility, so
  this is a real conceptual grouping, not an arbitrary one.
- After the merge only 2 categories remain, so a full one-hot would just add
  a second column that's the exact logical negation of the first — encoded
  instead as a single flag, `Is_Non_Bumi_Lot` (1 = Non Bumi Lot, 3179 rows;
  0 = Bumi Lot/Malay Reserved combined, 614 rows).

## Tenure Type — binary encoding

- Only 2 categories, 0 missing: Freehold (2311), Leasehold (1482).
- Encoded as `Freehold Indicator` via `.map({'Freehold': 1, 'Leasehold': 0})`,
  not a `== 'Freehold'` comparison — `==` against NaN silently returns False,
  which would misclassify a genuinely unknown tenure as Leasehold/0 rather
  than preserving it as NaN if this column ever has missing values in a
  future data pull. Same one-fewer-redundant-column reasoning as Land Title.
- Evidence: median price Freehold RM370,000 vs Leasehold RM305,000 (~21%
  higher), r=0.125, p=1.24×10⁻¹⁴ — confirms the market intuition that
  freehold tenure commands a premium.

## Floor Range — ordinal + a separate "known" flag

- Distribution: Medium (1327), Unknown (1017, filled in back in 3.5), High
  (792), Low (657).
- `Floor_Range_Ordinal`: Low→1, Medium→2, High→3 (real order, not arbitrary —
  encoding this way preserves the monotonic relationship that one-hot would
  discard). `Unknown` isn't part of that order, so it maps to NaN here.
- `Floor_Range_Known`: 1 if the original value was Low/Medium/High, 0 if
  Unknown — keeps that distinction as its own visible signal (same pairing
  pattern as `Property Age`/`Is_Off_Plan`).
- **`Floor_Range_Ordinal`'s NaN is deliberately NOT filled in 3.9.** Filling
  it (with the train-set median) is deferred to 3.11, after the train/test
  split, using only `X_train`'s median — the same leakage-avoidance reason
  Completion Year / # of Floors / Total Units / Parking Lot's imputation is
  deferred there. 3.9's output is left honest (real NaN) rather than quietly
  contaminated by a full-dataset statistic.

## Facilities — text cleanup + multi-hot (MultiLabelBinarizer)

- Cleanup: split on comma, strip whitespace, title-case, drop purely numeric
  junk (same "10" artefact filtered in `Listed_Facility_Count`).
- "Merge near-duplicate wording" (e.g. "Badminton" vs "Badminton Court") was
  planned for but turned out to have nothing to merge: the full vocabulary
  across all 3793 rows was checked and is exactly 14 distinct, non-overlapping
  facility names (Parking, Security, Playground, Swimming Pool, Lift,
  Gymnasium, Minimart, Barbeque area, Jogging Track, Multipurpose hall, Sauna,
  Tennis Court, Club house, Squash Court) plus the one junk "10" — no
  synonyms, no case variants. That merge step is a documented no-op here, kept
  in the pipeline in case a future data pull introduces messier wording.
- `MultiLabelBinarizer` → 14 `Has_<Facility>` columns (e.g. `Has_Swimming_Pool`,
  `Has_Gymnasium`), explicitly cast to `int` (the binarizer already outputs
  int64, but this isn't relied on implicitly).

## Dtype check (Step 7 of the 3.9 plan)

Every encoding-produced column in this section was confirmed `int64`, not
`bool`: `facility_encoded` via explicit `.astype(int)`, `Freehold Indicator`
via `.map()` to an int dict, `Is_Non_Bumi_Lot`/`Floor_Range_Known` via
`.astype(int)`. `Floor_Range_Ordinal` is the one intentional exception
(float64, since it's an ordinal scale with legitimate pending NaN, not a 0/1
flag). `state_dummies_train`/`property_type_dummies_train` (Part 3, in
`notes_train_test_split_log.md`) follow the same `.astype(int)` convention.
