# Stage 3 near-duplicate re-listing groups — manual classification

Context: `src/data_preprocessing.py` Stage 3 detects rows that are identical on every
column except `Ad List` and `description` (23 groups, 48 rows, after Stage 2's
Ad List merge on the 3793-row dataset). The code currently retains all of them,
reasoning that different `Ad List` values mean independent listing events and the
dataset has no timestamp/unit-number field to prove otherwise.

Manually reading the `description` text for all 23 groups shows this blanket
justification is too weak — most groups are near-certainly the same ad
re-scraped/re-listed, while a handful have real textual evidence of being
different physical units. This file records that classification so the
investigation doesn't need to be redone later.

State-level impact if left un-deduped (measured on the 3793-row post-Stage-2
dataset): Johor 1.84%, Kuala Lumpur 0.64%, Penang 0.41%, Selangor 0.26%,
Sabah 0.68%, Sarawak 0.88% — all small, but will trend upward as later cleaning
steps (missing value / outlier handling) shrink the denominator further.
Re-check this ratio once cleaning is finished, before EDA.

## Group A — identical description text, same agent (14 groups)
Strong evidence: same real ad, re-scraped or reposted unchanged. Recommended
treatment when/if Stage 3 is changed to dedupe: keep one row per group.

| Ad List pair | Building | Agent |
|---|---|---|
| 102315606 / 102498879 | The Netizen @ Bandar Tun Hussein Onn | Kinki Chan |
| 101960423 / 102904007 | Selangor » Cyberjaya | Kydd |
| 102652094 / 102652195 | Bandar Sierra Apartment | Luke Lu / Stephennie Yong |
| 103784715 / 103784759 | Lily & Jasmine | Suki |
| 103446635 / 103470394 | Mewah View Luxurious Apartments | SK GOO |
| 103336579 / 103470201 | Sri Akasia | Daniel Ling |
| 94044194 / 103450483 | Pelita Indah Condominium | Bryan Ng (Ad List gap is large — likely reposted much later) |
| 103150184 / 103151143 | Seksyen 2 Wangsa Maju Flat | Choong |
| 103223072 / 103470205 | Park Avenue (Tampoi Indah) | Daniel Ling |
| 92021859 / 101898183 | Melody Homes | Max Lee (large Ad List gap) |
| 102920898 / 103178916 | Majestic Maxim | Kinki Chan |
| 102229930 / 102999145 | Alam Prima | Dylan |
| 103758389 / 103758407 / 103758671 | Seri Kembangan (bigsizecondo.wasap.my) | — (3-row group) |
| 103245254 / 103245262 | Middleton @ Minden Heights | Amos Sin |

## Group B — description differs by minor wording only, same agent (4 groups)
Likely still the same underlying listing with an edited/updated ad, but not a
byte-for-byte match, so an automated exact-string-equality rule would miss
these. Flag for manual override if implementing an automated dedup rule.

| Ad List pair | Building | Note |
|---|---|---|
| 101547589 / 102709083 | Desa Skudai Apartment | Second post adds "fully new renovation" line |
| 100249876 / 100288529 | Samajaya Apartment | Loan installment figure differs (RM1,190 vs RM1,200) |
| 103784825 / 103786756 | Taman Maluri (Ulu Tiram) | Second post adds "Can Full Loan" line |
| 103791934 / 103796162 / 103804090 | Raintree Residence | 3-row group; later posts add a "FREE MOT" line |

## Group C — real evidence of distinct physical units (5 groups)
Keep both/all rows — do not dedupe these.

| Ad List pair | Building | Evidence |
|---|---|---|
| 103159773 / 103259332 | The Rise Collection 3 | Ad text explicitly says "Different units available on hand"; renovation status differs (partial vs fully renovated) |
| 103780382 / 103780408 | BJ Court Condominium | Text disagrees on size (710 vs 700 sqft) and view ("Good View" vs "2nd Bridge View") |
| 103492971 / 103492972 | Anggun Puri | Consecutive Ad List IDs, same agent, pricing term differs (NETT vs NEGO) — likely two units listed back-to-back |
| 102988049 / 103665195 | Fairville | Floor differs ("Medium Floor" vs "Medium-High Floor") |
| 103803955 / 103803959 | LSH Sentul | Completely different ad text and named agent (Faez vs unnamed) — likely co-listing or separate units at new-project launch |

## If/when Stage 3 is changed to dedupe
- Groups A + B → collapse to one row per group (same rule already used in Stage 2:
  keep the more complete row, or since content is ~identical, `keep='first'` is fine).
- Group C → leave untouched, both/all rows stay.
- Regardless of this decision, use a **group-aware train/test split** (group by
  the Stage 3 `check_cols` match, i.e. these 23 groups) so no group's rows end up
  split across train and test — this avoids leakage independent of the dedup
  decision above.
