# Matching model — Business Entity Resolution Challenge

Slots directly into your existing pipeline
(`github.com/vidub04/ml_hackathon_26`): `preprocess.py` → `blocking.py` →
**this (the matching stage)** → `utils/validate_submission.py`.

```
src/
├── features.py      # LCS, Damerau-Levenshtein, Jaccard, n-gram similarity + feature builder
├── io_utils.py       # TSV loading / id-list exploding / writing helpers
├── train_model.py    # trains LightGBM (falls back to XGBoost, then sklearn HGB)
├── predict.py         # scores test candidate pairs -> matching_results.tsv
└── requirements.txt
```

## Install

```bash
pip install -r src/requirements.txt
```

## 1. Train

Run this after your own `preprocess.py` + `blocking.py` steps (as in
`cleaning_dataset.py`), pointing at their outputs:

```bash
python3 src/train_model.py \
    --source1 data/processed/train_source1_clean.tsv \
    --source2 data/processed/train_source2_clean.tsv \
    --source3 data/processed/train_source3_clean.tsv \
    --ground-truth student_resource/dataset/train/train_ground_truth.tsv \
    --candidate-pairs output/candidate_pairs_train_eval.tsv \
    --model-out models/matcher.pkl
```

This:
- labels every candidate pair 1/0 against `train_ground_truth.tsv`
- builds the 4-family similarity feature vector on top of your cleaned
  columns (`features.py` -> `build_feature_matrix_clean`)
- does a **group** train/val split on `source1_entity_id` (no leakage)
- trains a gradient-boosted classifier with early stopping
- sweeps the decision threshold on the validation fold to maximise
  macro-averaged **F_0.5** (the exact leaderboard metric)
- saves `{model, feature_names, threshold}` to `models/matcher.pkl`

## 2. Predict

```bash
python3 src/predict.py \
    --source1 data/processed/test_source1_clean.tsv \
    --source2 data/processed/test_source2_clean.tsv \
    --source3 data/processed/test_source3_clean.tsv \
    --candidate-pairs output/candidate_pairs.tsv \
    --model models/matcher.pkl \
    --out-dir output
```

Writes `output/matching_results.tsv` (every Source-1 test entity, empty
list for singletons, deduped IDs) and copies the candidate set to
`output/candidate_pairs.tsv` so `matching_results.tsv` is guaranteed to
be a subset of it. Then validate with the challenge's own script:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

I ran `preprocess.py` -> `blocking.py` -> this matcher end to end on a
synthetic mini dataset to confirm the interfaces line up exactly
(column names, `candidate_entity_ids` format, `matching_results.tsv`
being a strict subset of `candidate_pairs.tsv`, no dup/self-match IDs,
every Source-1 entity present) — all checks pass.

## Features (`features.py`)

`build_feature_matrix_clean` consumes the columns `preprocess.py` already
produces, rather than re-normalizing raw text:

| Field from preprocess.py | Used for |
|---|---|
| `business_name_core` (suffix-stripped, unidecode-transliterated) | name similarity |
| `business_name_blockkey` (sorted core tokens) | word-order-invariant name similarity |
| `business_address_clean` | address similarity |
| `postal_code` (US ZIP / India PIN) | exact structured match feature |
| `legal_suffix` (Ltd/Pvt/Inc/...) | exact structured match feature |
| `country` | categorical match flag (open-set safe — France just falls through) |

| Family | Function | Applied to |
|---|---|---|
| Longest Common Subsequence | `lcs_ratio` (normalised `2*LCS/(len(a)+len(b))`) | `business_name_core`, `business_address_clean` |
| Damerau-Levenshtein | `damerau_levenshtein_ratio` (handles adjacent-char transpositions/typos) | `business_name_core`, `business_name_blockkey`, `business_address_clean` |
| Jaccard | `jaccard_similarity` (token-set) | `business_name_core`, `business_address_clean` |
| N-gram | `ngram_similarity` (character 2-/3-gram Jaccard/"shingles") | `business_name_core`, `business_address_clean` |

`features.py` also ships a self-contained, from-scratch normalization
path (`build_feature_matrix` / `normalize_name` / `normalize_address`)
for use on raw `business_name` / `business_address` directly, in case
you ever want to run the matcher without the `preprocess.py` step —
`train_model.py` / `predict.py` default to the clean-file path above
since that's this repo's actual pipeline.

No external API, geocoder or database is used anywhere — everything is
derived purely from fields already in your `*_clean.tsv` files, in line
with the fair-play rules.

## Model / license note

LightGBM and XGBoost are both OSS under permissive licenses
(MIT-compatible / Apache-2.0). The trained booster (a few hundred
shallow trees) is orders of magnitude under the 8B-parameter cap in the
constraints.
