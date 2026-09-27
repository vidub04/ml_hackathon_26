"""
Paste this into a notebook cell to sanity-check output/candidate_pairs.tsv
(the test-set candidate file) before handing it to Person 2. This checks the
same rules the official utils/validate_submission.py will later check for
matching_results.tsv + candidate_pairs.tsv together, but lets you catch
blocking-stage problems right now instead of waiting.
"""
import pandas as pd

test_s1_ids = set(pd.read_csv("data/processed/test_source1_clean.tsv", sep="\t",
                               dtype=str, keep_default_na=False)["entity_id"])
test_s2_ids = set(pd.read_csv("data/processed/test_source2_clean.tsv", sep="\t",
                               dtype=str, keep_default_na=False)["entity_id"])
test_s3_ids = set(pd.read_csv("data/processed/test_source3_clean.tsv", sep="\t",
                               dtype=str, keep_default_na=False)["entity_id"])
valid_candidate_ids = test_s2_ids | test_s3_ids

cand = pd.read_csv("output/candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False)

issues = []

# 1. Every test Source 1 entity appears exactly once
cand_ids_col = set(cand["source1_entity_id"])
missing = test_s1_ids - cand_ids_col
extra = cand_ids_col - test_s1_ids
dupes = cand["source1_entity_id"].duplicated().sum()
if missing:
    issues.append(f"{len(missing)} Source 1 test entities missing from candidate_pairs.tsv")
if extra:
    issues.append(f"{len(extra)} unexpected source1_entity_id values not in test set")
if dupes:
    issues.append(f"{dupes} duplicate source1_entity_id rows")

# 2. Every candidate ID referenced actually exists in the test set, and no
#    duplicates within a single row's list
bad_id_rows = 0
dup_id_rows = 0
total_candidates = 0
empty_rows = 0
for cand_str in cand["candidate_entity_ids"]:
    if not cand_str:
        empty_rows += 1
        continue
    ids = cand_str.split(",")
    total_candidates += len(ids)
    if len(ids) != len(set(ids)):
        dup_id_rows += 1
    if not set(ids).issubset(valid_candidate_ids):
        bad_id_rows += 1

if bad_id_rows:
    issues.append(f"{bad_id_rows} rows reference candidate IDs not present in test source2/3")
if dup_id_rows:
    issues.append(f"{dup_id_rows} rows contain duplicate IDs within their own list")

print(f"Rows in candidate_pairs.tsv: {len(cand):,}")
print(f"Empty candidate lists (blocking found nothing): {empty_rows:,}")
print(f"Avg candidates per entity: {total_candidates/len(cand):.1f}")
print()
if issues:
    print("ISSUES FOUND:")
    for i in issues:
        print(" -", i)
else:
    print("PASS: no structural issues found.")
