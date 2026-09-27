from preprocess import process_file

process_file("student_resource/dataset/train/train_source1.tsv", "data/processed/train_source1_clean.tsv")
process_file("student_resource/dataset/train/train_source2.tsv", "data/processed/train_source2_clean.tsv")
process_file("student_resource/dataset/train/train_source3.tsv", "data/processed/train_source3_clean.tsv")

#%%
from blocking import run_blocking
#%%
run_blocking(
    clean_dir="data/processed",
    split="train",
    ground_truth_path="student_resource/dataset/train/train_ground_truth.tsv",
    output_path="output/candidate_pairs_train_eval.tsv",
)
#%%
from preprocess import process_file

process_file("student_resource/dataset/test/test_source1.tsv", "data/processed/test_source1_clean.tsv")
process_file("student_resource/dataset/test/test_source2.tsv", "data/processed/test_source2_clean.tsv")
process_file("student_resource/dataset/test/test_source3.tsv", "data/processed/test_source3_clean.tsv")
#%%
run_blocking(
    clean_dir="data/processed",
    split="test",
    output_path="output/candidate_pairs.tsv",
)
#%%
import os
size_gb = os.path.getsize("output/candidate_pairs.tsv") / (1024**3)
print(f"{size_gb:.2f} GB")
#%%

france_s1 = pd.read_csv("data/processed/test_source1_clean.tsv", sep="\t", dtype=str, keep_default_na=False)
france_s1 = france_s1[france_s1.country == "France"].sample(5, random_state=1)
cand = pd.read_csv("output/candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False)
merged = france_s1.merge(cand, left_on="entity_id", right_on="source1_entity_id")
for _, row in merged.iterrows():
    print(row["business_name"], "->", len(row["candidate_entity_ids"].split(",")) if row["candidate_entity_ids"] else 0, "candidates")
#%%
from blocking import run_blocking
#%%
run_blocking(clean_dir="data/processed", split="train",
              ground_truth_path="student_resource/dataset/train/train_ground_truth.tsv",
              output_path="output/candidate_pairs_train_eval.tsv",
              max_candidates=300)  # or whatever you land on

#%%
run_blocking(clean_dir="data/processed", split="test",
             output_path="output/candidate_pairs.tsv",
             max_candidates=300)
