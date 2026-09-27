from blocking import run_blocking
#%%
run_blocking(
    clean_dir="data/processed",
    split="train",
    ground_truth_path="student_resource/dataset/train/train_ground_truth.tsv",
    output_path="output/candidate_pairs_train_eval.tsv",
)