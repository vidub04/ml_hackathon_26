from blocking import run_blocking
run_blocking(clean_dir="data/processed", split="test",
             output_path="output/candidate_pairs.tsv",
             max_candidates=300)
