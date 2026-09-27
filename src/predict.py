"""
Runs the trained matcher over the TEST candidate_pairs.tsv (produced by
the blocking stage) and writes the final output/matching_results.tsv,
plus a copy of the candidate set to output/candidate_pairs.tsv so the
submission package's output/ folder is self-contained.

Consumes the *_clean.tsv files from preprocess.py and the
output/candidate_pairs.tsv from blocking.py (this repo's own file layout).

Usage
-----
python3 src/predict.py \
    --source1 data/processed/test_source1_clean.tsv \
    --source2 data/processed/test_source2_clean.tsv \
    --source3 data/processed/test_source3_clean.tsv \
    --candidate-pairs output/candidate_pairs.tsv \
    --model models/matcher.pkl \
    --out-dir output \
    [--threshold 0.42]   # overrides the threshold stored in the model file
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_feature_matrix_clean  # noqa: E402
from io_utils import (  # noqa: E402
    load_clean_source,
    build_clean_lookup,
    iter_id_list_tsv_chunks,
    write_id_list_tsv,
)


def load_model(path: str):
    with open(path, "rb") as f:
        artifact = pickle.load(f)

    backend = artifact["backend"]
    model = artifact["model"]
    feat_names = artifact["feature_names"]

    if backend == "lightgbm":
        predict_fn = lambda X: model.predict(X, num_iteration=model.best_iteration)
    elif backend == "xgboost":
        import xgboost as xgb
        predict_fn = lambda X: model.predict(
            xgb.DMatrix(X, feature_names=feat_names),
            iteration_range=(0, getattr(model, "best_iteration", -1) + 1),
        )
    else:  # sklearn fallback
        predict_fn = lambda X: model.predict_proba(X)[:, 1]

    return predict_fn, artifact["threshold"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source1", required=True)
    ap.add_argument("--source2", required=True)
    ap.add_argument("--source3", required=True)
    ap.add_argument("--candidate-pairs", required=True,
                     help="candidate_pairs.tsv produced by the blocking stage for the TEST split")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--threshold", type=float, default=None,
                     help="override the threshold saved with the model")
    ap.add_argument("--max-matches-per-entity", type=int, default=None,
                     help="optional cap on matches kept per source1 entity "
                          "(highest score first) — leave unset for no cap")
    ap.add_argument("--chunksize", type=int, default=50_000,
                     help="rows (Source-1 entities) read from candidate_pairs.tsv per chunk. "
                          "Lower this if you hit a MemoryError. Unlike training, nothing here "
                          "is downsampled — every candidate is still scored, just not all at "
                          "once in memory.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[predict] loading sources...")
    s1_df = load_clean_source(args.source1)
    s2_df = load_clean_source(args.source2)
    s3_df = load_clean_source(args.source3)
    s1_lookup = build_clean_lookup(s1_df)
    s2s3_lookup = {**build_clean_lookup(s2_df), **build_clean_lookup(s3_df)}
    valid_s2s3_ids = set(s2s3_lookup.keys())

    predict_fn, saved_threshold = load_model(args.model)
    threshold = args.threshold if args.threshold is not None else saved_threshold
    print(f"[predict] using threshold={threshold:.3f}")

    # ------------------------------------------------------------------
    # Stream candidate_pairs.tsv in chunks (one row = one Source-1 entity's
    # full candidate list — never split across chunk boundaries), scoring
    # each chunk and keeping only entity_id -> matched ids, instead of
    # holding a scored DataFrame for the entire file in memory at once.
    # ------------------------------------------------------------------
    print(f"[predict] streaming {args.candidate_pairs} in chunks of {args.chunksize} rows...")
    matches_grouped = {}
    total_candidates = 0
    n_dropped_invalid = 0

    for chunk in iter_id_list_tsv_chunks(
        args.candidate_pairs, "source1_entity_id", "candidate_entity_ids", chunksize=args.chunksize
    ):
        s1_ids_out, cand_ids_out = [], []
        for s1_id, id_list in zip(chunk["source1_entity_id"], chunk["candidate_entity_ids"]):
            if not isinstance(id_list, str) or not id_list.strip():
                continue
            cand_ids = [c.strip() for c in id_list.split(",") if c.strip()]
            cand_ids = list(dict.fromkeys(cand_ids))
            for c in cand_ids:
                if c in valid_s2s3_ids:
                    s1_ids_out.append(s1_id)
                    cand_ids_out.append(c)
                else:
                    n_dropped_invalid += 1

        if not s1_ids_out:
            continue
        chunk_pairs = pd.DataFrame({"source1_entity_id": s1_ids_out, "entity_id": cand_ids_out})
        total_candidates += len(chunk_pairs)

        X, _ = build_feature_matrix_clean(chunk_pairs, s1_lookup, s2s3_lookup)
        chunk_pairs["score"] = predict_fn(X)

        matched = chunk_pairs[chunk_pairs["score"] >= threshold]
        if args.max_matches_per_entity:
            matched = matched.sort_values("score", ascending=False)
            matched = matched.groupby("source1_entity_id").head(args.max_matches_per_entity)

        for s1_id, group in matched.groupby("source1_entity_id")["entity_id"]:
            matches_grouped.setdefault(s1_id, []).extend(dict.fromkeys(group))

    if n_dropped_invalid:
        print(f"[predict] dropped {n_dropped_invalid} candidate ids not present in test source files")
    print(f"[predict] scored {total_candidates} candidate pairs")

    # every source1 test entity must appear exactly once, even with no matches
    all_s1_ids = s1_df["entity_id"].tolist()
    result_df = pd.DataFrame({
        "source1_entity_id": all_s1_ids,
        "matched_entity_ids": [matches_grouped.get(sid, []) for sid in all_s1_ids],
    })
    write_id_list_tsv(result_df, str(out_dir / "matching_results.tsv"),
                       "source1_entity_id", "matched_entity_ids")
    print(f"[predict] wrote {out_dir / 'matching_results.tsv'}  "
          f"({(result_df['matched_entity_ids'].apply(len) > 0).sum()} entities with >=1 match)")

    # keep the exact candidate set used for inference alongside the results,
    # so matching_results.tsv is guaranteed to be a subset of candidate_pairs.tsv
    # (a plain file copy — doesn't require loading the file into memory)
    dest_cand = out_dir / "candidate_pairs.tsv"
    src_cand = Path(args.candidate_pairs)
    if src_cand.resolve() != dest_cand.resolve():
        shutil.copyfile(src_cand, dest_cand)
    print(f"[predict] candidate set copied to {dest_cand}")


if __name__ == "__main__":
    main()