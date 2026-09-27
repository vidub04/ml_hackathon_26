"""
Trains the pairwise entity-matching classifier on top of an already-built
candidate set (blocking/candidate-generation is assumed done upstream —
this script only consumes its output: a candidate_pairs.tsv for the
training split).

Pipeline
--------
1. Load train_source{1,2,3}.tsv and train_ground_truth.tsv.
2. Load the training candidate_pairs.tsv produced by the blocking stage.
3. Label every candidate pair 1/0 using the ground truth.
4. Build the LCS / Damerau-Levenshtein / Jaccard / n-gram feature vector
   for every candidate pair (features.py).
5. Group-split by source1_entity_id (never split a Source-1 entity's
   candidates across train/val — that would leak).
6. Train a LightGBM (falls back to XGBoost, then sklearn GBM) binary
   classifier with early stopping.
7. Sweep the decision threshold on the validation fold to maximise the
   macro-averaged F_0.5 exactly as the leaderboard computes it, and save
   it alongside the model.

Consumes the *_clean.tsv files produced by preprocess.py (business_name_core,
business_name_blockkey, business_address_clean, postal_code, legal_suffix)
and the candidate_pairs_train_eval.tsv produced by blocking.py, matching
this repo's own pipeline (cleaning_dataset.py) end to end.

Usage
-----
python3 src/train_model.py \
    --source1 data/processed/train_source1_clean.tsv \
    --source2 data/processed/train_source2_clean.tsv \
    --source3 data/processed/train_source3_clean.tsv \
    --ground-truth student_resource/dataset/train/train_ground_truth.tsv \
    --candidate-pairs output/candidate_pairs_train_eval.tsv \
    --model-out models/matcher.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_feature_matrix_clean, CLEAN_FEATURE_NAMES  # noqa: E402
from io_utils import (  # noqa: E402
    load_clean_source,
    build_clean_lookup,
    load_id_list_tsv,
    explode_id_list,
)


# --------------------------------------------------------------------------
# Model backend: prefer LightGBM, fall back to XGBoost, then sklearn.
# Both LightGBM and XGBoost ship under permissive OSS licenses and the
# trained booster stays well under the 8B-parameter constraint (a few
# hundred shallow trees), satisfying the challenge's model-license rule.
# --------------------------------------------------------------------------

def _fit_lightgbm(X_tr, y_tr, X_val, y_val):
    import lightgbm as lgb

    pos = max(y_tr.sum(), 1)
    neg = max(len(y_tr) - y_tr.sum(), 1)
    train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=CLEAN_FEATURE_NAMES)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set, feature_name=CLEAN_FEATURE_NAMES)
    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_data_in_leaf=20,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        scale_pos_weight=neg / pos,
        verbosity=-1,
        seed=42,
    )
    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    predict_fn = lambda X: model.predict(X, num_iteration=model.best_iteration)
    return model, predict_fn, "lightgbm"


def _fit_xgboost(X_tr, y_tr, X_val, y_val):
    import xgboost as xgb

    pos = max(y_tr.sum(), 1)
    neg = max(len(y_tr) - y_tr.sum(), 1)
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=CLEAN_FEATURE_NAMES)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=CLEAN_FEATURE_NAMES)
    params = dict(
        objective="binary:logistic",
        eval_metric="auc",
        eta=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=neg / pos,
        seed=42,
    )
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dval, "val")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )
    predict_fn = lambda X: model.predict(
        xgb.DMatrix(X, feature_names=CLEAN_FEATURE_NAMES), iteration_range=(0, model.best_iteration + 1)
    )
    return model, predict_fn, "xgboost"


def _fit_sklearn_fallback(X_tr, y_tr, X_val, y_val):
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.05, max_iter=500, random_state=42
    )
    model.fit(X_tr, y_tr)
    predict_fn = lambda X: model.predict_proba(X)[:, 1]
    return model, predict_fn, "sklearn_hgb"


def fit_model(X_tr, y_tr, X_val, y_val):
    for fitter in (_fit_lightgbm, _fit_xgboost, _fit_sklearn_fallback):
        try:
            return fitter(X_tr, y_tr, X_val, y_val)
        except ImportError:
            continue
    raise RuntimeError("No usable gradient-boosting backend found.")


# --------------------------------------------------------------------------
# F_0.5 macro-average scoring (mirrors the leaderboard formula exactly)
# --------------------------------------------------------------------------

def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def macro_f05(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> float:
    """pred_df / gt_df: columns [source1_entity_id, matched_entity_ids]
    (comma-joined string, '' for no match). Every source1 entity present
    in gt_df must be scored (missing predictions count as empty)."""
    pred_map = dict(zip(pred_df["source1_entity_id"], pred_df["matched_entity_ids"]))
    scores = []
    for s1_id, gt_ids_str in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
        gt_set = {x for x in str(gt_ids_str).split(",") if x} if gt_ids_str else set()
        pred_str = pred_map.get(s1_id, "")
        pred_set = {x for x in str(pred_str).split(",") if x} if pred_str else set()

        if not gt_set and not pred_set:
            scores.append(1.0)
            continue
        if not pred_set:
            scores.append(0.0)
            continue
        tp = len(gt_set & pred_set)
        precision = tp / len(pred_set)
        recall = tp / len(gt_set) if gt_set else 0.0
        scores.append(f_beta(precision, recall, beta=0.5))
    return float(np.mean(scores)) if scores else 0.0


def scores_to_matches(pairs_val: pd.DataFrame, probs: np.ndarray, threshold: float) -> pd.DataFrame:
    keep = pairs_val.assign(score=probs)
    keep = keep[keep["score"] >= threshold]
    grouped = (
        keep.groupby("source1_entity_id")["entity_id"]
        .apply(lambda ids: ",".join(dict.fromkeys(ids)))
        .reset_index()
        .rename(columns={"entity_id": "matched_entity_ids"})
    )
    return grouped


def tune_threshold(pairs_val: pd.DataFrame, probs: np.ndarray, gt_val: pd.DataFrame) -> float:
    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        pred = scores_to_matches(pairs_val, probs, t)
        score = macro_f05(pred, gt_val)
        if score > best_score:
            best_score, best_t = score, t
    print(f"[train] best threshold={best_t:.2f}  val macro-F0.5={best_score:.4f}")
    return best_t


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source1", required=True)
    ap.add_argument("--source2", required=True)
    ap.add_argument("--source3", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--candidate-pairs", required=True,
                     help="candidate_pairs.tsv produced by the blocking stage for the TRAIN split")
    ap.add_argument("--model-out", default="models/matcher.pkl")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)

    print("[train] loading sources...")
    s1_df = load_clean_source(args.source1)
    s2_df = load_clean_source(args.source2)
    s3_df = load_clean_source(args.source3)
    s1_lookup = build_clean_lookup(s1_df)
    s2s3_lookup = {**build_clean_lookup(s2_df), **build_clean_lookup(s3_df)}

    gt_df = load_id_list_tsv(args.ground_truth, "source1_entity_id", "matched_entity_ids")
    cand_df = load_id_list_tsv(args.candidate_pairs, "source1_entity_id", "candidate_entity_ids")

    # every source1 entity must be represented, even singletons w/ no candidates
    all_s1_ids = s1_df["entity_id"].tolist()
    gt_full = pd.DataFrame({"source1_entity_id": all_s1_ids}).merge(
        gt_df, on="source1_entity_id", how="left"
    )
    gt_full["matched_entity_ids"] = gt_full["matched_entity_ids"].fillna("")

    gt_pairs = explode_id_list(gt_df, "source1_entity_id", "matched_entity_ids")
    gt_pairs["label"] = 1
    positive_set = set(zip(gt_pairs["source1_entity_id"], gt_pairs["entity_id"]))

    cand_pairs = explode_id_list(cand_df, "source1_entity_id", "candidate_entity_ids")
    cand_pairs = cand_pairs.drop_duplicates(subset=["source1_entity_id", "entity_id"]).reset_index(drop=True)
    cand_pairs["label"] = cand_pairs.apply(
        lambda r: 1 if (r["source1_entity_id"], r["entity_id"]) in positive_set else 0, axis=1
    )

    n_missed = len(positive_set - set(zip(cand_pairs["source1_entity_id"], cand_pairs["entity_id"])))
    if n_missed:
        print(f"[train] WARNING: {n_missed} ground-truth pairs are not present in "
              f"candidate_pairs.tsv (blocking recall ceiling < 1.0). These cannot "
              f"be recovered by the matcher — revisit blocking if this is large.")

    print(f"[train] {len(cand_pairs)} candidate pairs, "
          f"{cand_pairs['label'].sum()} positive / {(cand_pairs['label']==0).sum()} negative")

    print("[train] building features...")
    X, feat_names = build_feature_matrix_clean(cand_pairs, s1_lookup, s2s3_lookup)
    y = cand_pairs["label"].to_numpy()

    # group split on source1_entity_id so a whole entity's candidates stay together
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(X, y, groups=cand_pairs["source1_entity_id"]))

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    pairs_val = cand_pairs.iloc[val_idx].reset_index(drop=True)

    print(f"[train] train={len(X_tr)} val={len(X_val)}")
    model, predict_fn, backend = fit_model(X_tr, y_tr, X_val, y_val)
    print(f"[train] fitted backend: {backend}")

    val_probs = predict_fn(X_val)

    val_s1_ids = pairs_val["source1_entity_id"].unique().tolist()
    gt_val = gt_full[gt_full["source1_entity_id"].isin(val_s1_ids)].reset_index(drop=True)

    threshold = tune_threshold(pairs_val, val_probs, gt_val)

    baseline_pred = scores_to_matches(pairs_val, val_probs, 0.5)
    print(f"[train] macro-F0.5 @0.50 = {macro_f05(baseline_pred, gt_val):.4f} "
          f"(for reference vs tuned threshold above)")

    artifact = {
        "backend": backend,
        "model": model,
        "feature_names": feat_names,
        "threshold": float(threshold),
    }
    with open(args.model_out, "wb") as f:
        pickle.dump(artifact, f)
    print(f"[train] saved model + threshold -> {args.model_out}")


if __name__ == "__main__":
    main()
