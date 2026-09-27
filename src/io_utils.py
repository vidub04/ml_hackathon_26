"""
Shared I/O helpers: loading source TSVs / candidate_pairs.tsv / ground truth,
and exploding the comma-separated ID-list columns into one-row-per-pair
DataFrames that features.build_feature_matrix expects.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


def load_source(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    expected = {"entity_id", "business_name", "business_address", "country"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    return df


def build_lookup(df: pd.DataFrame) -> Dict[str, Tuple[str, str, str]]:
    """entity_id -> (business_name, business_address, country)"""
    return {
        row.entity_id: (row.business_name, row.business_address, row.country)
        for row in df.itertuples(index=False)
    }


CLEAN_COLS = [
    "entity_id", "business_name_core", "business_name_blockkey",
    "business_address_clean", "postal_code", "legal_suffix", "country",
]


def load_clean_source(path: str) -> pd.DataFrame:
    """Loads a *_clean.tsv file produced by preprocess.py. Requires the
    normalized columns it adds (business_name_core, business_name_blockkey,
    business_address_clean, postal_code, legal_suffix) in addition to the
    raw ones."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = set(CLEAN_COLS) - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing expected preprocess.py output columns: {missing}. "
            f"Run preprocess.py on the raw source file first."
        )
    return df


def build_clean_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """entity_id -> dict of the CLEAN_COLS fields (minus entity_id itself)."""
    fields = [c for c in CLEAN_COLS if c != "entity_id"]
    return {
        row.entity_id: {f: getattr(row, f) for f in fields}
        for row in df.itertuples(index=False)
    }


def load_id_list_tsv(path: str, id_col: str, list_col: str) -> pd.DataFrame:
    """Loads a two-column TSV like ground_truth / candidate_pairs /
    matching_results (source1_entity_id, <comma-separated ids>).

    WARNING: reads the whole file into memory at once. Fine for
    ground_truth.tsv (small — one row per Source-1 entity, short lists),
    but candidate_pairs.tsv can be tens of GB at full dataset scale (see
    blocking.py's own docstring). For that file, use
    iter_id_list_tsv_chunks below instead."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if id_col not in df.columns or list_col not in df.columns:
        raise ValueError(f"{path} must have columns [{id_col}, {list_col}], got {list(df.columns)}")
    return df


def iter_id_list_tsv_chunks(path: str, id_col: str, list_col: str, chunksize: int = 50_000):
    """Streams a candidate_pairs.tsv-shaped file in chunks of `chunksize`
    ROWS (each row = one Source-1 entity's full candidate list), instead
    of loading the whole file into memory at once. This is what keeps
    train_model.py / predict.py working on a multi-GB candidate file:
    peak memory is bounded by chunksize, not by total file size, since
    every Source-1 entity's row is self-contained (one row is never split
    across chunk boundaries — chunking is row-wise)."""
    reader = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunksize)
    for chunk in reader:
        if id_col not in chunk.columns or list_col not in chunk.columns:
            raise ValueError(f"{path} must have columns [{id_col}, {list_col}], got {list(chunk.columns)}")
        yield chunk


def explode_id_list(df: pd.DataFrame, id_col: str, list_col: str) -> pd.DataFrame:
    """Turns a (source1_entity_id, 'S2-1,S2-2,S3-9') frame into one row
    per (source1_entity_id, entity_id) candidate pair. Rows with an empty
    list are dropped (they contribute no positive/candidate pairs)."""
    records = []
    for s1_id, id_list in zip(df[id_col], df[list_col]):
        if not isinstance(id_list, str) or not id_list.strip():
            continue
        for cand in id_list.split(","):
            cand = cand.strip()
            if cand:
                records.append((s1_id, cand))
    return pd.DataFrame(records, columns=["source1_entity_id", "entity_id"])


def write_id_list_tsv(df: pd.DataFrame, path: str, id_col: str, list_col: str) -> None:
    """Writes a (source1_entity_id -> [entity_ids]) mapping in the exact
    matching_results.tsv / candidate_pairs.tsv format: one row per source1
    entity, comma-joined ids, empty string for no matches, no index."""
    out = df.copy()
    out[list_col] = out[list_col].apply(
        lambda ids: ",".join(dict.fromkeys(ids)) if ids else ""  # dedup, keep order
    )
    out = out[[id_col, list_col]]
    out.to_csv(path, sep="\t", index=False)