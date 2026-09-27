"""
Shared I/O helpers: loading source TSVs / candidate_pairs.tsv / ground truth,
and exploding the comma-separated ID-list columns into one-row-per-pair
DataFrames that features.build_feature_matrix expects.
"""

from __future__ import annotations

import csv
import sys
from typing import Dict, Iterator, Tuple

import pandas as pd

# Some candidate_pairs.tsv rows (a Source-1 entity with many candidates)
# can have a `candidate_entity_ids` field far longer than Python csv's
# default 131072-char field-size limit. Raise it as high as this platform
# allows (csv uses a C long internally, which can overflow on 32-bit
# builds / some Windows Python builds even at 64-bit sys.maxsize, hence
# the halving retry loop).
_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(_limit)
        break
    except OverflowError:
        _limit = int(_limit / 10)


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
    iter_id_list_tsv_rows below instead."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if id_col not in df.columns or list_col not in df.columns:
        raise ValueError(f"{path} must have columns [{id_col}, {list_col}], got {list(df.columns)}")
    return df


def iter_id_list_tsv_rows(path: str, id_col: str, list_col: str) -> Iterator[Tuple[str, str]]:
    """Streams a candidate_pairs.tsv-shaped file ONE ROW AT A TIME using
    Python's built-in csv module rather than pandas.read_csv.

    Why not pandas here: pandas' C parser buffers more internally than
    just "chunksize rows" — if any single row's candidate_entity_ids
    field is very long (an entity with a large candidate list), the C
    tokenizer can raise `pandas.errors.ParserError: ... out of memory`
    while building a chunk, even with a small chunksize. Reading with
    Python's csv module avoids that: memory use is bounded by one row's
    length at a time via normal buffered file I/O, with no hidden
    internal buffering beyond that. This is slower per-row than pandas'
    C engine, but it is the version that reliably doesn't crash on a
    memory-constrained machine with a multi-GB candidate file.

    Yields (source1_entity_id, candidate_entity_ids) tuples, in file order.
    """
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration:
            return
        try:
            id_idx = header.index(id_col)
            list_idx = header.index(list_col)
        except ValueError:
            raise ValueError(f"{path} must have columns [{id_col}, {list_col}], got {header}")

        for row in reader:
            if not row:
                continue
            s1_id = row[id_idx] if id_idx < len(row) else ""
            id_list = row[list_idx] if list_idx < len(row) else ""
            yield s1_id, id_list


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