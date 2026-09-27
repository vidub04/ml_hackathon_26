"""
Feature engineering for the Business Entity Resolution matching model.

Implements the 4 requested similarity families entirely from scratch
(pure Python / numpy — no external fuzzy-matching library required):

    1. Longest Common Subsequence (LCS)              -> address matching
    2. Damerau-Levenshtein edit distance              -> name matching
    3. Jaccard similarity (token-set based)            -> name + address
    4. Character n-gram similarity (Jaccard on shingles) -> name + address

Every candidate pair (source1_record, source2_or_3_record) is turned into
a fixed-length numeric feature vector that is fed to the XGBoost / LightGBM
matcher in train_model.py / predict.py.

NOTE ON FAIR-PLAY RULES: everything here runs purely on the fields already
present in the provided TSVs (business_name, business_address, country).
No external API, geocoder, or lookup table is used anywhere in this module,
in line with the challenge's "no external data lookup" constraint.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np

# --------------------------------------------------------------------------
# 1. Text normalisation
# --------------------------------------------------------------------------

# Common legal-suffix / abbreviation expansions seen in the noise patterns
# described in the problem statement (Corp/Corporation, Pvt/Private, ...).
# Keys and values are already lower-cased; applied with word boundaries.
_NAME_ABBREV = {
    "corp": "corporation",
    "inc": "incorporated",
    "ltd": "limited",
    "co": "company",
    "pvt": "private",
    "llc": "llc",
    "llp": "llp",
    "intl": "international",
    "mfg": "manufacturing",
    "assoc": "associates",
    "bros": "brothers",
    "&": "and",
}

_ADDR_ABBREV = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "ln": "lane",
    "dr": "drive",
    "apt": "apartment",
    "fl": "floor",
    "flr": "floor",
    "bldg": "building",
    "no": "number",
    "nr": "near",
    "opp": "opposite",
    "sec": "sector",
    "colony": "colony",
    "ngr": "nagar",
    "hwy": "highway",
    "ste": "suite",
    "sq": "square",
    "&": "and",
}

_PUNCT_RE = re.compile(r"[^\w\s]")
_MULTI_WS_RE = re.compile(r"\s+")


def _strip_accents(text: str) -> str:
    """Fold transliteration/accent variants (e.g. 'café' -> 'cafe')."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _apply_abbrev(tokens: List[str], mapping: Dict[str, str]) -> List[str]:
    return [mapping.get(tok, tok) for tok in tokens]


def _base_clean(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    if text.strip().lower() in ("nan", "none", ""):
        return ""
    text = _strip_accents(text)
    text = text.lower()
    text = text.replace("&", " and ")
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTI_WS_RE.sub(" ", text).strip()
    return text


@lru_cache(maxsize=200_000)
def normalize_name(text: str) -> str:
    """Normalise a business_name field: casefold, strip punctuation,
    expand common legal-suffix abbreviations, drop accents."""
    cleaned = _base_clean(text)
    if not cleaned:
        return ""
    tokens = cleaned.split(" ")
    tokens = _apply_abbrev(tokens, _NAME_ABBREV)
    return " ".join(tokens)


@lru_cache(maxsize=200_000)
def normalize_address(text: str) -> str:
    """Normalise a business_address field: casefold, strip punctuation,
    expand common street/unit abbreviations, drop accents."""
    cleaned = _base_clean(text)
    if not cleaned:
        return ""
    tokens = cleaned.split(" ")
    tokens = _apply_abbrev(tokens, _ADDR_ABBREV)
    return " ".join(tokens)


def normalize_country(text: str) -> str:
    if text is None:
        return ""
    t = str(text).strip().lower()
    aliases = {
        "us": "united states", "usa": "united states", "u.s.": "united states",
        "u.s.a.": "united states", "united states of america": "united states",
        "india": "india", "in": "india",
        "france": "france", "fr": "france",
    }
    return aliases.get(t, t)


# --------------------------------------------------------------------------
# 2. Core string-similarity primitives
# --------------------------------------------------------------------------

def lcs_length(a: str, b: str) -> int:
    """Classic O(n*m) longest common subsequence length, space-optimised
    to two rolling rows."""
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    if n < m:
        a, b, n, m = b, a, m, n
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = prev[j] if prev[j] >= curr[j - 1] else curr[j - 1]
        prev, curr = curr, prev
    return prev[m]


def lcs_ratio(a: str, b: str) -> float:
    """Normalised LCS similarity in [0, 1]: 2*LCS / (len(a) + len(b))."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    total = len(a) + len(b)
    return (2.0 * lcs_length(a, b)) / total if total else 0.0


def damerau_levenshtein_distance(a: str, b: str) -> int:
    """Optimal-string-alignment Damerau-Levenshtein distance (insert,
    delete, substitute, adjacent transposition)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la

    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j

    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,       # deletion
                d[i][j - 1] + 1,       # insertion
                d[i - 1][j - 1] + cost,  # substitution
            )
            if (
                i > 1 and j > 1
                and a[i - 1] == b[j - 2]
                and a[i - 2] == b[j - 1]
            ):
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)  # transposition
    return d[la][lb]


def damerau_levenshtein_ratio(a: str, b: str) -> float:
    """Normalised similarity in [0, 1]: 1 - dist / max(len(a), len(b))."""
    if not a and not b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    dist = damerau_levenshtein_distance(a, b)
    return max(0.0, 1.0 - dist / max_len)


def _tokenize(text: str) -> set:
    return set(text.split()) if text else set()


def jaccard_similarity(a: str, b: str) -> float:
    """Token-set Jaccard similarity: |A ∩ B| / |A ∪ B|."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def _char_ngrams(text: str, n: int = 3) -> set:
    """Character n-grams (shingles) over the whitespace-joined string,
    padded so short tokens (e.g. 2-letter words) still contribute."""
    if not text:
        return set()
    padded = f"  {text}  "
    if len(padded) < n:
        return {padded}
    return {padded[i:i + n] for i in range(len(padded) - n + 1)}


def ngram_similarity(a: str, b: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity."""
    ga, gb = _char_ngrams(a, n), _char_ngrams(b, n)
    if not ga and not gb:
        return 1.0
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    union = len(ga | gb)
    return inter / union if union else 0.0


def token_sort_ratio_dl(a: str, b: str) -> float:
    """Damerau-Levenshtein ratio after sorting tokens alphabetically —
    neutralises word-order transpositions (e.g. 'Star Bakery' vs
    'Bakery Star')."""
    sa = " ".join(sorted(a.split())) if a else ""
    sb = " ".join(sorted(b.split())) if b else ""
    return damerau_levenshtein_ratio(sa, sb)


# --------------------------------------------------------------------------
# 3. Feature vector assembly for a (source1, candidate) pair
# --------------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    # ---- name features ----
    "name_dl_ratio",
    "name_dl_ratio_sorted",
    "name_jaccard",
    "name_ngram3_sim",
    "name_ngram2_sim",
    "name_lcs_ratio",
    "name_exact_match",
    "name_len_diff_ratio",
    "name_first_token_match",
    # ---- address features ----
    "addr_lcs_ratio",
    "addr_dl_ratio",
    "addr_jaccard",
    "addr_ngram3_sim",
    "addr_len_diff_ratio",
    "addr_exact_match",
    "addr_digit_jaccard",
    # ---- country / misc ----
    "country_match",
    "country_both_known",
    "source_is_s3",
]


def _digits_only(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit())


def _len_diff_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    denom = max(la, lb, 1)
    return abs(la - lb) / denom


def build_pair_features(
    name1: str, addr1: str, country1: str,
    name2: str, addr2: str, country2: str,
    source_is_s3: bool,
) -> np.ndarray:
    """Build the fixed-length feature vector for one candidate pair.
    Inputs should be the RAW (un-normalised) fields; normalisation is
    applied internally and cached (lru_cache) so repeated records across
    many candidate pairs are only normalised once each.
    """
    n1, n2 = normalize_name(name1), normalize_name(name2)
    a1, a2 = normalize_address(addr1), normalize_address(addr2)
    c1, c2 = normalize_country(country1), normalize_country(country2)

    d1, d2 = _digits_only(a1), _digits_only(a2)

    feats = [
        damerau_levenshtein_ratio(n1, n2),
        token_sort_ratio_dl(n1, n2),
        jaccard_similarity(n1, n2),
        ngram_similarity(n1, n2, n=3),
        ngram_similarity(n1, n2, n=2),
        lcs_ratio(n1, n2),
        1.0 if n1 and n1 == n2 else 0.0,
        _len_diff_ratio(n1, n2),
        1.0 if (n1.split() and n2.split() and n1.split()[0] == n2.split()[0]) else 0.0,

        lcs_ratio(a1, a2),
        damerau_levenshtein_ratio(a1, a2),
        jaccard_similarity(a1, a2),
        ngram_similarity(a1, a2, n=3),
        _len_diff_ratio(a1, a2),
        1.0 if a1 and a1 == a2 else 0.0,
        jaccard_similarity(d1, d2) if (d1 or d2) else 0.0,

        1.0 if (c1 and c2 and c1 == c2) else 0.0,
        1.0 if (c1 and c2) else 0.0,
        1.0 if source_is_s3 else 0.0,
    ]
    return np.asarray(feats, dtype=np.float32)


# --------------------------------------------------------------------------
# 3b. Feature assembly on top of preprocess.py's *_clean.tsv output
# --------------------------------------------------------------------------
# preprocess.py already does normalization better than the generic path
# above for this dataset: unidecode transliteration of Devanagari names,
# postal-code extraction (US ZIP / India PIN), and legal-suffix splitting
# (business_name_core / business_name_blockkey / legal_suffix). Rather than
# re-normalizing raw text and losing that work, this builder consumes those
# columns directly and adds the two extra signals (postal_code, legal_suffix)
# that the generic path can't see.

CLEAN_FEATURE_NAMES: List[str] = [
    # ---- name features (on business_name_core: legal suffix already stripped) ----
    "name_dl_ratio",
    "name_jaccard",
    "name_ngram3_sim",
    "name_ngram2_sim",
    "name_lcs_ratio",
    "name_exact_match",
    "name_len_diff_ratio",
    # ---- blockkey features (sorted tokens: word-order invariant) ----
    "blockkey_dl_ratio",
    "blockkey_exact_match",
    # ---- address features (on business_address_clean) ----
    "addr_lcs_ratio",
    "addr_dl_ratio",
    "addr_jaccard",
    "addr_ngram3_sim",
    "addr_len_diff_ratio",
    "addr_exact_match",
    # ---- structured signals preprocess.py already extracted ----
    "postal_match",
    "postal_both_present",
    "legal_suffix_match",
    "legal_suffix_both_present",
    # ---- country / misc ----
    "country_match",
    "country_both_known",
    "source_is_s3",
]

CLEAN_LOOKUP_FIELDS = (
    "business_name_core", "business_name_blockkey", "business_address_clean",
    "postal_code", "legal_suffix", "country",
)


def build_pair_features_clean(
    name_core1: str, blockkey1: str, addr1: str, postal1: str, suffix1: str, country1: str,
    name_core2: str, blockkey2: str, addr2: str, postal2: str, suffix2: str, country2: str,
    source_is_s3: bool,
) -> np.ndarray:
    """Feature vector built directly from preprocess.py's already-cleaned
    columns — no re-normalization performed here."""
    n1, n2 = name_core1 or "", name_core2 or ""
    b1, b2 = blockkey1 or "", blockkey2 or ""
    a1, a2 = addr1 or "", addr2 or ""
    p1, p2 = postal1 or "", postal2 or ""
    s1, s2 = suffix1 or "", suffix2 or ""
    c1, c2 = country1 or "", country2 or ""

    feats = [
        damerau_levenshtein_ratio(n1, n2),
        jaccard_similarity(n1, n2),
        ngram_similarity(n1, n2, n=3),
        ngram_similarity(n1, n2, n=2),
        lcs_ratio(n1, n2),
        1.0 if n1 and n1 == n2 else 0.0,
        _len_diff_ratio(n1, n2),

        damerau_levenshtein_ratio(b1, b2),
        1.0 if b1 and b1 == b2 else 0.0,

        lcs_ratio(a1, a2),
        damerau_levenshtein_ratio(a1, a2),
        jaccard_similarity(a1, a2),
        ngram_similarity(a1, a2, n=3),
        _len_diff_ratio(a1, a2),
        1.0 if a1 and a1 == a2 else 0.0,

        1.0 if (p1 and p2 and p1 == p2) else 0.0,
        1.0 if (p1 and p2) else 0.0,
        1.0 if (s1 and s2 and s1 == s2) else 0.0,
        1.0 if (s1 and s2) else 0.0,

        1.0 if (c1 and c2 and c1 == c2) else 0.0,
        1.0 if (c1 and c2) else 0.0,
        1.0 if source_is_s3 else 0.0,
    ]
    return np.asarray(feats, dtype=np.float32)


def build_feature_matrix_clean(pairs_df, s1_lookup, s2s3_lookup) -> Tuple[np.ndarray, List[str]]:
    """s1_lookup / s2s3_lookup: entity_id -> dict with CLEAN_LOOKUP_FIELDS keys
    (see io_utils.build_clean_lookup)."""
    empty = {f: "" for f in CLEAN_LOOKUP_FIELDS}
    rows = []
    for s1_id, cand_id in zip(pairs_df["source1_entity_id"], pairs_df["entity_id"]):
        r1 = s1_lookup.get(s1_id, empty)
        r2 = s2s3_lookup.get(cand_id, empty)
        is_s3 = str(cand_id).startswith("S3-")
        rows.append(
            build_pair_features_clean(
                r1["business_name_core"], r1["business_name_blockkey"], r1["business_address_clean"],
                r1["postal_code"], r1["legal_suffix"], r1["country"],
                r2["business_name_core"], r2["business_name_blockkey"], r2["business_address_clean"],
                r2["postal_code"], r2["legal_suffix"], r2["country"],
                is_s3,
            )
        )
    X = np.vstack(rows) if rows else np.zeros((0, len(CLEAN_FEATURE_NAMES)), dtype=np.float32)
    return X, CLEAN_FEATURE_NAMES


def build_feature_matrix(pairs_df, s1_lookup, s2s3_lookup) -> Tuple[np.ndarray, List[str]]:
    """Vectorised helper used by train_model.py / predict.py.

    pairs_df: DataFrame with columns [source1_entity_id, entity_id]
              (one row per candidate pair — already exploded).
    s1_lookup / s2s3_lookup: dict entity_id -> (name, address, country)
    Returns (X, kept_pair_index) where kept_pair_index lines up 1:1 with
    the rows of pairs_df (rows with an unknown id are still included with
    a zero-ish "unknown" feature vector so nothing silently drops).
    """
    rows = []
    for s1_id, cand_id in zip(pairs_df["source1_entity_id"], pairs_df["entity_id"]):
        name1, addr1, country1 = s1_lookup.get(s1_id, ("", "", ""))
        name2, addr2, country2 = s2s3_lookup.get(cand_id, ("", "", ""))
        is_s3 = str(cand_id).startswith("S3-")
        rows.append(
            build_pair_features(name1, addr1, country1, name2, addr2, country2, is_s3)
        )
    X = np.vstack(rows) if rows else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    return X, FEATURE_NAMES