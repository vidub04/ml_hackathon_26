"""
preprocess.py
=============
Cleaning + normalization stage for the Business Entity Resolution challenge.

What this does, per source file (source1 / source2 / source3, train or test):
  1. Unicode-normalizes and transliterates non-Latin script names (e.g. Devanagari)
     to a romanized approximation, using `unidecode`. This matters a lot here:
     Source 1 is always Latin-script, but a large chunk of India records in
     Source 2/3 are written in Devanagari script, so raw string similarity would
     silently fail on those pairs without this step.
  2. Lowercases, strips punctuation, collapses whitespace.
  3. Normalizes legal-entity tokens in business names (Limited -> ltd,
     Private -> pvt, Incorporated -> inc, ...) and separates them out as a
     `legal_suffix` column, plus a `business_name_core` column (name with the
     legal suffix stripped) and `business_name_blockkey` (sorted core tokens,
     word-order invariant -- useful for blocking).
  4. Normalizes common address abbreviations (Street -> st, Road -> rd, ...).
  5. Extracts a postal code (US 5-digit ZIP / India 6-digit PIN) into its own
     column when present in the raw address, using the record's country to
     pick which pattern to look for.

Output: one cleaned TSV per input file, with the original columns preserved
plus the new normalized columns appended. Nothing is dropped, so any
downstream step can still fall back to the raw text.

Usage:
    python preprocess.py --input dataset/train/train_source1.tsv \
                          --output data/processed/train_source1_clean.tsv

    # or process a whole folder of source files at once:
    python preprocess.py --input-dir dataset/train --output-dir data/processed

Requires: pandas, unidecode  (pip install pandas unidecode)
"""

import argparse
import os
import re
import time
import unicodedata

import pandas as pd
from unidecode import unidecode as _unidecode

# ----------------------------------------------------------------------------
# Low-level text cleaning
# ----------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9\s]")

# Collapses dotted acronyms like "L.L.C." / "P.C." / "D.D.S." into "LLC" / "PC"
# / "DDS" *before* punctuation stripping. Without this, "L.L.C." tokenizes
# into three separate single-letter tokens ("l", "l", "c") instead of "llc",
# so the legal-suffix detector never recognizes it. Matches periods that sit
# directly between two letters with no space (so "Mr. Smith" is untouched,
# since there's a space after "Mr.").
_DOTTED_ACRONYM_RE = re.compile(r"(?<=[A-Za-z])\.(?=[A-Za-z])")


def to_ascii(text: str) -> str:
    """Unicode-normalize then transliterate non-Latin scripts to a romanized
    approximation (e.g. Devanagari business names -> rough Latin spelling)."""
    if not isinstance(text, str) or text == "":
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _DOTTED_ACRONYM_RE.sub("", text)
    return _unidecode(text)


def basic_clean(text: str) -> str:
    """Lowercase, ampersand -> 'and', strip punctuation, collapse whitespace."""
    text = to_ascii(text)
    text = text.lower()
    text = text.replace("&", " and ")
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


# ----------------------------------------------------------------------------
# Business name normalization
# ----------------------------------------------------------------------------

# Map noisy legal-entity tokens to a canonical short form.
NAME_TOKEN_SYNONYMS = {
    "limited": "ltd",
    "ltd": "ltd",
    "private": "pvt",
    "pvt": "pvt",
    "incorporated": "inc",
    "incorporation": "inc",
    "inc": "inc",
    "corporation": "corp",
    "corp": "corp",
    "company": "co",
    "co": "co",
    "llp": "llp",
    "llc": "llc",
    "pc": "pc",
    "pllc": "pllc",
    "lp": "lp",
    "enterprises": "ent",
    "enterprise": "ent",
    "and": "and",
}

# Tokens that, when they trail a name, are treated as a legal suffix and
# pulled out into their own column (a name can have more than one, e.g.
# "... Pvt Ltd").
LEGAL_SUFFIX_TOKENS = {"ltd", "pvt", "inc", "corp", "co", "llp", "llc", "pc", "pllc", "lp"}

# unidecode's transliteration of Devanagari is only a rough phonetic
# approximation -- "लिमिटेड" (limited) comes out as "limittedd", not
# "limited". Exact-match synonym lookup misses these, so fall back to
# prefix matching on a few long, safe stems (long enough that they won't
# accidentally match unrelated words).
FUZZY_SUFFIX_STEMS = (
    ("limit", "ltd"),
    ("praiv", "pvt"),   # transliterated "private"
    ("priv", "pvt"),
    ("corp", "corp"),
    ("compan", "co"),
    ("incorp", "inc"),
)


def _fuzzy_suffix_normalize(token: str) -> str:
    if token in NAME_TOKEN_SYNONYMS:
        return NAME_TOKEN_SYNONYMS[token]
    for stem, canonical in FUZZY_SUFFIX_STEMS:
        if token.startswith(stem):
            return canonical
    return token


def normalize_business_name(raw_name: str):
    """Returns (name_clean, name_core, name_blockkey, legal_suffix).

    name_clean     - fully cleaned name, synonyms normalized, suffix kept in place
    name_core      - name_clean with trailing legal-suffix tokens removed
    name_blockkey  - name_core tokens sorted alphabetically (word-order invariant)
    legal_suffix   - the legal-suffix tokens that were stripped, space-joined
    """
    cleaned = basic_clean(raw_name)
    tokens = cleaned.split()
    tokens = [_fuzzy_suffix_normalize(t) for t in tokens]

    core_tokens = list(tokens)
    suffix_tokens = []
    while core_tokens and core_tokens[-1] in LEGAL_SUFFIX_TOKENS:
        suffix_tokens.insert(0, core_tokens.pop())

    name_clean = " ".join(tokens)
    name_core = " ".join(core_tokens)
    name_blockkey = " ".join(sorted(core_tokens))
    legal_suffix = " ".join(suffix_tokens)

    return name_clean, name_core, name_blockkey, legal_suffix


# ----------------------------------------------------------------------------
# Address normalization
# ----------------------------------------------------------------------------

ADDRESS_TOKEN_SYNONYMS = {
    "street": "st", "st": "st",
    "road": "rd", "rd": "rd",
    "avenue": "ave", "ave": "ave",
    "boulevard": "blvd", "blvd": "blvd",
    "drive": "dr", "dr": "dr",
    "lane": "ln", "ln": "ln",
    "apartment": "apt", "apt": "apt",
    "number": "no", "no": "no",
    "floor": "fl", "fl": "fl", "flr": "fl",
    "building": "bldg", "bldg": "bldg",
    "highway": "hwy", "hwy": "hwy",
    "opposite": "opp", "opp": "opp",
    "near": "near",
    "post": "po", "po": "po",
    "box": "box",
    "suite": "ste", "ste": "ste",
    "unit": "unit",
}

US_ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
IN_PIN_RE = re.compile(r"\b\d{6}\b")


def normalize_address(raw_addr: str, country: str):
    """Returns (address_clean, postal_code)."""
    cleaned = basic_clean(raw_addr)
    tokens = cleaned.split()
    tokens = [ADDRESS_TOKEN_SYNONYMS.get(t, t) for t in tokens]
    address_clean = " ".join(tokens)

    raw = raw_addr or ""
    postal = ""
    if country == "US":
        m = US_ZIP_RE.search(raw)
        if m:
            postal = m.group().split("-")[0]
    elif country == "India":
        m = IN_PIN_RE.search(raw)
        if m:
            postal = m.group()
    else:
        # Unknown / open-set country (e.g. France in the test set): try both
        # patterns rather than assuming a format.
        m = US_ZIP_RE.search(raw) or IN_PIN_RE.search(raw)
        if m:
            postal = m.group().split("-")[0]

    return address_clean, postal


# ----------------------------------------------------------------------------
# Chunked file processing
# ----------------------------------------------------------------------------

REQUIRED_COLS = ["entity_id", "business_name", "business_address", "country"]


def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    names = chunk["business_name"].fillna("")
    addrs = chunk["business_address"].fillna("")
    countries = chunk["country"].fillna("")

    name_results = names.map(normalize_business_name)
    chunk["business_name_clean"] = [r[0] for r in name_results]
    chunk["business_name_core"] = [r[1] for r in name_results]
    chunk["business_name_blockkey"] = [r[2] for r in name_results]
    chunk["legal_suffix"] = [r[3] for r in name_results]

    addr_results = [normalize_address(a, c) for a, c in zip(addrs, countries)]
    chunk["business_address_clean"] = [r[0] for r in addr_results]
    chunk["postal_code"] = [r[1] for r in addr_results]

    return chunk


def process_file(input_path: str, output_path: str, chunksize: int = 200_000):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    reader = pd.read_csv(input_path, sep="\t", dtype=str, chunksize=chunksize)
    total_rows = 0
    start = time.time()
    first = True

    for i, chunk in enumerate(reader):
        missing = [c for c in REQUIRED_COLS if c not in chunk.columns]
        if missing:
            raise ValueError(f"{input_path}: missing expected columns {missing}")

        chunk = process_chunk(chunk)
        chunk.to_csv(
            output_path,
            sep="\t",
            index=False,
            mode="w" if first else "a",
            header=first,
        )
        first = False
        total_rows += len(chunk)
        elapsed = time.time() - start
        print(f"  [{os.path.basename(input_path)}] processed {total_rows:,} rows "
              f"({elapsed:.1f}s elapsed)")

    print(f"Done: {input_path} -> {output_path} ({total_rows:,} rows, "
          f"{time.time() - start:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Clean & normalize entity resolution source files.")
    parser.add_argument("--input", help="Single input TSV file")
    parser.add_argument("--output", help="Single output TSV file (paired with --input)")
    parser.add_argument("--input-dir", help="Folder containing *_source1/2/3.tsv files")
    parser.add_argument("--output-dir", help="Folder to write cleaned files into")
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()

    if args.input and args.output:
        process_file(args.input, args.output, args.chunksize)
        return

    if args.input_dir and args.output_dir:
        files = [f for f in os.listdir(args.input_dir) if f.endswith(".tsv") and "ground_truth" not in f]
        if not files:
            print(f"No .tsv source files found in {args.input_dir}")
            return
        for fname in sorted(files):
            in_path = os.path.join(args.input_dir, fname)
            out_name = fname.replace(".tsv", "_clean.tsv")
            out_path = os.path.join(args.output_dir, out_name)
            process_file(in_path, out_path, args.chunksize)
        return

    parser.error("Provide either (--input and --output) or (--input-dir and --output-dir)")


if __name__ == "__main__":
    main()
