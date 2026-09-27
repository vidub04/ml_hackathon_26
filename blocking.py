"""
blocking.py
===========
Candidate generation (blocking) stage for the Business Entity Resolution
challenge. Takes the *_clean.tsv files produced by preprocess.py and, for
every Source 1 entity, produces a shortlist of plausible Source 2 / Source 3
candidates -- without ever comparing all pairs (2.2M x 10.3M is ~22.7 billion
pairs, computationally impossible).

APPROACH: inverted-index blocking, not pairwise similarity search.
An inverted index maps a "key" (a name token, or a character n-gram) to the
list of entity_ids that contain it. To find candidates for a Source 1 entity,
you look up its own tokens/n-grams in the index and union whatever entity_ids
come back -- this only touches records that share *something*, so it scales
to millions of rows on ordinary RAM instead of requiring a full similarity
matrix (which would need far more RAM/compute than a laptop has for this
dataset size).

Three signals are combined (unioned) per entity, because relying on just one
loses recall on cases it happens to miss:
  1. Token blocking on `business_name_blockkey` (word-level, order-invariant)
  2. Character-trigram blocking on `business_name_core` (typo/fuzzy tolerant,
     and critical for the transliterated Devanagari names where token overlap
     is often weak -- e.g. "raam maarketting" vs "ram marketing" share no
     exact tokens but share plenty of overlapping trigrams)
  3. Exact postal_code match (rare in this data, but strong signal when present)

Very common tokens/n-grams (appearing in more than `max_df_ratio` of records)
are pruned from the index before lookup -- they blow up candidate-set size
for a given entity without adding real discriminating power (true matches
almost always share more than just one generic word).

Everything is partitioned by country first: a US Source 1 entity is only
ever compared against US Source 2/3 records, and processed one country at a
time so peak memory never covers the whole dataset at once.

Usage:
    # generate candidates for the training set and evaluate against ground truth
    python blocking.py --clean-dir data/processed --split train \
                        --ground-truth dataset/train/train_ground_truth.tsv \
                        --output output/candidate_pairs_train.tsv

    # generate candidates for the test set (no ground truth to evaluate against)
    python blocking.py --clean-dir data/processed --split test \
                        --output output/candidate_pairs.tsv
"""

import argparse
import os
import time
from collections import defaultdict

import pandas as pd

BLOCK_COLS = ["entity_id", "business_name_core", "business_name_blockkey", "business_address_clean", "postal_code", "country"]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_clean(path: str) -> pd.DataFrame:
    """Load only the columns blocking actually needs (skips the long raw
    text columns) to keep memory down."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, usecols=BLOCK_COLS)


# ----------------------------------------------------------------------------
# Index building
# ----------------------------------------------------------------------------

def char_ngrams(text: str, n: int = 3):
    text = text.replace(" ", "_")
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def build_index_streaming(entity_ids, texts, key_func, max_df_ratio: float = 0.01, hard_cap: int = 2000):
    """Builds an inverted index {key: [entity_id, ...]} in a single pass over
    (entity_id, text) pairs, WITHOUT ever materializing a full per-document
    list of key-sets for the whole corpus first. That intermediate list is
    what blows up memory at this scale (millions of short-lived string
    objects for n-grams/tokens held all at once) -- streaming avoids it.

    Posting lists longer than `max_df_ratio * n` OR the absolute `hard_cap`
    are dropped. The ratio alone isn't enough for character n-grams: with
    only 17,576 possible 3-letter combinations, common trigrams appear in a
    huge share of records regardless of corpus size, so an absolute cap is
    needed too -- otherwise a single query touches tens of thousands of
    postings and the whole pass becomes too slow to finish.
    """
    index = defaultdict(list)
    for eid, text in zip(entity_ids, texts):
        for k in key_func(text):
            index[k].append(eid)

    n = len(entity_ids)
    max_postings = min(max(50, int(n * max_df_ratio)), hard_cap)
    pruned = {k: v for k, v in index.items() if len(v) <= max_postings}
    return pruned


def build_postal_index(entity_ids, postal_codes):
    index = defaultdict(list)
    for eid, pc in zip(entity_ids, postal_codes):
        if pc:
            index[pc].append(eid)
    return index


# ----------------------------------------------------------------------------
# Candidate generation for one country partition (generator: yields one
# entity's candidates at a time so results are never all held in memory
# simultaneously -- important at this scale, since 800K+ entities each
# holding a candidate set can itself be gigabytes if accumulated in a dict)
# ----------------------------------------------------------------------------

def generate_candidates_for_country(s1_part: pd.DataFrame, corpus_part: pd.DataFrame,
                                     max_df_ratio: float = 0.01, min_ngram_overlap: int = 3,
                                     min_addr_overlap: int = 1, max_candidates: int = 300):
    """corpus_part is Source 2 + Source 3 records for this country, already
    concatenated. Yields (source1_entity_id, candidate_id_set) one at a time.

    min_ngram_overlap: a single shared trigram is common by chance (only
    17,576 possible 3-letter combinations exist, so with millions of records
    many trigrams are shared coincidentally). Requiring several shared
    trigrams before trusting the n-gram signal keeps candidate sets from
    exploding while still catching typo/transliteration-level similarity.

    max_candidates: hard cap on candidates per Source 1 entity, keeping only
    the highest-scoring ones when exceeded. Without this, a small number of
    entities with generic names/addresses (or an unlucky combination of
    common-but-not-pruned tokens) can have candidate sets in the thousands,
    which is what made the uncapped candidate_pairs.tsv balloon to 30+ GB
    and imply billions of pairs for the matching model to score.
    """
    from collections import Counter

    corpus_ids = corpus_part["entity_id"].tolist()

    token_index = build_index_streaming(
        corpus_ids, corpus_part["business_name_blockkey"].tolist(),
        lambda t: set(t.split()), max_df_ratio,
    )
    ngram_index = build_index_streaming(
        corpus_ids, corpus_part["business_name_core"].tolist(),
        char_ngrams, max_df_ratio,
    )
    postal_index = build_postal_index(corpus_ids, corpus_part["postal_code"].tolist())
    # Address token index: some true matches have completely unrelated
    # business names (e.g. a DBA / shell name change) but an identical or
    # near-identical address -- name-only blocking misses these entirely, so
    # address overlap is a first-class signal here, not a minor booster.
    addr_index = build_index_streaming(
        corpus_ids, corpus_part["business_address_clean"].tolist(),
        lambda t: set(t.split()), max_df_ratio,
    )
    del corpus_ids

    for s1_id, blockkey, core, postal, addr in zip(
        s1_part["entity_id"], s1_part["business_name_blockkey"],
        s1_part["business_name_core"], s1_part["postal_code"],
        s1_part["business_address_clean"],
    ):
        # Accumulate a relevance score per candidate instead of a plain set,
        # so that when an entity has more candidates than `max_candidates`
        # we can keep the *strongest* ones rather than an arbitrary cut.
        # Weights are deliberately simple (not tuned/learned) -- this is
        # still a blocking-stage triage, not the final classifier; the goal
        # is "don't throw away the true match when trimming", not precision.
        scores = Counter()

        token_hits = set()
        for tok in blockkey.split():
            token_hits.update(token_index.get(tok, ()))
        for cid in token_hits:
            scores[cid] += 3  # exact/near-exact name token match: strong signal

        ngram_counts = Counter()
        for ng in char_ngrams(core):
            for cid in ngram_index.get(ng, ()):
                ngram_counts[cid] += 1
        for cid, cnt in ngram_counts.items():
            if cnt >= min_ngram_overlap:
                scores[cid] += cnt * 0.2  # more shared trigrams = more confidence

        addr_counts = Counter()
        addr_tokens = addr.split()
        for tok in addr_tokens:
            for cid in addr_index.get(tok, ()):
                addr_counts[cid] += 1
        # Common tokens (city/state names, "road", "no", ...) are already
        # pruned out of addr_index by build_index_streaming, so whatever
        # tokens survive to be looked up here are already the specific,
        # discriminating ones (plot/society/locality names).
        for cid, cnt in addr_counts.items():
            if cnt >= min_addr_overlap:
                scores[cid] += cnt * 2  # specific shared address tokens: strong signal

        if postal:
            for cid in postal_index.get(postal, ()):
                scores[cid] += 10  # exact postal code match: very strong, rare enough to trust heavily

        if len(scores) > max_candidates:
            candidates = {cid for cid, _ in scores.most_common(max_candidates)}
        else:
            candidates = set(scores.keys())

        yield s1_id, candidates


# ----------------------------------------------------------------------------
# Streaming evaluation accumulator (avoids holding all candidates in memory
# at once -- stats are folded in country-partition by country-partition)
# ----------------------------------------------------------------------------

class EvalAccumulator:
    def __init__(self, ground_truth_path):
        gt = pd.read_csv(ground_truth_path, sep="\t", dtype=str, keep_default_na=False)
        self.true_matches = {
            s1_id: set(m.split(",")) if m else set()
            for s1_id, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"])
        }
        self.total_true = 0
        self.total_found = 0
        self.total_candidates = 0
        self.entities_full_recall = 0
        self.n_entities = 0

    def update_one(self, s1_id, cand_ids):
        true_ids = self.true_matches.get(s1_id, set())
        self.total_candidates += len(cand_ids)
        self.n_entities += 1
        if not true_ids:
            self.entities_full_recall += 1
            return
        found = true_ids & cand_ids
        self.total_true += len(true_ids)
        self.total_found += len(found)
        if found == true_ids:
            self.entities_full_recall += 1

    def report(self):
        pair_recall = self.total_found / self.total_true if self.total_true else float("nan")
        avg_candidates = self.total_candidates / self.n_entities if self.n_entities else 0
        print("=== Blocking evaluation ===")
        print(f"Source 1 entities evaluated:            {self.n_entities:,}")
        print(f"Pair-level recall (true matches found):  {pair_recall:.4f}  ({self.total_found:,}/{self.total_true:,})")
        print(f"Entities with ALL true matches found:    {self.entities_full_recall/self.n_entities:.4f}  ({self.entities_full_recall:,}/{self.n_entities:,})")
        print(f"Avg candidates per Source 1 entity:       {avg_candidates:.1f}")


# ----------------------------------------------------------------------------
# Callable entry point (use this directly from a notebook -- no argparse/CLI
# involved, just normal Python arguments)
# ----------------------------------------------------------------------------

def run_blocking(clean_dir: str, split: str, output_path: str, ground_truth_path: str = None,
                  max_df_ratio: float = 0.01, min_ngram_overlap: int = 3, min_addr_overlap: int = 1,
                  max_candidates: int = 300):
    """Does exactly what the CLI does, but as a plain function call -- import
    this in a notebook instead of shelling out to `python blocking.py ...`.

    split: "train" or "test"
    ground_truth_path: pass this (train split only) to also print recall /
        reduction-ratio evaluation stats as it runs.

    Example (paste into a notebook cell):
        from blocking import run_blocking
        run_blocking(
            clean_dir="data/processed",
            split="train",
            ground_truth_path="student_resource/dataset/train/train_ground_truth.tsv",
            output_path="output/candidate_pairs_train_eval.tsv",
        )
    """
    prefix = "train" if split == "train" else "test"
    s1_path = os.path.join(clean_dir, f"{prefix}_source1_clean.tsv")
    s2_path = os.path.join(clean_dir, f"{prefix}_source2_clean.tsv")
    s3_path = os.path.join(clean_dir, f"{prefix}_source3_clean.tsv")

    print("Loading cleaned files...")
    s1 = load_clean(s1_path)
    s2 = load_clean(s2_path)
    s3 = load_clean(s3_path)
    corpus = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    print(f"  source1: {len(s1):,} rows | source2+3 corpus: {len(corpus):,} rows")

    evaluator = EvalAccumulator(ground_truth_path) if (split == "train" and ground_truth_path) else None

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    countries = s1["country"].unique()
    t0 = time.time()

    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write("source1_entity_id\tcandidate_entity_ids\n")

        for country in countries:
            s1_part = s1[s1["country"] == country]
            corpus_part = corpus[corpus["country"] == country]
            print(f"[{country}] source1={len(s1_part):,}  corpus={len(corpus_part):,} "
                  f"({time.time()-t0:.1f}s elapsed)")

            for s1_id, cand_ids in generate_candidates_for_country(
                s1_part, corpus_part, max_df_ratio, min_ngram_overlap, min_addr_overlap, max_candidates
            ):
                out_f.write(f"{s1_id}\t{','.join(sorted(cand_ids))}\n")
                if evaluator is not None:
                    evaluator.update_one(s1_id, cand_ids)

    print(f"Candidate generation done in {time.time()-t0:.1f}s -> {output_path}")

    if evaluator is not None:
        evaluator.report()


# ----------------------------------------------------------------------------
# CLI entry point (thin wrapper around run_blocking, for shell/terminal use)
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate blocking candidate pairs.")
    parser.add_argument("--clean-dir", required=True, help="Folder with *_clean.tsv files from preprocess.py")
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--ground-truth", help="Path to train_ground_truth.tsv (train split only, for evaluation)")
    parser.add_argument("--output", required=True, help="Path to write candidate_pairs.tsv")
    parser.add_argument("--max-df-ratio", type=float, default=0.01,
                         help="Prune index keys appearing in more than this fraction of records")
    parser.add_argument("--min-ngram-overlap", type=int, default=3,
                         help="Min shared name character-trigrams required before trusting that signal alone")
    parser.add_argument("--min-addr-overlap", type=int, default=1,
                         help="Min shared (already-pruned, so already-specific) address tokens required. "
                              "1=max recall/more candidates, higher=fewer candidates/lower recall. See README notes.")
    parser.add_argument("--max-candidates", type=int, default=300,
                         help="Hard cap on candidates per Source 1 entity, keeping only the "
                              "highest-scoring ones when exceeded (controls output file size)")
    args = parser.parse_args()

    run_blocking(
        clean_dir=args.clean_dir, split=args.split, output_path=args.output,
        ground_truth_path=args.ground_truth, max_df_ratio=args.max_df_ratio,
        min_ngram_overlap=args.min_ngram_overlap, min_addr_overlap=args.min_addr_overlap,
        max_candidates=args.max_candidates,
    )


if __name__ == "__main__":
    main()
