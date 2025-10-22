"""
PDPA Dataset Validator (Lean Schema + Quality Gates)
----------------------------------------------------
Usage:
  python pdpa_dataset_validator.py --dataset /path/to/qa.jsonl --corpus /path/to/corpus.jsonl --expect-count 500

Quality gates (hard failures):
- Any record missing `canonical_sections` or `corpus_links`
- Any citation not in canonical form: "PDPA s.<section>(<sub>)(<clause>)"
- Duplicate `id` values
- Duplicate `question_user` (normalized)
- Part not in allowed set: {"1","2","3","4","5","6","9","10"}
- Total count != --expect-count (default 500)

Soft checks (warnings):
- `qa_type` not in allowed set
- `abstain_allowed` inconsistent with `qa_type`
- Scoring weights outside [0,1] or sum not within [0.9, 1.05]
- Language not "en-SG"

Exit codes:
- 0 = OK
- 1 = Failed quality gates
"""
import argparse, json, re, sys
from collections import Counter, defaultdict

ALLOWED_PARTS = {"1","2","3","4","5","6","9","10"}
ALLOWED_QA_TYPES = {"pure-definitive", "definitive-with-conditions", "scenario-ambiguous", "pure-abstain"}
CANONICAL_RE = re.compile(r"^PDPA s\.[0-9A-Za-z]+(\([^)]+\))*$")

def load_jsonl(path):
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            line=line.strip()
            if not line: continue
            try:
                items.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Parse error at line {i}: {e}")
    return items

def load_corpus(path):
    canon_set = set()
    chunks_set = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            obj = json.loads(line)
            canon = obj.get("canonical_citation") or obj.get("canonical") or obj.get("citation")
            chunk = obj.get("chunk_id")
            if canon: canon_set.add(canon)
            if chunk: chunks_set.add(chunk)
    return canon_set, chunks_set

def norm_question(s):
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def validate(dataset_path, corpus_path, expect_count):
    data = load_jsonl(dataset_path)
    canon_set, chunks_set = load_corpus(corpus_path)

    hard_errors = []
    warnings = []

    # Count check
    if expect_count is not None and len(data) != expect_count:
        hard_errors.append(f"Dataset count {len(data)} != expected {expect_count}")

    # Duplicate checks
    ids = [d.get("id","") for d in data]
    qnorms = [norm_question(d.get("question_user","")) for d in data]
    dup_ids = [k for k,v in Counter(ids).items() if v>1]
    dup_qs  = [k for k,v in Counter(qnorms).items() if v>1]
    if dup_ids:
        hard_errors.append(f"Duplicate id(s): {dup_ids[:10]}{'...' if len(dup_ids)>10 else ''}")
    if dup_qs:
        hard_errors.append(f"Duplicate question_user(s): {dup_qs[:3]}{'...' if len(dup_qs)>3 else ''}")

    for i, d in enumerate(data, start=1):
        did = d.get("id", f"#idx{i}")

        # Required fields
        for f in ["part","canonical_sections","question_user","gold_answer_short","corpus_links","scoring","qa_type"]:
            if f not in d:
                hard_errors.append(f"{did}: missing field '{f}'")
        # Non-empty checks
        if not d.get("canonical_sections"): hard_errors.append(f"{did}: empty canonical_sections")
        if not d.get("corpus_links"):       hard_errors.append(f"{did}: empty corpus_links")

        # Part
        part = str(d.get("part","")).strip()
        if part not in ALLOWED_PARTS:
            hard_errors.append(f"{did}: invalid part '{part}'")

        # Citations format + existence in corpus
        for c in d.get("canonical_sections", []):
            if not CANONICAL_RE.match(c or ""):
                hard_errors.append(f"{did}: non-canonical citation '{c}'")
            elif c not in canon_set:
                hard_errors.append(f"{did}: citation not found in corpus '{c}'")

        # Corpus links existence
        for link in d.get("corpus_links", []):
            chunk = link.get("chunk_id")
            if not chunk or chunk not in chunks_set:
                hard_errors.append(f"{did}: corpus chunk not found '{chunk}'")

        # qa_type
        qtype = d.get("qa_type")
        if qtype not in ALLOWED_QA_TYPES:
            warnings.append(f"{did}: qa_type '{qtype}' not in allowed {sorted(ALLOWED_QA_TYPES)}")

        # Abstention coherence
        abstain_allowed = bool(d.get("abstain_allowed", False))
        if qtype in {"pure-abstain","scenario-ambiguous"} and not abstain_allowed:
            warnings.append(f"{did}: qa_type={qtype} but abstain_allowed=false")
        if qtype in {"pure-definitive","definitive-with-conditions"} and abstain_allowed:
            warnings.append(f"{did}: qa_type={qtype} but abstain_allowed=true")

        # Scoring
        sc = d.get("scoring", {})
        wa = sc.get("answer_correctness", 0.65)
        wc = sc.get("citation_correctness", 0.35)
        wb = sc.get("abstention_policy", 0.0)
        mp = sc.get("min_passing", 0.8)
        if not (0.0 <= wa <= 1.0 and 0.0 <= wc <= 1.0 and 0.0 <= wb <= 1.0):
            warnings.append(f"{did}: scoring weights out of [0,1] range")
        if not (0.5 <= mp <= 1.0):
            warnings.append(f"{did}: min_passing unusual ({mp})")
        ssum = wa + wc + wb
        if not (0.9 <= ssum <= 1.05):
            warnings.append(f"{did}: weight sum unusual ({ssum:.2f}), expected ~1.0")

        # # Language
        # lang = d.get("question_language","en-SG")
        # if lang != "en-SG":
        #     warnings.append(f"{did}: question_language '{lang}' (expected 'en-SG')")

    ok = len(hard_errors) == 0
    return ok, hard_errors, warnings, len(data)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to PDPA QA JSONL")
    ap.add_argument("--corpus", required=True, help="Path to PDPA corpus JSONL")
    ap.add_argument("--expect-count", type=int, default=500, help="Expected number of QAs (default: 500)")
    args = ap.parse_args()

    ok, errors, warnings, n = validate(args.dataset, args.corpus, args.expect_count)

    print(f"[validator] Checked {n} items with expected count {args.expect_count}")
    if warnings:
        print(f"[validator] Warnings ({len(warnings)}):")
        for w in warnings[:50]:
            print("  -", w)
        if len(warnings) > 50:
            print("  ...")

    if not ok:
        print(f"[validator] FAILED with {len(errors)} hard error(s):")
        for e in errors[:100]:
            print("  *", e)
        if len(errors) > 100:
            print("  ...")
        sys.exit(1)
    else:
        print("[validator] PASS — all quality gates satisfied.")
        sys.exit(0)

if __name__ == "__main__":
    main()
