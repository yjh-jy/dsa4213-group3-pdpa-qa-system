"""
merge_runs_json.py
- Strictly enforces max count (or manifest quotas) to avoid >500 items
- Repairs common citation variants and maps to corpus; drops items that cannot be grounded
- Guarantees non-empty corpus_links for every written QA
- Stable sequential IDs; optional append mode; optional validator run

Usage:
  python src/qa-gen/manual/merge_runs_json.py  \
    --corpus data/corpus/corpus_subsection_v1.jsonl \
    --inputs data/qa/prompts/runs/*.jsonl \
    --out data/qa/pdpa_qa_500.jsonl \
    --run-validator src/qa-gen/pdpa_dataset_validator.py \
    --max-count 500 \
    [--manifest data/qa/manifest.json]
    
"""

import argparse, json, re, sys, subprocess
from collections import defaultdict, Counter
from datetime import datetime


# --- Helpers -----------------------------------------------------------------

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

CANONICAL_RE = re.compile(r"^PDPA s\.[0-9A-Za-z]+(\([^)]+\))*$")

def normalize_citation(c: str) -> str:
    if not c: return ""
    c = str(c)
    # Strip smart-excerpt labels like " [A]"
    c = re.sub(r"\s*\[[A-Z]\]\s*$", "", c)
    c = re.sub(r"(?i)^\s*pdpa\s*", "", c)
    c = re.sub(r"(?i)^\s*s\.\s*", "", c).strip()
    c = re.sub(r"\s+", "", c)
    return f"PDPA s.{c}"

# Replace your old loader with this:
def build_corpus_index(corpus_path):
    import json, re
    from collections import defaultdict
    exact = defaultdict(list)   # "PDPA s.4(1)(0)" -> [{"chunk_id": "...", "part": "4"}]
    heads = set()               # {"PDPA s.4", "PDPA s.51", ...}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            obj = json.loads(line)
            canon = obj.get("canonical_citation") or obj.get("citation")
            chunk = obj.get("chunk_id")
            part  = str(obj.get("part","")).strip()
            if not (canon and chunk): 
                continue
            exact[canon].append({"chunk_id": chunk, "part": part})
            m = re.match(r"^(PDPA s\.[0-9A-Za-z]+)", canon)
            if m: heads.add(m.group(1))
    return exact, heads

def split_citation(canon: str):
    import re
    m = re.match(r"^(PDPA s\.[0-9A-Za-z]+)((\([^)]+\))*)$", canon)
    if not m: 
        return canon, []
    head, tail = m.group(1), (m.group(2) or "")
    subs = re.findall(r"\([^)]+\)", tail)
    return head, subs

def try_map_citation(canon: str, exact, heads):
    """Return (mapped_canon, links) with links ALWAYS a list of dicts [{"chunk_id","part"}]."""
    # 1) exact
    if canon in exact:
        return canon, exact[canon]

    head, subs = split_citation(canon)
    def join(head, subs): return head + "".join(subs)

    # 2) progressive reduction
    for k in range(len(subs)-1, -1, -1):
        reduced = join(head, subs[:k])
        if reduced in exact:
            return reduced, exact[reduced]

    # 3) (0) fallback at each depth
    for k in range(len(subs), 0, -1):
        subs0 = subs[:k-1] + ["(0)"]
        fallback = join(head, subs0)
        if fallback in exact:
            return fallback, exact[fallback]

    # 4) bare head
    if head in exact:
        return head, exact[head]

    # not found
    return None, []


def load_existing_ids(path):
    max_id = 0
    seen_questions = set()
    if not path: return max_id, seen_questions
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            m = re.search(r"PDPA-QA-(\d{4})", obj.get("id",""))
            if m: max_id = max(max_id, int(m.group(1)))
            q = norm_space(obj.get("question_user","")).lower()
            if q: seen_questions.add(q)
    return max_id, seen_questions

# --- Main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", default=None)
    ap.add_argument("--max-count", type=int, default=500, help="Stop after writing this many QAs.")
    ap.add_argument("--manifest", default=None, help="Optional: enforce per-Part quotas from manifest['quotas'].")
    ap.add_argument("--run-validator", default=None)
    args = ap.parse_args()

    exact, heads = build_corpus_index(args.corpus)

    # Optional per-Part quotas
    part_quotas = None
    if args.manifest:
        with open(args.manifest, "r", encoding="utf-8") as f:
            mfest = json.load(f)
        # Expect quotas keyed by part string
        q = mfest.get("quotas", {})
        # If you adopted the revised plan I suggested:
        # q = {"1":20,"2":20,"3":24,"4":48,"5":22,"6":39,"9":207,"10":120}
        part_quotas = {str(k): int(v) for k,v in q.items()}

    # Prepare output / IDs
    out_mode = "w"
    next_id = 1
    seen_questions = set()
    per_part_written = Counter()

    if args.append:
        out_mode = "a"
        max_id, seen_questions = load_existing_ids(args.append)
        next_id = max_id + 1

    def new_id():
        nonlocal next_id
        _id = f"PDPA-QA-{next_id:04d}"
        next_id += 1
        return _id

    written = 0
    skipped_unmappable = 0
    skipped_quota = 0
    skipped_dupe = 0

    out_f = open(args.out, out_mode, encoding="utf-8")
    paths = sorted(args.inputs, key=lambda x: int(x.split("data/qa/prompts/runs/")[1].split("_")[0].split("part")[1]))
    
    for path in paths:
        if written >= args.max_count: break
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        if not isinstance(data, list):
            raise ValueError(f"{path} is not a JSON array.")
        
        # Assembling the final 500 JSON in the required schema
        for item in data:
            if written >= args.max_count: break

            # Basic field hygiene
            q = norm_space(item.get("question_user") or "")
            if not q:
                continue
            key = q.lower()
            if key in seen_questions:
                skipped_dupe += 1
                continue
            seen_questions.add(key)

            qa_type = str(item.get("qa_type","pure-definitive"))
            gold_short = norm_space(item.get("gold_answer_short") or "")
            gold_ext   = norm_space(item.get("gold_answer_extended") or gold_short)

            # Normalized citations (strip labels like [A], upper/lower, whitespace)
            raw_cites = item.get("canonical_sections") or []
            cites = [normalize_citation(c) for c in raw_cites if c]
            # Extract from answer if missing
            if not cites:
                cites = re.findall(r"PDPA s\.[0-9A-Za-z]+(?:\([^)]+\))*", gold_short) or []
                cites = [normalize_citation(c) for c in cites]

            # Validate and map citations → corpus_links
            corpus_links = []
            parts_seen = set()

            clean_cites = []
            for c in cites:
                # # Fast sanity reject obvious garbage like s.12(0)(a)
                # if "(0)" in c:
                #     continue
                # Enforce canonical pattern
                if not CANONICAL_RE.match(c):
                    continue

                # Try mapping
                mapped, links = try_map_citation(c, exact, heads)
                if not links:
                    continue  # drop unmappable
                first = links[0]                                  # guaranteed dict now
                chunk_id = first["chunk_id"]
                part = first["part"]
                corpus_links.append({"doc_id":"PDPA","chunk_id":chunk_id})
                parts_seen.add(str(part))
                clean_cites.append(mapped)

            if not corpus_links:
                skipped_unmappable += 1
                continue  # DROP: cannot ground in corpus

            # Part assignment (first mapped part)
            part = next(iter(parts_seen)) if parts_seen else path.split("_")[0].split("part")[-1]

            # Enforce per-Part quotas (if provided)
            if part_quotas:
                cap = part_quotas.get(str(part))
                if cap is not None and per_part_written[str(part)] >= cap:
                    skipped_quota += 1
                    continue

            # Compose record
            record = {
                "id": new_id(),
                "part": str(part),
                "canonical_sections": clean_cites,
                "question_user": q,
                "question_variants": item.get("question_variants", []),
                "question_intent": item.get("question_intent", []),
                "question_language": "en-SG",
                "gold_answer_short": gold_short,
                "gold_answer_extended": gold_ext,
                "abstain_allowed": bool(qa_type in ["pure-abstain","scenario-ambiguous"]),
                "abstain_triggers": item.get("abstain_triggers", []),
                "abstain_gold_message": item.get("abstain_gold_message", ""),
                "ask_for_clarification_suggestions": item.get("ask_for_clarification_suggestions", []),
                "corpus_links": corpus_links,
                "retrieval_hints": [],
                "difficulty": "medium",
                "question_style": "consumer-plain",
                "expected_answer_style": "lawyer-plain",
                "version": "1.0.0",
                "last_reviewed_utc": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "scoring": {"answer_correctness":0.65,"citation_correctness":0.35,"abstention_policy":0.0,"min_passing":0.8},
                "evaluation_notes": "",
                "qa_type": qa_type
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            per_part_written[str(part)] += 1

            if written >= args.max_count:
                break

    out_f.close()

    # Summary
    print(f"[merge] Wrote {written} QA lines → {args.out}")
    if part_quotas:
        print("[merge] Per-Part counts:", dict(per_part_written))
    print(f"[merge] Skipped (unmappable citations): {skipped_unmappable}")
    print(f"[merge] Skipped (duplicates): {skipped_dupe}")
    if part_quotas:
        print(f"[merge] Skipped (quota exceeded): {skipped_quota}")

    # Optional validator
    if args.run_validator:
        cmd = [sys.executable, args.run_validator, "--dataset", args.out, "--corpus", args.corpus]
        print("[merge] Running validator:", " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            print("[merge] Validator reported issues (exit code != 0).")

if __name__ == "__main__":
    main()
