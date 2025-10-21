"""
PDPA RAG Evaluation Harness
----------------------------------------
Usage:
  python pdpa_eval_harness.py --gold /path/to/pdpa_golden.jsonl --pred /path/to/preds.jsonl --out /path/to/report.csv

Predictions JSONL format (per line):
{
  "id": "PDPA-QA-0001",
  "answer_text": "string (model answer)",
  "citations": ["PDPA s.13", "PDPA s.14(1)"],
  "abstained": false,
  "abstain_reason": "optional string explaining why"
}

This harness scores:
- A (answer correctness): token F1 (lean & dependency-free)
- C (citation correctness): recall on required, with light penalty for extras
- B (abstention policy): according to gold abstention fields and qa_type
Then aggregates using the per-item weights in the gold 'scoring' field.

Note: This is a lightweight, dependency-free baseline (no embeddings). Feel free to swap A with your own semantic scorer.
"""
import argparse, json, re, csv, sys
from collections import defaultdict

CANONICAL_RE = re.compile(r"^PDPA s\.[0-9A-Za-z]+(\([^)]+\))*$")

def load_jsonl(path):
    items = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            obj=json.loads(line)
            items[obj['id']] = obj
    return items

def normalize_text(s):
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s):
    s = normalize_text(s)
    toks = re.findall(r"[a-z0-9]+", s)
    return toks

def token_f1(pred, gold):
    p = tokenize(pred)
    g = tokenize(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    from collections import Counter
    pc, gc = Counter(p), Counter(g)
    overlap = sum(min(pc[t], gc[t]) for t in pc)
    precision = overlap / max(1, sum(pc.values()))
    recall    = overlap / max(1, sum(gc.values()))
    if precision+recall == 0: return 0.0
    return 2*precision*recall/(precision+recall)

def norm_citation(c):
    c = c.strip()
    # force prefix, collapse spaces
    c = re.sub(r"(?i)\bpdpa\b\s*", "", c)
    c = re.sub(r"(?i)^s\.", "", c).strip()
    c = re.sub(r"\s+", "", c)
    return f"PDPA s.{c}"

def score_citations(required, model_list):
    # normalize
    R = set(norm_citation(x) for x in required)
    M = set(norm_citation(x) for x in model_list)
    if not R:
        # no required citations: neutral (full score if model doesn't hallucinate)
        return 1.0 if not M else 0.8
    rec = len(R & M) / len(R)
    extra = len(M - R)
    C = rec if extra == 0 else rec * 0.7
    return max(0.0, min(1.0, C))

def detect_abstain_text(s):
    s = normalize_text(s)
    flags = [
        "i don't have enough information",
        "insufficient information",
        "insufficient facts",
        "cannot answer confidently",
        "need more information",
        "need clarification"
    ]
    return any(f in s for f in flags)

def score_abstain(gold, pred):
    # Determine gold scenario
    qa_type = gold.get("qa_type", "pure-definitive")
    abstain_allowed = bool(gold.get("abstain_allowed", False))
    triggers = set(t.lower() for t in gold.get("abstain_triggers", []))
    # Model outcome
    abstained = bool(pred.get("abstained", False))
    reason = normalize_text(pred.get("abstain_reason", ""))
    mentioned_trigger = any(t for t in triggers if t and t in reason)
    # If no explicit abstained flag, try detect from answer text
    if not abstained:
        abstained = detect_abstain_text(pred.get("answer_text",""))
    # Scoring by case
    if qa_type == "pure-abstain" or (abstain_allowed and qa_type in {"scenario-ambiguous"}):
        if abstained and (mentioned_trigger or not triggers):
            return 1.0
        elif abstained:
            return 0.6
        else:
            # answered when abstention expected/allowed
            return 0.3 if qa_type=="scenario-ambiguous" else 0.0
    else:
        # expected to answer
        return 0.0 if abstained else 1.0

def main(args):
    gold = load_jsonl(args.gold)
    preds = load_jsonl(args.pred)
    # Prepare report
    rows = []
    missing = []
    for qid, g in gold.items():
        p = preds.get(qid)
        if not p:
            missing.append(qid)
            continue
        # A: token F1 vs gold short (basic baseline replacement for semantic similarity)
        A = token_f1(p.get("answer_text",""), g.get("gold_answer_short",""))
        # C: citation correctness
        C = score_citations(g.get("canonical_sections", []), p.get("citations", []))
        # B: abstention policy
        B = score_abstain(g, p)
        # weights
        sc = g.get("scoring", {"answer_correctness":0.65, "citation_correctness":0.35, "abstention_policy":0.0, "min_passing":0.8})
        wA, wC, wB = sc.get("answer_correctness",0.65), sc.get("citation_correctness",0.35), sc.get("abstention_policy",0.0)
        overall = max(0.0, min(1.0, wA*A + wC*C + wB*B))
        passed = overall >= sc.get("min_passing", 0.8)
        rows.append({
            "id": qid,
            "A_answer": f"{A:.3f}",
            "C_citation": f"{C:.3f}",
            "B_abstain": f"{B:.3f}",
            "overall": f"{overall:.3f}",
            "pass": int(passed)
        })
    # Write CSV
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id","A_answer","C_citation","B_abstain","overall","pass"])
        writer.writeheader()
        writer.writerows(rows)
    # STDERR summary
    sys.stderr.write(f"Evaluated {len(rows)} items; missing predictions for {len(missing)}.\n")
    if missing:
        sys.stderr.write("Missing IDs: " + ", ".join(missing[:10]) + ("..." if len(missing)>10 else "") + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
