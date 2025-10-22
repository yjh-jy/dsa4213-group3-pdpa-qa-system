#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage:

python src/data-related/sample_chunks.py \
  --corpus data/corpus/corpus_subsection_v1.jsonl \
  --manifest data/qa/manifest.json \
  --out data/qa/selected_chunks.json
"""

import argparse, json, hashlib, re
from collections import defaultdict

def load_corpus(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            items.append(json.loads(line))
    return items

def stable_score(seed:int, key:str) -> int:
    h = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(h[:16], 16)

def family_of(canonical: str) -> str:
    if not canonical: return "UNK"
    m = re.match(r"^(PDPA s\.[0-9A-Za-z]+)", canonical)
    return m.group(1) if m else canonical

def stratified_pick(candidates, quota, seed):
    buckets = defaultdict(list)
    for c in candidates:
        fam = family_of(c.get("canonical_citation") or "")
        c["_fam"] = fam
        c["_score"] = stable_score(seed, c.get("chunk_id",""))
        buckets[fam].append(c)
    for fam in buckets:
        buckets[fam].sort(key=lambda x: (x["_score"], x.get("chunk_id","")))
    fams = sorted(buckets.keys())
    res = []
    idx = 0
    while len(res) < quota and any(buckets.values()):
        fam = fams[idx % len(fams)]
        if buckets[fam]:
            res.append(buckets[fam].pop(0))
        idx += 1
        if all(len(v)==0 for v in buckets.values()):
            break
    return res[:quota]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    seed = int(manifest.get("global_seed", 424242))
    quotas = manifest.get("quotas", {})

    corpus = load_corpus(args.corpus)
    by_part = defaultdict(list)
    for obj in corpus:
        p = str(obj.get("part","")).strip()
        if p in quotas:
            by_part[p].append(obj)

    selected = []
    for part, quota in quotas.items():
        pool = by_part.get(str(part), [])
        if not pool: continue
        take = min(len(pool), int(quota))
        picked = stratified_pick(pool, take, seed)
        for c in picked:
            selected.append({
                "part": str(c.get("part","")),
                "chunk_id": c.get("chunk_id"),
                "canonical_citation": c.get("canonical_citation"),
                "section": c.get("section"),
                "subsection": c.get("subsection"),
                "text_preview": (c.get("text","")[:500])
            })

    selected.sort(key=lambda x: (x["part"], x["canonical_citation"] or "", x["chunk_id"] or ""))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    print(f"[sampler] Selected {len(selected)} chunks → {args.out}")

if __name__ == "__main__":
    main()
