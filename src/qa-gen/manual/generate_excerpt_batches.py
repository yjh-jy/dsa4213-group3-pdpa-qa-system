"""
Used to generate the short prompts texts in data/qa/prompts/pdpa_part_XX_batches.txt
Example Usage:

python src/qa-gen/manual/generate_excerpt_batches.py --corpus data/corpus/corpus_subsection_v1.jsonl --outdir data/qa/prompts

"""
import argparse, json
from collections import defaultdict, OrderedDict
from pathlib import Path

def shorten(s, n):
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n].rstrip() + "…"

def split_long(s, max_len):
    s = " ".join((s or "").split())
    parts, i = [], 0
    while i < len(s):
        chunk = s[i:i+max_len]
        j = chunk.rfind(". ")
        if j >= 200:
            chunk = chunk[:j+1]
        parts.append(chunk.strip())
        i += len(chunk)
    return parts

def make_batches(items, min_size=8, max_size=12):
    batch, batches = [], []
    for item in items:
        batch.append(item)
        if len(batch) >= max_size:
            batches.append(batch); batch = []
    if batch:
        if len(batch) < min_size and batches and len(batches[-1]) > min_size:
            need = min_size - len(batch)
            move = batches[-1][-need:]
            batches[-1] = batches[-1][:-need]
            batch = move + batch
        batches.append(batch)
    return batches

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--excerpt-char-target", type=int, default=600)
    ap.add_argument("--full-char-threshold", type=int, default=1200)
    ap.add_argument("--split-char-threshold", type=int, default=3000)
    ap.add_argument("--batch-min", type=int, default=8)
    ap.add_argument("--batch-max", type=int, default=12)
    args = ap.parse_args()

    out_dir = Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)

    recs = []
    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            recs.append(json.loads(line))

    by_part = defaultdict(OrderedDict)
    for r in recs:
        part = str(r.get("part","")).strip()
        canon = r.get("canonical_citation") or r.get("citation") or ""
        text  = r.get("text","") or ""
        if not (part and canon and text): 
            continue
        if canon in by_part[part]:
            continue

        tlen = len(text)
        if tlen <= args.full_char_threshold:
            excerpt = " ".join(text.split())
            items = [(canon, excerpt)]
        elif tlen >= args.split_char_threshold:
            chunks = split_long(text, args.excerpt_char_target)
            items = []
            for i, ch in enumerate(chunks, start=1):
                label = f" [{chr(64+i)}]"
                items.append((canon + label, ch))
        else:
            excerpt = shorten(text, args.excerpt_char_target)
            items = [(canon, excerpt)]

        for (cc, ex) in items:
            by_part[part][cc] = ex

    created = []
    for part in sorted(by_part.keys(), key=lambda x: (int(x) if x.isdigit() else 99, x)):
        items = list(by_part[part].items())
        batches = make_batches(items, args.batch_min, args.batch_max)
        ppath = Path(out_dir) / f"pdpa_part_{part}_batches.txt"
        with open(ppath, "w", encoding="utf-8") as f:
            f.write(f"# PDPA Part {part} — Smart batches for ChatGPT (v3 prompt)\n\n")
            for i, batch in enumerate(batches, start=1):
                sections = [canon for canon,_ in batch]
                excerpts = "\n".join([f"{canon}: {text}" for canon, text in batch])
                f.write(f"""---
PART: {part}
SECTIONS INCLUDED: {", ".join(sections)}
TEXT EXCERPTS:
\"\"\"
{excerpts}
\"\"\"

# Suggestion: In the prompt, set N between {max(args.batch_min, len(batch))} and {len(batch)*4}.
""")
        created.append(str(ppath))

    print("[smart-excerpt] Wrote:")
    for c in created:
        print("-", c)

if __name__ == "__main__":
    main()
