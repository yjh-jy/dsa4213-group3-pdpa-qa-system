"""
This script is used to generate templates for human raters to evaluate the generated response from test set for PDPABench.
The generated responses are in randomized order and the answers are in the last column.
Users of this script should blank out the last column before handing it to the raters to ensure fairness.

Usage:
python src/rag_service/eval_runs/qwen_3_4b/qualitative_eval/human/create_double_blind.py \
  --sys1 src/rag_service/eval_runs/qwen_3_4b/quantitative_eval/non_reasoning/no_rag_20251105T170423/detailed_results.jsonl \
  --sys2 src/rag_service/eval_runs/qwen_3_4b/quantitative_eval/non_reasoning/rag_20251105T164809/detailed_results.jsonl \
  --out rater1_template.xlsx \
  --max_n 50 \
  --seed 101

- Change the seed for different raters 
- Choose the input file path of the generated response you wish to compare, here we choose the non reasoning models' responses of rag and non rag variants

"""

import argparse, json, random
import pandas as pd
from pathlib import Path

def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            q = obj.get("query", "")
            pred = obj.get("prediction", {})
            ans = pred.get("answer_text", "") if isinstance(pred, dict) else ""
            rows.append({"query": q, "answer": ans})
    df = pd.DataFrame(rows).drop_duplicates(subset=["query"]).reset_index(drop=True)
    return df

def build_rater_sheet(pairs: pd.DataFrame, seed: int) -> pd.DataFrame:
    # Randomize row order
    pairs = pairs.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    # Flip which side shows RAG per row (True => keep No-RAG left; False => swap)
    random.seed(seed)
    side = [random.choice([True, False]) for _ in range(len(pairs))]

    out_rows = []
    for i, keep_sys1_left in enumerate(side):
        row = pairs.iloc[i]
        if keep_sys1_left:
            s1, s2 = row["answer_sys1"], row["answer_sys2"]  # left: no-RAG, right: RAG
            rag_side = "S2"
        else:
            s1, s2 = row["answer_sys2"], row["answer_sys1"]  # left: RAG, right: no-RAG
            rag_side = "S1"
        out_rows.append({
            "prompt_id": i + 1,
            "prompt_text": row["query"],
            "system1_output": s1,
            "system2_output": s2,
            "preferred": "",            # rater fills: S1 / S2 / Tie
            "factuality_s1": "",        # 1–5
            "usefulness_s1": "",        # 1–5
            "factuality_s2": "",        # 1–5
            "usefulness_s2": "",        # 1–5
            "comment": "",              # optional
            "_rag_side_for_analysis": rag_side,  # optional helper (you may delete if you want it fully blind)
        })
    return pd.DataFrame(out_rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys1", required=True, help="JSONL for System 1 (no-RAG)")
    ap.add_argument("--sys2", required=True, help="JSONL for System 2 (RAG)")
    ap.add_argument("--out", default="rater1_template.xlsx", help="Output .xlsx path")
    ap.add_argument("--max_n", type=int, default=52, help="Number of questions to include (default 52)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    d1 = load_jsonl(Path(args.sys1))  # no-RAG
    d2 = load_jsonl(Path(args.sys2))  # RAG

    # Align by query (prefer exact join; fallback to index if no overlap)
    merged = pd.merge(d1, d2, on="query", how="inner", suffixes=("_sys1", "_sys2"))
    if len(merged) == 0:
        n = min(len(d1), len(d2))
        merged = pd.DataFrame({
            "query": d1["query"].iloc[:n].tolist(),
            "answer_sys1": d1["answer"].iloc[:n].tolist(),
            "answer_sys2": d2["answer"].iloc[:n].tolist(),
        })
    else:
        merged = merged.rename(columns={"answer_sys1": "answer_sys1", "answer_sys2": "answer_sys2"})

    merged = merged.drop_duplicates(subset=["query"]).reset_index(drop=True)
    if args.max_n and args.max_n > 0:
        merged = merged.iloc[:args.max_n].copy()

    # Build rater 1 sheet (ratings) + mapping preview
    ratings = build_rater_sheet(merged, args.seed)
    mapping = merged.rename(columns={
        "query": "prompt_text",
        "answer_sys1": "system1_answer (no-RAG)",
        "answer_sys2": "system2_answer (RAG)",
    })
    mapping.insert(0, "prompt_id (same order as 'ratings' sheet after shuffle)", range(1, len(mapping) + 1))

    # Write Excel with two sheets
    with pd.ExcelWriter(args.out, engine="openpyxl") as xw:
        ratings.to_excel(xw, sheet_name="ratings", index=False)
        mapping.to_excel(xw, sheet_name="preview_mapping", index=False)

    print(f"[OK] Wrote {args.out} with {len(ratings)} items.")

if __name__ == "__main__":
    main()
