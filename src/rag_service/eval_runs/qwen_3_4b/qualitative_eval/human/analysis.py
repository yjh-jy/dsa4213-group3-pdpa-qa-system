#!/usr/bin/env python3
"""
Usage:
    cd src/rag_service/eval_runs/qwen_3_4b/qualitative_eval/human
    python analysis.py --raters rater1_answers.xlsx rater2_answers.xlsx rater3_answers.xlsx [rater4.csv ...]
  # Optional: if your side column has a custom name common to all files:
  # --side_col _rag_side_for_analysis

Each rater file must contain:
  prompt_text, system1_output, system2_output, preferred,
  factuality_s1, usefulness_s1, factuality_s2, usefulness_s2,
  and a per-row side mapping ('S1'/'S2'):
    - column '_rag_side_for_analysis' OR
    - if not present, the script uses the **last column** as the side column.
"""

import argparse
import math
import re
from pathlib import Path
import pandas as pd
from collections import Counter

# ---------- utils ----------

def norm(s):
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    return re.sub(r"\s+", " ", s.strip())

def read_table(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

def exact_binomial_test(k, n, p=0.5):
    def pmf(x):
        from math import comb
        return comb(n, x) * (p ** x) * ((1 - p) ** (n - x))
    obs = pmf(k)
    return sum(pmf(x) for x in range(n + 1) if pmf(x) <= obs + 1e-15)

def get_rag_side_column(df: pd.DataFrame, explicit_col: str | None):
    if explicit_col and explicit_col in df.columns:
        return explicit_col
    if "_rag_side_for_analysis" in df.columns:
        return "_rag_side_for_analysis"
    return df.columns[-1]

def summarize_with_side(df: pd.DataFrame, side_col: str):
    rag_wins = rag_losses = ties = unmapped = 0
    mapped_labels = []  # 'RAG'/'BASE'/'Tie'
    for _, row in df.iterrows():
        side = str(row.get(side_col, "")).strip()
        pref = str(row.get("preferred", "")).strip()
        if side not in ("S1", "S2"):
            unmapped += 1
            continue
        if pref == "Tie":
            ties += 1
            mapped_labels.append("Tie")
        elif pref in ("S1", "S2"):
            if pref == side:
                rag_wins += 1
                mapped_labels.append("RAG")
            else:
                rag_losses += 1
                mapped_labels.append("BASE")
        else:
            unmapped += 1
    n_eff = rag_wins + rag_losses
    pval = exact_binomial_test(rag_wins, n_eff, p=0.5) if n_eff > 0 else float("nan")
    out = {
        "rag_wins": rag_wins,
        "rag_losses": rag_losses,
        "ties": ties,
        "n_effective": n_eff,
        "p_value_sign_test": pval,
        "unmapped_or_missing": unmapped,
    }
    return out, mapped_labels

def extract_rag_ratings(df: pd.DataFrame, side_col: str):
    rag_facts, rag_use, base_facts, base_use = [], [], [], []
    for _, row in df.iterrows():
        side = str(row.get(side_col, "")).strip()
        if side == "S1":
            fs_rag = row.get("factuality_s1", None); us_rag = row.get("usefulness_s1", None)
            fs_base = row.get("factuality_s2", None); us_base = row.get("usefulness_s2", None)
        elif side == "S2":
            fs_rag = row.get("factuality_s2", None); us_rag = row.get("usefulness_s2", None)
            fs_base = row.get("factuality_s1", None); us_base = row.get("usefulness_s1", None)
        else:
            continue
        try:
            rag_facts.append(float(fs_rag)); rag_use.append(float(us_rag))
            base_facts.append(float(fs_base)); base_use.append(float(us_base))
        except (TypeError, ValueError):
            continue
    mean = lambda x: round(sum(x)/len(x), 3) if x else float("nan")
    return {
        "RAG":  {"factuality_mean": mean(rag_facts),  "usefulness_mean": mean(rag_use),  "n": len(rag_use)},
        "BASE": {"factuality_mean": mean(base_facts), "usefulness_mean": mean(base_use), "n": len(base_use)},
    }

# ---------- agreement for N raters (Fleiss' kappa + raw agreement) ----------

def fleiss_kappa_and_raw_agreement(label_matrix, labels=("RAG", "BASE", "Tie")):
    """
    label_matrix: list of dict counts per item, e.g. [{'RAG':2,'BASE':0,'Tie':0}, ...]
    Returns: (kappa, raw_observed_agreement_Pbar)
    """
    cats = list(labels)
    m = len(label_matrix)              # items
    if m == 0: 
        return float("nan"), float("nan")
    n = sum(label_matrix[0].values())  # raters per item (assumed constant)
    if n == 0: 
        return float("nan"), float("nan")

    # Category proportions across all items
    p = []
    for c in cats:
        p_c = sum(item.get(c,0) for item in label_matrix) / (m * n)
        p.append(p_c)

    # Item-wise observed agreement
    P = []
    for item in label_matrix:
        s = sum(item.get(c,0)**2 for c in cats)
        P_i = (s - n) / (n*(n-1)) if n > 1 else 0.0
        P.append(P_i)

    Pbar = sum(P)/m           # Raw observed agreement (mean of item agreements)
    Pbar_e = sum(pc**2 for pc in p)  # Expected chance agreement

    if Pbar_e == 1.0:
        return 1.0, Pbar
    kappa = (Pbar - Pbar_e) / (1 - Pbar_e) if (1 - Pbar_e) != 0 else float("nan")
    return kappa, Pbar

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raters", nargs="+", required=True, help="List of rater files (.xlsx or .csv).")
    ap.add_argument("--side_col", default=None,
                    help="Optional: column name containing per-row RAG side ('S1'/'S2'). "
                         "If omitted, uses '_rag_side_for_analysis' or the last column.")
    args = ap.parse_args()

    # Load all raters
    raters = []
    for path in args.raters:
        df = read_table(path)
        side_col = get_rag_side_column(df, args.side_col)
        raters.append((Path(path).name, df, side_col))

    # --- Per-rater outputs ---
    print("=== Pairwise preference (per rater, per-row RAG mapping) ===")
    per_rater_maps = {}   # name -> dict prompt_text -> label
    for name, df, side_col in raters:
        s, _ = summarize_with_side(df, side_col)
        print(f"{name}:", s)
        # Build mapping by prompt to enable cross-rater agreement later
        m = {}
        for _, row in df.iterrows():
            side = str(row.get(side_col, "")).strip()
            pref = str(row.get("preferred", "")).strip()
            if side not in ("S1","S2") or pref not in ("S1","S2","Tie"):
                continue
            lbl = "Tie" if pref == "Tie" else ("RAG" if pref == side else "BASE")
            m[norm(row.get("prompt_text",""))] = lbl
        per_rater_maps[name] = m

    # --- Aggregate across all raters (concatenate and unify side col) ---
    unified = []
    for _, df, side_col in raters:
        unified.append(df.rename(columns={side_col: "_side_col"}))
    both = pd.concat(unified, ignore_index=True)

    agg, _ = summarize_with_side(both, "_side_col")
    print("\n=== Pairwise preference (aggregate, per-row mapping) ===")
    print(agg)

    print("\n=== Mean ratings (aggregate, RAG vs BASE) ===")
    print(extract_rag_ratings(both, "_side_col"))

    # --- Inter-rater agreement: Fleiss' κ + Raw observed agreement on common prompts ---
    prompts = None
    for _, df, _ in raters:
        pset = set(norm(x) for x in df["prompt_text"].dropna().tolist())
        prompts = pset if prompts is None else (prompts & pset)
    prompts = sorted(prompts) if prompts else []

    label_matrix = []
    for q in prompts:
        counts = Counter()
        n_present = 0
        for name in per_rater_maps:
            lab = per_rater_maps[name].get(q, None)
            if lab in ("RAG","BASE","Tie"):
                counts[lab] += 1
                n_present += 1
        if n_present > 0:
            label_matrix.append(counts)

    kappa, pbar = fleiss_kappa_and_raw_agreement(label_matrix, labels=("RAG","BASE","Tie"))
    print("\n=== Inter-rater agreement on common prompts ===")
    print({
        "fleiss_kappa": (round(kappa,3) if kappa==kappa else "nan"),
        "raw_observed_agreement": (round(pbar,3) if pbar==pbar else "nan"),  # proportion in [0,1]
        "raw_observed_agreement_pct": (round(pbar*100,1) if pbar==pbar else "nan"),
        "n_items": len(label_matrix),
        "n_raters_expected": (len(raters) if raters else 0),
    })

    # --- Per-rater mean ratings (for convenience) ---
    print("\n=== Mean ratings (per rater, RAG vs BASE) ===")
    for name, df, side_col in raters:
        print(name + ":", extract_rag_ratings(df, side_col))

if __name__ == "__main__":
    main()
