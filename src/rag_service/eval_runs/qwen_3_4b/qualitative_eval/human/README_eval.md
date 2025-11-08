
# Human Eval (2 Raters, 50 Prompts) — Quick Start

## Files
- `create_double_blind.py`: Run this script to generate a template file with the PDPABench_test questions in randomized order. Follow the instructions there on the file paths.
- `raterX_template.xlsx`: Fill these with judgments.
- `analysis.py`: Run to compute win rates, significance (sign test), Cohen's kappa, and rating summaries.

## Rater instructions 
For each prompt:
1) Read the prompt, then **both answers** (System 1 & 2). Order is randomized.
2) Choose **preferred**: `S1`, `S2`, or `Tie`.
3) Rate **factuality** and **usefulness** (1–5) for each system, according to the scoring rubrics below.
4) Optional: short **comment**.

Rules: Judge content quality (accuracy, usefulness, clarity). Ignore style unless it affects usefulness. Stay blind to system identity.

### Usefulness Rubric

| **Score**                 | **Lay Rater (Perceived Helpfulness)**                                                                  | **Expert Rater (Legal Relevance & Completeness)**                                             |
| ------------------------- | ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| **1 – Not Useful**        | Unclear, irrelevant, or confusing; doesn’t address the question.                                       | Off-topic or legally irrelevant; omits core provision.                                        |
| **2 – Weakly Useful**     | Somewhat related but lacks clarity or omits key explanation; reader still confused.                    | Mentions related topic but misses material context or misapplies the statute.                 |
| **3 – Moderately Useful** | Provides partial understanding or vague explanation; understandable but incomplete.                    | Covers main idea but lacks depth, completeness, or nuance in legal framing.                   |
| **4 – Useful**            | Clearly answers the question, improves understanding, or highlights relevant law; easy to follow.      | Legally relevant, sufficiently complete, and consistent with statutory scope.                 |
| **5 – Highly Useful**     | Very clear, well-explained, and directly satisfies informational need; feels complete and trustworthy. | Fully relevant, accurate, and comprehensive response with precise alignment to legal context. |

### Factuality Rubric
| **Score**                                       | **Lay Rater (Perceived Groundedness)**                                                                                  | **Expert Rater (Legal Correctness)**                                                                             |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **1 – Ungrounded / Incorrect**                  | The answer makes unsupported or speculative claims; no citation or reasoning is evident; feels unreliable or “made-up.” | The cited provision does not exist, is misquoted, or the interpretation is clearly wrong or contradicts the law. |
| **2 – Weakly Grounded / Partially Incorrect**   | Some reference to law or reasoning, but unclear or feels disconnected from the question; confidence is low.             | The answer cites a vaguely related section or omits key conditions that change the legal meaning.                |
| **3 – Moderately Grounded / Partially Correct** | Appears based on real law or reasoning, but lacks clear linkage or full explanation; plausible but uncertain.           | Cites a generally correct section but overgeneralizes, omits exceptions, or mixes unrelated provisions.          |
| **4 – Well Grounded / Largely Correct**         | Clearly draws from legal text or reasoning; feels trustworthy and evidence-based though not perfectly precise.          | Interpretation aligns with cited section; minor omissions or simplifications acceptable for summary use.         |
| **5 – Fully Grounded / Correct**                | Strongly supported by explicit citation and reasoning; confident, coherent, and self-consistent.                        | All cited provisions are accurate and properly applied; the explanation is legally faithful and complete.        |


## How to run analysis
``` bash
cd src/rag_service/eval_runs
python analysis.py \
  --rater1 rater1_answers.xlsx \
  --rater2 rater2_answers.xlsx \
  --preview preview_pairs.csv
```
Outputs:
- Per-rater and aggregate **win/tie/loss** for RAG 
- **Two-sided sign test** (exact binomial).
- **Cohen's kappa** on preferences.
- Mean 1–5 **factuality**/**usefulness** per system.
