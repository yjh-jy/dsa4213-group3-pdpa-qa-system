
# Human Eval (N Raters, 50 Prompts)

## Files
- `create_double_blind.py`: Run this script to generate a template file with the PDPABench_test questions in randomized order. Follow the instructions regarding the file paths.
- `raterX_template.xlsx`: Before filling these with judgments, remember to blank out the last column as it contains the answers to which system is rag.
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
1. Navigate
   ``` bash
   cd src/rag_service/eval_runs/qwen_3_4b/qualitative_eval/human
   ```
2. Run
   ``` bash
   python analysis.py --raters rater1_answers.xlsx rater2_answers.xlsx
      # [rater3_answers.csv ...] you can add how many raters you have
   ```
### Outputs (in terminal window):
- Per-rater and aggregate **win/tie/loss** for RAG 
- **Two-sided sign test** (exact binomial).
- **Cohen's kappa** on preferences.
- Mean 1–5 **factuality**/**usefulness** per system.
  
### Example output:
```
=== Pairwise preference (per rater, per-row RAG mapping) ===
rater1_answers.xlsx: {'rag_wins': 13, 'rag_losses': 20, 'ties': 17, 'n_effective': 33, 'p_value_sign_test': 0.296206368599087, 'unmapped_or_missing': 0}
rater2_answers.xlsx: {'rag_wins': 28, 'rag_losses': 15, 'ties': 7, 'n_effective': 43, 'p_value_sign_test': 0.0659940344557981, 'unmapped_or_missing': 0}
rater3_answers.xlsx: {'rag_wins': 36, 'rag_losses': 14, 'ties': 0, 'n_effective': 50, 'p_value_sign_test': 0.0026021714567221466, 'unmapped_or_missing': 0}
rater4_answers.xlsx: {'rag_wins': 36, 'rag_losses': 14, 'ties': 0, 'n_effective': 50, 'p_value_sign_test': 0.0026021714567221466, 'unmapped_or_missing': 0}

=== Pairwise preference (aggregate, per-row mapping) ===
{'rag_wins': 113, 'rag_losses': 63, 'ties': 24, 'n_effective': 176, 'p_value_sign_test': 0.00020218969307107758, 'unmapped_or_missing': 0}

=== Mean ratings (aggregate, RAG vs BASE) ===
{'RAG': {'factuality_mean': 4.5, 'usefulness_mean': 4.225, 'n': 200}, 'BASE': {'factuality_mean': 3.955, 'usefulness_mean': 3.95, 'n': 200}}

=== Inter-rater agreement on common prompts ===
{'fleiss_kappa': 0.13, 'raw_observed_agreement': 0.507, 'raw_observed_agreement_pct': 50.7, 'n_items': 50, 'n_raters_expected': 4}

=== Mean ratings (per rater, RAG vs BASE) ===
rater1_answers.xlsx: {'RAG': {'factuality_mean': 4.26, 'usefulness_mean': 4.14, 'n': 50}, 'BASE': {'factuality_mean': 4.36, 'usefulness_mean': 4.34, 'n': 50}}
rater2_answers.xlsx: {'RAG': {'factuality_mean': 4.72, 'usefulness_mean': 4.24, 'n': 50}, 'BASE': {'factuality_mean': 3.96, 'usefulness_mean': 3.72, 'n': 50}}
rater3_answers.xlsx: {'RAG': {'factuality_mean': 4.7, 'usefulness_mean': 4.38, 'n': 50}, 'BASE': {'factuality_mean': 4.34, 'usefulness_mean': 4.1, 'n': 50}}
rater4_answers.xlsx: {'RAG': {'factuality_mean': 4.32, 'usefulness_mean': 4.14, 'n': 50}, 'BASE': {'factuality_mean': 3.16, 'usefulness_mean': 3.64, 'n': 50}}
```