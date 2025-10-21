# 📘 PDPA QA Dataset Schema Card (Lean Version)

This document describes the **minimal JSONL schema** used for all PDPA Question–Answer (QA) items.  
It is designed for **citation-aware**, **abstention-capable** evaluation of the RAG system on the Personal Data Protection Act (PDPA, Singapore).

---

## 1. Identification & Structure

| Field | Type | Description |
|--------|------|-------------|
| `id` | `string` | Unique QA identifier (e.g., `PDPA-QA-0001`). Sequential and stable across versions. |
| `part` | `string` | PDPA Part number (e.g., `"3"`, `"5"`, `"9"`). Used for coverage balancing and analysis. |
| `canonical_sections` | `array[string]` | Canonical citations such as `["PDPA s.14(1)", "PDPA s.14(2)"]`. Defines the statutory grounding of the QA. |

## 2. Question Definition

| Field | Type | Description |
|--------|------|-------------|
| `question_user` | `string` | The main natural-language question, phrased in **consumer-plain English**. |
| `question_variants` | `array[string]` | Paraphrased forms of the same question to improve retriever robustness. |
| `question_intent` | `array[string]` | Topical tags like `["consent"]`, `["access","correction"]`, `["breach"]`. Used for analysis purposes |
| `question_language` | `string` | Language code, e.g., `"en-SG"`. |

## 3. Ground-Truth Answers

| Field | Type | Description |
|--------|------|-------------|
| `gold_answer_short` | `string` | Concise, legally correct answer (1–3 sentences). Used for answer-scoring. |
| `gold_answer_extended` | `string` | Longer explanation or rationale for human interpretability. |


## 4. Abstention Policy

| Field | Type | Description |
|--------|------|-------------|
| `abstain_allowed` | `boolean` | Whether abstaining is acceptable if facts are missing. |
| `abstain_triggers` | `array[string]` | Key missing facts that justify abstention (e.g., `"purpose not specified"`). |
| `abstain_gold_message` | `string` | Canonical message expected when abstaining. |
| `ask_for_clarification_suggestions` | `array[string]` | Suggested clarifying questions the model should ask. |


## 5. Corpus Linking

| Field | Type | Description |
|--------|------|-------------|
| `corpus_links` | `array[object]` | Points to relevant corpus chunks. Each object: `{ "doc_id": "PDPA", "chunk_id": "PDPA-14-1-0" }`. Serves as the **source of truth** for citation evaluation. |
| `retrieval_hints` | `array[string]` | Optional hints for retrieval (“look for breach notification timelines”). |


## 6. Metadata & Style

| Field | Type | Description |
|--------|------|-------------|
| `difficulty` | `string` | One of `"easy"`, `"medium"`, `"hard"`. |
| `question_style` | `string` | Tone of user query (`"consumer-plain"`, `"lawyer-technical"`). |
| `expected_answer_style` | `string` | Tone of desired answer (`"lawyer-plain"`, `"legal-precise"`). |
| `version` | `string` | Legal corpus version, e.g., `"PDPA consolidated as of 2024-06-01"`. |
| `last_reviewed_utc` | `string (ISO)` | UTC timestamp of last manual review. |

## 7. Scoring Configuration

| Field | Type | Description |
|--------|------|-------------|
| `scoring` | `object` | Weighting for evaluation. Example:<br>`{"answer_correctness":0.65,"citation_correctness":0.35,"abstention_policy":0.0,"min_passing":0.8}` |
| `evaluation_notes` | `string` | Optional reviewer comments. |
| `qa_type` | `string` | Type of question: <br>`"pure-definitive"` – factual; <br>`"definitive-with-conditions"` – depends on facts; <br>`"scenario-ambiguous"` – partial info; <br>`"pure-abstain"` – must abstain. |


## Design Principles

| Goal | How Achieved |
|------|---------------|
| **Compact** | Only essential fields kept; no duplication between citations and corpus links. |
| **Citation-aware** | Each QA links directly to PDPA corpus chunks for retriever testing. |
| **Abstention-aware** | Explicit abstain policy fields allow factual uncertainty testing. |
| **Evaluation-ready** | Fully compatible with `pdpa_eval_harness.py`. |
| **Human-interpretable** | Every field corresponds to a clear reviewer decision. |


## Example Record

```json
{
  "id": "PDPA-QA-0008",
  "part": "4",
  "canonical_sections": ["PDPA s.14(1)", "PDPA s.14(2)"],
  "question_user": "How does the PDPA define valid consent?",
  "question_variants": [
    "What makes consent valid under PDPA?",
    "When is consent considered valid under the law?"
  ],
  "question_intent": ["consent"],
  "question_language": "en-SG",
  "gold_answer_short": "Consent is valid only if the individual has been informed of the purpose and has agreed to it.",
  "gold_answer_extended": "Under PDPA s.14(1)-(2), consent must be informed and specific to the stated purpose before collection, use, or disclosure of personal data.",
  "abstain_allowed": false,
  "abstain_triggers": [],
  "abstain_gold_message": "",
  "ask_for_clarification_suggestions": [],
  "corpus_links": [
    {"doc_id": "PDPA", "chunk_id": "PDPA-14-1-0"},
    {"doc_id": "PDPA", "chunk_id": "PDPA-14-2-0"}
  ],
  "retrieval_hints": ["definition of valid consent"],
  "difficulty": "medium",
  "question_style": "consumer-plain",
  "expected_answer_style": "lawyer-plain",
  "version": "PDPA consolidated as of 2024-06-01",
  "last_reviewed_utc": "2025-10-20T08:00:00Z",
  "scoring": {
    "answer_correctness": 0.65,
    "citation_correctness": 0.35,
    "abstention_policy": 0.0,
    "min_passing": 0.8
  },
  "evaluation_notes": "",
  "qa_type": "pure-definitive"
}
```

## Version Control & Validation

- Each JSONL line = one QA record.  
- Validate using `pdpa_dataset_validator.py` (see below).  
- Keep all IDs stable; update `last_reviewed_utc` when content changes.  
- Total dataset size is **500** QAs.
