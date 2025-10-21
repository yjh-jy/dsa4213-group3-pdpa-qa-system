# Workflow (path to 500 QAs)

## 1. Generate batches per Part:
Generates chat-friendly blocks with smarter excerpting:
- Full text for short sections (≤ --full-char-threshold, default 1200 chars).
- Trimmed excerpts around --excerpt-char-target (default 600 chars).
- Auto-splitting very long sections into [A], [B], … when over --split-char-threshold (default 3000 chars).
- Outputs 8–12 sections per batch by default (tunable).

```python
python src/qa-gen/manual/generate_excerpt_batches.py --corpus data/corpus/corpus_subsection_v1.jsonl --outdir data/qa/prompts
```
## 2. For each pdpa_part_X_batches.txt:
- Open a fresh ChatGPT chat for each part to avoid context rot. (We used ChatGPT 5 auto)
- Paste your v3 prompt.
- Paste one batch block from the .txt file.
- Set N (e.g., 50) and save the JSON output to data/qa/prompts/runs/partX_batchY.json. The total N selected will slightly exceed the total 500, we will trim it down later.

## 3. Merge and validate:
Merges one or more ChatGPT JSON arrays into the minimal schema, normalizes citations, adds corpus_links, assigns stable IDs, and runs validator.
   
```python
python merge_chatgpt_json.py \
  --corpus data/corpus/corpus_subsection_v1.jsonl \
  --inputs data/qa/prompts/runs/*.json \
  --out data/qa/pdpa_qa_500.jsonl \
  --run-validator src/qa-gen/pdpa_dataset_validator.py \
  --expect-count 500
```