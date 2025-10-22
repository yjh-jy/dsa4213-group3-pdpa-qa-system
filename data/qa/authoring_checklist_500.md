
# Authoring Checklist & Quotas — 500 PDPA QAs (Parts 1–6, 9–10)

## Target distribution (by Part)
- Part 1 — **25**
- Part 2 — **25**
- Part 3 — **90**
- Part 4 — **70**
- Part 5 — **90**
- Part 6 — **70**
- Part 9 — **95**
- Part 10 — **35**
**Total: 500**

**Abstention items:** aim for **15–20%** overall (≈75–100 items), spread across parts with fact-dependent scenarios (e.g., consent scope, breach thresholds, transfer safeguards).

---

## Each QA must include (example)
- `question_user` in *consumer-plain* tone (1–2 variants encouraged).
- `gold_answer_short` in *lawyer-plain* (2–5 sentences).
- `canonical_sections` (1–3 entries) in canonical form: `PDPA s.<section>(<sub>)`.
- `corpus_links` with at least one valid `{doc_id:'PDPA', chunk_id:'...'}`.
- `qa_type`: one of `pure-definitive`, `definitive-with-conditions`, `scenario-ambiguous`, `pure-abstain`.
- `scoring` weights (defaults: 0.65 / 0.35 / 0.0 unless abstention is expected).
- `id`: simple sequential `PDPA-QA-XXXX` (maintain stability once assigned).

For the full schema, please refer to `schema_card.md`. 


## Quality checklist (tick before commit)
- [ ] **Tone check**: Q is consumer-plain; A is lawyer-plain.
- [ ] **Specificity**: Answer addresses the exact question asked.
- [ ] **Citations**: Canonicalized; validate against corpus; no unrelated cites.
- [ ] **Corpus link**: Points to the correct chunk(s); opens without error.
- [ ] **Ambiguity**: If facts are missing, prefer `scenario-ambiguous` or `pure-abstain`.
- [ ] **Variants**: Add at least one paraphrase for retrieval robustness (optional but recommended).
- [ ] **Diff noise**: No trailing spaces, weird unicode dashes, or smart quotes in citations.
- [ ] **Pass CI**: Schema validator runs cleanly; IDs unique; no duplicate questions.

---

## Suggested topic coverage by Part
**Part 1** — Purpose, scope, exclusions (domestic/personal), definitions likely to affect scope.  
**Part 2** — PDPC functions, advisory committees, delegation, investigations overview.  
**Part 3** — Accountability principle; policies, DPO; protection, accuracy, retention, access/correction interface.  
**Part 4** — Consent (actual/deemed), withdrawals, exceptions, reasonable purpose.  
**Part 5** — Access and correction rights, refusals (safety, confidentiality, national interest), response timelines.  
**Part 6** — Data breach: assessment, notification to PDPC and individuals, exceptions, timelines.  
**Part 9** — DNC regime, specified messages, offences (48D etc.), PDPC directions and penalties, appeals.  
**Part 10** — Confidentiality, national interest certificate, exemptions, corporate/officer liability.

---

## File hygiene & CI
- Keep JSONL one object per line; UTF-8; LF line endings.
- Run a schema & citation validator before merging.
- Maintain a running `CHANGELOG.md` for edits to QAs or citations.
