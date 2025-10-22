"""
Build PDPA QAs from selected chunks deterministically and assign stable IDs.

Usage:
  python src/data-related/build_qas.py \
    --selected data/qa/selected_chunks.json \
    --manifest data/qa/manifest.json \
    --out data/qa/pdpa_qa_500.jsonl \
    --mode mock
    [--mode mock|llm]

Modes:
- mock (default): produce placeholder QA content deterministically (no LLM calls)
- llm: call your LLM client function (generate_with_llm)

Repro knobs:
- global_seed (manifest)
- Deterministic order and ID assignment
"""

import argparse, json, re, random, time, os
from datetime import datetime, timezone

def normalize_citation(c):
    c = (c or "").strip()
    c = re.sub(r"(?i)^pdpa\s*", "", c)
    c = re.sub(r"(?i)^s\.", "", c).strip()
    c = re.sub(r"\s+", "", c)
    return f"PDPA s.{c}"

def make_id(idx:int) -> str:
    return f"PDPA-QA-{idx:04d}"

def generate_mock(chunk, rng):
    canon = normalize_citation(chunk.get("canonical_citation"))
    part = str(chunk.get("part",""))
    section = canon.split("s.")[-1]
    q = f"In PDPA {section}, what does it say in simple terms?"
    a = f"{canon} sets out the rule in plain terms. Organisations must comply with the stated requirements and exceptions in the PDPA."
    return {
        "question_user": q,
        "question_variants": [q, f"In simple terms: {q.lower()}"],
        "gold_answer_short": a,
        "gold_answer_extended": a,
        "question_intent": ["accountability"] if part in {"1","2","3"} else ["enforcement"] if part in {"9","10"} else ["consent"],
        "qa_type": "pure-definitive",
        "abstain_allowed": False,
        "abstain_triggers": [],
        "abstain_gold_message": "",
        "ask_for_clarification_suggestions": []
    }

def generate_with_llm(chunk, model, decoding, seed):
    """
    OpenAI-compatible client call to model="gpt-5-thinking".
    Requires OPENAI_API_KEY in environment. Returns strict JSON fields.
    """
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("OpenAI client not installed. Run: pip install openai>=1.40.0") from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=api_key)

    canon = (chunk.get("canonical_citation") or "").strip()
    part = str(chunk.get("part",""))
    text_snippet = (chunk.get("text_preview","") or "")[:1200]
    default_intent = "accountability" if part in {"1","2","3"} else ("enforcement" if part in {"9","10"} else "consent")

    system_msg = (
        "You are GPT-5 Thinking, a meticulous Singapore PDPA assistant. "
        "Write plain-English (lawyer-plain) content grounded ONLY in the PDPA text provided. "
        "Keep answers concise (2-5 sentences). ALWAYS cite PDPA sections in canonical form like 'PDPA s.14(1)'. "
        "If facts are missing in a way that changes the legal outcome, abstain: give a short reason and ONE clarifying question. "
        "Output STRICT JSON with the keys requested. Do NOT add commentary."
    )

    user_payload = {
        "task": "Produce a PDPA QA in strict JSON for the given statute chunk.",
        "pdpa_part": part,
        "canonical_section": canon,
        "statute_excerpt": text_snippet,
        "defaults": {
            "question_style": "consumer-plain",
            "expected_answer_style": "lawyer-plain",
            "question_intent_hint": default_intent
        },
        "schema": {
            "question_user": "string",
            "question_variants": ["string","string"],
            "gold_answer_short": "string",
            "gold_answer_extended": "string",
            "question_intent": ["string"],
            "qa_type": "one of ['pure-definitive','definitive-with-conditions','scenario-ambiguous','pure-abstain']",
            "abstain_allowed": "boolean",
            "abstain_triggers": ["string"],
            "abstain_gold_message": "string",
            "ask_for_clarification_suggestions": ["string"]
        }
    }

    response_format = {"type": "json_object"}
    temperature = float(decoding.get("temperature", 0.2))
    top_p = float(decoding.get("top_p", 0.9))
    max_tokens = int(decoding.get("max_tokens", 450))

    last_err = None
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role":"system","content":system_msg},
                    {"role":"user","content":json.dumps(user_payload, ensure_ascii=False)}
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                response_format=response_format,
                seed=seed
            )
            content = resp.choices[0].message.content
            data = json.loads(content)

            # Normalize arrays/flags
            data["qa_type"] = str(data.get("qa_type","no-response"))
            data["abstain_allowed"] = bool(data.get("abstain_allowed", data["qa_type"] in ["pure-abstain","scenario-ambiguous"]))
            for k in ["question_variants","question_intent","abstain_triggers","ask_for_clarification_suggestions"]:
                v = data.get(k, [])
                if isinstance(v, str): 
                    v=[v]
                data[k] = v
            for k in ["question_user", "gold_answer_short"]:
                if not data.get(k): 
                    data[k] = ""
            return data
        except Exception as e:
            last_err = e
            time.sleep(1.2 * (attempt+1))
    raise RuntimeError(f"LLM call failed after retries: {last_err}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["mock","llm"], default="mock")
    args = ap.parse_args()

    with open(args.selected, "r", encoding="utf-8") as f:
        selected = json.load(f)
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    seed = int(manifest.get("global_seed", 424242))
    model = manifest.get("model", "gpt-5-thinking")
    decoding = manifest.get("decoding", {"temperature":0.2,"top_p":0.9,"max_tokens":450})
    version = manifest.get("version", "PDPA consolidated as of 2024-06-01")

    selected.sort(key=lambda x: (x["part"], x["canonical_citation"] or "", x["chunk_id"] or ""))
    rng = random.Random(seed)

    out_lines = []
    for idx, c in enumerate(selected, start=1):
        canon = normalize_citation(c.get("canonical_citation"))
        part = str(c.get("part",""))
        if args.mode == "llm":
            qa = generate_with_llm(c, model, decoding, seed + idx)
        else:
            qa = generate_mock(c, rng)

        record = {
            "id": make_id(idx),
            "part": part,
            "canonical_sections": [canon] if canon else [],
            "question_user": qa["question_user"],
            "question_variants": qa.get("question_variants", []),
            "question_intent": qa.get("question_intent", []),
            "question_language": "en-SG",
            "gold_answer_short": qa["gold_answer_short"],
            "gold_answer_extended": qa.get("gold_answer_extended", qa["gold_answer_short"]),
            "abstain_allowed": qa.get("abstain_allowed", False),
            "abstain_triggers": qa.get("abstain_triggers", []),
            "abstain_gold_message": qa.get("abstain_gold_message", ""),
            "ask_for_clarification_suggestions": qa.get("ask_for_clarification_suggestions", []),
            "corpus_links": [{"doc_id":"PDPA","chunk_id": c["chunk_id"]}] if c.get("chunk_id") else [],
            "retrieval_hints": [],
            "difficulty": "medium",
            "question_style": "consumer-plain",
            "expected_answer_style": "lawyer-plain",
            "version": version,
            "last_reviewed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "scoring": {"answer_correctness":0.65,"citation_correctness":0.35,"abstention_policy":0.0,"min_passing":0.8},
            "evaluation_notes": "",
            "qa_type": qa.get("qa_type","pure-definitive")
        }
        out_lines.append(record)

    with open(args.out, "w", encoding="utf-8") as f:
        for obj in out_lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[builder] Wrote {len(out_lines)} QAs → {args.out}")

if __name__ == "__main__":
    main()
