import random
from typing import Any, Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import *
from enum import Enum

# ---------------------------- Models -------------------------
tok = AutoTokenizer.from_pretrained(HF_GENERATOR_MODEL, trust_remote_code=True)
gen = AutoModelForCausalLM.from_pretrained(HF_GENERATOR_MODEL, dtype=DTYPE, device_map="auto", trust_remote_code=True)
print("SLM Model loaded at: ", gen.device)

# --------------------------- Utilities --------------------------
CITE_RE = re.compile(r"\[PDPA s\.[0-9A-Za-z]+(?:\(\d+\))*?(?:\([a-z]\))?\]")

class Intent(str, Enum):
    OBLIGATION = "obligation"
    EXCEPTION = "exception"
    DEFINITION = "definition"
    TIMELINE = "timeline"
    PENALTY = "penalty"
    DEFAULT = "default"

def detect_intent(q: str) -> Intent:
    qs = q.lower()
    # if any(w in qs for w in ["must ", "required", "responsible", "duty"]):
    #     return Intent.OBLIGATION
    # if any(w in qs for w in ["unless", "except", "does not apply", "despite", "subject to"]):
    #     return Intent.EXCEPTION
    # if any(w in qs for w in ["define", "definition", "what does", "means", "meaning of"]):
    #     return Intent.DEFINITION
    # if any(w in qs for w in ["when ", "within", "how soon", "deadline", "days", "hours"]):
    #     return Intent.TIMELINE
    # if any(w in qs for w in ["fine", "penalt", "offence", "liable"]):
    #     return Intent.PENALTY
    return Intent.DEFAULT # always use the default format regardless of intent

def _sample_few_shots(intent: Intent, k: int = 1) -> str:
    bank = FEW_SHOTS.get(intent, [])
    if not bank:
        return ""
    chosen = random.sample(bank, k=min(k, len(bank)))
    blocks = []
    for q,a in chosen:
        blocks.append(f"Example Q: {q}\nExample A: {a}\n")
    return "\n".join(blocks)

# ------------------------ Template Library ----------------------
BASE_SYSTEM = (
    "You are a careful Singapore data-privacy assistant.\n"
    "Use ONLY the information from QUOTED EVIDENCES.\n"
    "Each QUOTED EVIDENCE will be prepended with its reranked and retrieval scores like [RERANK=...|RRF=...], these scores can be useful in determining the order of relevance of the evidences provided.\n"
    "ALL QUOTED EVIDENCES are relevant in some manner.\n"
    "After looking at all the QUOTED EVIDENCES, if you deem either the context or evidences provided to be insufficient, ABSTAIN from answering: \"I’m not sure; this appears outside the provided statute.\" AND ask ONE clarifying question.\n"
    "You can ONLY either abstain or answer to the best of your abilities. DO NOT do both."
)

TEMPLATES: Dict[Intent, str] = {
    # Optional templates to explore for extension
    Intent.OBLIGATION: (
        "{SYSTEM}\n\nFORMAT:\nShort answer (1–2 sentences)\n• Who must act\n• What action\n• When/Condition\nSources: [s.xx]\n\nEVIDENCE:\n{EVIDENCE}\n"
    ),
    Intent.EXCEPTION: (
        "{SYSTEM}\n\nFORMAT:\nShort answer (rule vs exception)\n• General rule\n• Exceptions / when it does not apply\nSources: [s.xx]\n\nEVIDENCE:\n{EVIDENCE}\n"
    ),
    Intent.DEFINITION: (
        "{SYSTEM}\n\nFORMAT:\nShort answer (quote + plain-English paraphrase)\n• Verbatim definition\n• Plain-English meaning\nSources: [s.xx]\n\nEVIDENCE:\n{EVIDENCE}\n"
    ),
    Intent.TIMELINE: (
        "{SYSTEM}\n\nFORMAT:\nShort answer (deadline-focused)\n• Trigger\n• Deadline / time limit\n• Who is responsible\nSources: [s.xx]\n\nEVIDENCE:\n{EVIDENCE}\n"
    ),
    Intent.PENALTY: (
        "{SYSTEM}\n\nFORMAT:\nShort answer (penalty-focused)\n• Who is liable\n• Penalty / range\n• Conditions\nSources: [s.xx]\n\nEVIDENCE:\n{EVIDENCE}\n"
    ),
    # Default template used
    Intent.DEFAULT: (
        "{SYSTEM}\n\nEVIDENCE:\n{EVIDENCE}\n\nFORMAT:\nWrite in lawyer-plain English. Answer concisely in 2-5 sentences. Preserve numbers and legal terms. Include the specific canonical citations in your sentences like this: PDPA s.xx(xx), except when abstaining. Do not use any other format to quote."
    ),
}

# Few-shot exemplars per intent (keep very short). Not used.
FEW_SHOTS: Dict[Intent, List[Tuple[str, str]]] = {
    Intent.OBLIGATION: [
        (
            "Do we have to report a notifiable data breach?",
            "Yes. Organisations must notify the Commission of a notifiable data breach within 3 days. [PDPA s.26D(2)]\n"
            "• Who: The organisation\n• What: Notify the Commission of a notifiable data breach\n• When: As soon as practicable, no later than 3 days\nSources: [PDPA s.26D(2)]"
        )
    ],
    Intent.EXCEPTION: [
        (
            "Are we always required to notify affected individuals?",
            "Not always. An organisation may be exempt from notifying affected individuals where specified conditions apply. [PDPA s.26D(5)]\n"
            "• Rule: Notify affected individuals\n• Exception: Exemptions apply under certain conditions\nSources: [PDPA s.26D(5)]"
        )
    ],
    Intent.DEFINITION: [
        (
            "What does \"personal data\" mean?",
            "\"Personal data\" means data about an identifiable individual. [PDPA s.2(1)]\n"
            "• Verbatim: data about an identifiable individual\n• Plain-English: information that can identify a person\nSources: [PDPA s.2(1)]"
        )
    ],
    Intent.TIMELINE: [],
    Intent.PENALTY: [],
    Intent.DEFAULT: [],
}

# ---------------------- Prompt Assembly -------------------------

def _quote_evidence(evidence: List[Dict[str, Any]]) -> str:
    lines = []
    for e in evidence:
        cid = e.get("id")
        rer = e.get("rerank_score")
        rrfs = e.get("rrf_score")
        prefix = ""
        # Include soft hints so the model can prioritize, but keep them optional
        # e.g., [RERANK=0.84|RRF=0.12]
        if rer is not None or rrfs is not None:
            parts = []
        if rer is not None:
            parts.append(f"RERANK={rer:.3f}")
        if rrfs is not None:
            parts.append(f"RRF={rrfs:.3f}")
        prefix = f"[{'|'.join(parts)}] "
        txt = (e.get("text", "").replace('"', '\"').strip())
        lines.append(f"{prefix}[{cid}] \"{txt}\"")
    return "\n".join(lines)


def build_prompt(question: str, evidence: List[Dict[str, Any]], intent: Optional[Intent] = None, fewshot_k: int = 1) -> str:
    intent = intent or detect_intent(question)
    tmpl = TEMPLATES[intent]
    # shots = _sample_few_shots(intent, fewshot_k)
    shots = None
    ev = _quote_evidence(evidence)
    sys = BASE_SYSTEM
    prompt = tmpl.format(SYSTEM=(sys + ("\n\n" + shots if shots else "")), EVIDENCE=ev, QUESTION=question)
    return prompt

def slm_generate(sys_prompt: str, question: str, max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = TEMPERATURE) -> str:
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question}
    ]
    text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tok([text], return_tensors="pt").to(gen.device)

    generated_ids = gen.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tok.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response