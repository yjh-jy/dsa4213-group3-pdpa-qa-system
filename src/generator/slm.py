import random
from typing import Any, Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from config import *
from enum import Enum
import re

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


# --- minimal helpers / patterns ---
_THINK_BLOCK   = re.compile(r"<think>(.*?)(?:</think>|$)", flags=re.S)   # closed or truncated
_IM_END_TAIL   = re.compile(r"\s*<\|im_end\|>\s*$")

# class _StopOnText(StoppingCriteria):
#     def __init__(self, tok, substrings):
#         self.tok = tok
#         self.substrings = tuple(substrings)
#     def __call__(self, input_ids, scores, **kwargs):
#         text = self.tok.decode(input_ids[0], skip_special_tokens=False)
#         return any(s in text for s in self.substrings)
class _StopOnText(StoppingCriteria):
    def __init__(self, tok, substrings, start_len: int, min_new_tokens: int = 0):
        self.tok = tok
        self.substrings = tuple(substrings)
        self.start_len = start_len         # where generated tokens begin
        self.min_new_tokens = min_new_tokens
    def __call__(self, input_ids, scores, **kwargs):
        # only decode the newly generated tail
        new_len = input_ids.shape[1] - self.start_len
        if new_len < self.min_new_tokens:
            return False
        tail_ids = input_ids[0, self.start_len:]
        tail_text = self.tok.decode(tail_ids, skip_special_tokens=False)
        return any(s in tail_text for s in self.substrings)
    
def slm_generate(
    sys_prompt: str,
    question: str,
    max_new_tokens: int,
    use_reasoning: bool,
    *,
    think_ratio: float = 0.6,                    # pass-1 budget share if reasoning
    answer_stops: tuple = ("\nCitations:", "<|im_end|>"),
    print_thinking: bool = True,
) -> Tuple[str, str]:
    """
    Two-pass when use_reasoning=True, single-pass otherwise.
    Returns (content, thinking). 'content' is always cleaned of think blocks & tail markers.
    """

    # ---------- shared prompt ----------
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": question},
    ]
    base_text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=use_reasoning
    )
    base_inputs = tok([base_text], return_tensors="pt").to(gen.device)

    # ---------- non-thinking: single pass ----------
    if not use_reasoning:
        out = gen.generate(
            **base_inputs,
            max_new_tokens=max_new_tokens,
            temperature=TEMPERATURE_QWEN3_NON_THINKING,
            top_p=TOP_P_QWEN3_NON_THINKING,
            top_k=TOP_K_QWEN3_NON_THINKING,
        )
        new_ids = out[0][base_inputs.input_ids.shape[1]:].tolist()
        decoded = tok.decode(new_ids, skip_special_tokens=False)

        thinking = (_THINK_BLOCK.search(decoded).group(1).strip()
                    if _THINK_BLOCK.search(decoded) else "")
        content  = _THINK_BLOCK.sub("", decoded).strip()
        content  = _IM_END_TAIL.sub("", content).rstrip()
        content  = content.replace("<think>", "").replace("</think>", "")
        if print_thinking and thinking:
            print(thinking)
        return content, thinking

    # ---------- reasoning: two pass ----------
    # split budget
    think_tokens  = max(1, int(max_new_tokens * think_ratio))
    answer_tokens = max(1, max_new_tokens - think_tokens)

    # PASS 1: think until </think> (or budget)
    start_len1 = base_inputs.input_ids.shape[1]
    stop_think = StoppingCriteriaList([
        _StopOnText(tok, ["</think>"], start_len=start_len1, min_new_tokens=8)
    ])

    out1 = gen.generate(
        **base_inputs,
        max_new_tokens=think_tokens,
        do_sample=True,
        temperature=TEMPERATURE_QWEN3_THINKING,
        top_p=TOP_P_QWEN3_THINKING,
        top_k=TOP_K_QWEN3_THINKING,
        stopping_criteria=stop_think,
    )
    new_ids1 = out1[0][base_inputs.input_ids.shape[1]:].tolist()
    raw1 = tok.decode(new_ids1, skip_special_tokens=False)

    # extract thinking (closed preferred; else truncated)
    m_closed = re.search(r"<think>(.*?)</think>", raw1, flags=re.S)
    m_open   = re.search(r"<think>(.*)$",        raw1, flags=re.S)
    thinking = (m_closed.group(1) if m_closed else (m_open.group(1) if m_open else "")).strip()
    closed   = m_closed is not None
    if print_thinking and thinking:
        print(thinking)

    # If not closed, we still proceed but mark fallback for answer generation
    # PASS 2: final answer, reasoning disabled, with clear stops
    # We re-template with enable_thinking=False so the model won't reopen think mode.
    # Provide a small assistant cue to anchor the answer section.
    assist_cue = (f"<think>{thinking}</think>\nFinal Answer: " "Summarise your reasoning clearly in 2-3 sentences using the retrieved citations." if closed and thinking else "Final Answer:"
)
    messages2 = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": question},
        {"role": "assistant", "content": assist_cue},
        {"role": "user", "content": "/no_think Provide only the final answer."},
    ]
    text2 = tok.apply_chat_template(
        messages2, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs2 = tok([text2], return_tensors="pt").to(gen.device)

    start_len = inputs2.input_ids.shape[1]
    stop_answer = StoppingCriteriaList([
        _StopOnText(tok, answer_stops, start_len=start_len, min_new_tokens=12)
    ])

    out2 = gen.generate(
        **inputs2,
        max_new_tokens=answer_tokens,
        do_sample=True,
        temperature=TEMPERATURE_QWEN3_NON_THINKING,
        top_p=TOP_P_QWEN3_NON_THINKING,
        top_k=TOP_K_QWEN3_NON_THINKING,
        stopping_criteria=stop_answer,
    )

    new_ids2 = out2[0][inputs2.input_ids.shape[1]:].tolist()
    decoded2 = tok.decode(new_ids2, skip_special_tokens=False)
    

    # clean answer
    content = _THINK_BLOCK.sub("", decoded2).strip()
    content = _IM_END_TAIL.sub("", content).rstrip()
    # final safety: if answer somehow empty (rare), quick fallback single pass non-thinking
    if not content.strip():
        out_f = gen.generate(
            **base_inputs,
            max_new_tokens=answer_tokens,
            temperature=TEMPERATURE_QWEN3_NON_THINKING,
            top_p=TOP_P_QWEN3_NON_THINKING,
            top_k=TOP_K_QWEN3_NON_THINKING,
        )
        ids_f = out_f[0][base_inputs.input_ids.shape[1]:].tolist()
        content = tok.decode(ids_f, skip_special_tokens=False)
        content = _THINK_BLOCK.sub("", content).strip()
        content = _IM_END_TAIL.sub("", content).rstrip()

    return content, thinking
