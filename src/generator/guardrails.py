import re
from typing import Any, Dict, List
from config import *

# ---------------------------- Guardrails helper functions------------------------

def sentence_split(s: str) -> List[str]:
    return [t for t in re.split(r"(?<=[\.!?])\s+", s.strip()) if t]


def citation_coverage(answer: str) -> float:
    sents = sentence_split(answer)
    if not sents:
        return 0.0
    covered = sum(1 for s in sents if CITE_RE.search(s))
    return covered / len(sents)

def extract_citations(answer: str) -> List[str]:
    """
    Extract only explicit 'PDPA s.xx(x)' style citations,
    regardless of whether they are bracketed or not.
    """
    cites = set()
    for m in CITE_RE.finditer(answer):
        c = m.group(0).strip("[]").strip()
        # Normalize spacing and prefix
        c = re.sub(r"\s+", " ", c)
        c = c.replace("PDPA s. ", "PDPA s.")
        cites.add(c)
    return sorted(cites)


def early_abstain(scores, chosen_ids: List[str]) -> bool:
    if not chosen_ids:
        return True
    return max(scores) < TAU_RETR

def decide_abstain(rrf_top: float,
                   rrf_margin: float,
                   ce_top: float,
                   ce_margin: float,
                   thresholds: dict):
    """
    Returns (abstain, reason_tag)
    """
    if rrf_top < thresholds["rrf_top"] or rrf_margin < thresholds["rrf_margin"]:
        return True, "retrieval_low_confidence"
    if ce_top < thresholds["ce_top"] or ce_margin < thresholds["ce_margin"]:
        return True, "rerank_low_confidence"

    return False, ""
