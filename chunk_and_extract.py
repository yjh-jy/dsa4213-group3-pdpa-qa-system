#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
chunk_and_extract.py

Restored full chunk_and_extract pipeline with structured logging.

This file:
 - preserves the original parsing, structure extraction, token-aware chunking,
   canonical citations, reference detection, and corpus.jsonl output
 - adds logging statements at every major stage for observability

Usage:
    python chunk_and_extract.py build --input data/corpus/raw_pdpa.txt --output_dir data/corpus --filename corpus_subsection_v1
    
Dependencies:
    pip install python-docx beautifulsoup4 sentence-transformers transformers tqdm

"""

import os
import re
import sys
import json
import uuid
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from tqdm import tqdm

# parsers
try:
    from docx import Document
    import docx2txt
    
except Exception:
    DocxDocument = None

from bs4 import BeautifulSoup

# tokenization for token-aware chunking
from transformers import AutoTokenizer

# -----------------------------
# Logging configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# -----------------------------
# Configuration / heuristics
# -----------------------------

# Default tokenizer: compatible with sentence-transformers / E5 style models
DEFAULT_TOKENIZER = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking hyperparams
CHUNK_TOKEN_TARGET = 512
OVERLAP_TOKENS = 96
MAX_TOKENS_PER_SECTION = 5000
MIN_TOKENS_PER_CHUNK = 20

# Headings detection regexes (simple, adjustable)
PART_RE = re.compile(r"^PART\s+(\d+)", re.IGNORECASE)
# Capture section number (e.g. 1, 2A) and the rest of the line (which may be body text)
SECTION_RE = re.compile(r'^\s*(\d+[A-Z]?)\.\s*(.*)$', re.IGNORECASE)

# Subsection on its own line: "(1) rest of text..."
SUBSECTION_RE = re.compile(r"^\(\s*(\d+[A-Z]?)\)\s*(.*)$", re.IGNORECASE)

# Subsection inline following section marker, e.g. "—(1) text" or "(1) text"
# This will match leading dashes/spaces then "(n)" then optional remainder text.
SUBSECTION_INLINE_RE = re.compile(r"^[\-\—\–\s]*\(\s*(\d+)\s*\)\s*(.*)$", re.IGNORECASE)

# Fallback canonical citation format
CANON_FMT = "{law_short} s.{section}{subsection}"

# -----------------------------
# Utility helpers
# -----------------------------

def safe_text(s: Optional[str]) -> str:
    return (s or "").strip()


def gen_chunk_id(doc_id: str, part: Optional[str], section: Optional[str], subsection: Optional[str], idx: int) -> str:
    parts = [doc_id.replace(' ', '_')]
    if part:
        parts.append(f"Part{part}")
    if section:
        parts.append(f"Sec{section}")
    if subsection:
        parts.append(f"Sub{subsection}")
    parts.append(f"chunk{idx}")
    return "::".join(parts)


# -----------------------------
# Parsing into Parts
# -----------------------------


def split_raw_txt_into_blocks(raw_text: str, law_short: str = "PDPA") -> List[Dict]:
    """
    Parse raw legal text into structured units at the subsection level.
    
    - Each PART becomes a block
    - Sections are detected by numbers like '1.', '2A.', '3.—', but only if preceded by a likely title line
    - Subsections are strictly (1), (2), (3)
    - Expanded points like (a), (i) are appended inside subsection paragraphs
    """
    logger.info("Splitting text into Part-level blocks...")
    
    # --- Step 1: Split into Part blocks ---
    part_blocks: List[Dict] = []
    current_block_lines: List[str] = []
    
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if PART_RE.match(line) and line != 'PART 1':
            if current_block_lines:
                part_blocks.append({"text": "\n".join(current_block_lines)})
            current_block_lines = [line]
        else:
            current_block_lines.append(line)
    
    if current_block_lines:
        part_blocks.append({"text": "\n".join(current_block_lines)})
    
    logger.info(f"Found {len(part_blocks)} Parts")

    return part_blocks

# -----------------------------
# Parsing from Parts to Subsections
# -----------------------------

def parse_parts_to_subsections(part_blocks: List[Dict], law_short: str = "PDPA") -> List[Dict]:
    """
    For each Part block in `part_blocks` (each block['text'] is the full text of a Part),
    parse it into subsection-level units.

    Returns a list of dicts:
      {
        "law_short": ...,
        "part": "1",
        "section": "2A",
        "section_title": "Interpretation",
        "subsection": "1",          # "0" if no explicit subsection exists
        "paragraphs": [... lines ...]
      }
    """
    out_subsections: List[Dict] = []

    for block in part_blocks:
        # split and keep all lines (preserve order), strip trailing spaces
        raw_lines = [ln.rstrip() for ln in block.get("text", "").splitlines()]
        lines = [ln for ln in raw_lines if ln.strip() != ""]

        # Determine part id from the first few lines (if any)
        cur_part = None
        for ln in lines[:3]:
            m = PART_RE.match(ln.strip())
            if m:
                cur_part = m.group(1)
                break

        # state
        cur_section = None
        cur_section_title = None
        cur_subsection = None
        cur_paras: List[str] = []
        prev_nonempty = None  # last non-empty line seen (for title detection)

        def flush_current():
            nonlocal cur_paras
            if cur_section is None:
                return
            cur_sub_id = cur_subsection if cur_subsection is not None else "0"
            if cur_paras:
                out_subsections.append({
                    "law_short": law_short,
                    "part": cur_part,
                    "section": cur_section,
                    "section_title": cur_section_title,
                    "subsection": cur_sub_id,
                    "paragraphs": cur_paras.copy()
                })
            cur_paras = []

        i = 0
        while i < len(lines):
            ln = lines[i].strip()
            ln = ln.replace("\t", " ")

            # check for explicit PART header
            m = PART_RE.match(ln)
            if m:
                flush_current()
                cur_part = m.group(1)
                cur_section = None
                cur_section_title = None
                cur_subsection = None
                cur_paras = []
                prev_nonempty = ln
                i += 1
                continue

            # check for SECTION at start of line (may include remainder)
            m = SECTION_RE.match(ln)
            if m:
                # flush any previous subsection
                flush_current()

                cur_section = m.group(1)
                remainder = m.group(2).strip()  # remainder after "2." on same line

                # section title is the previous non-empty line, if available and not a PART header
                if prev_nonempty and not PART_RE.match(prev_nonempty):
                    cur_section_title = prev_nonempty
                else:
                    cur_section_title = None

                # Now check if remainder contains an inline subsection like "—(1) text" or "(1) text"
                inline_m = SUBSECTION_INLINE_RE.match(remainder)
                if inline_m:
                    # treat as subsection start immediately
                    cur_subsection = inline_m.group(1)
                    inline_rest = inline_m.group(2).strip()
                    cur_paras = []
                    if inline_rest:
                        cur_paras.append(inline_rest)
                else:
                    # start default subsection "0" and include remainder if present
                    cur_subsection = "0"
                    cur_paras = []
                    if remainder:
                        cur_paras.append(remainder)
                prev_nonempty = ln
                i += 1
                continue

            # check for SUBSECTION on its own line, e.g., "(1) rest..."
            m = SUBSECTION_RE.match(ln)
            if m:
                # flush any existing subsection content
                flush_current()

                cur_subsection = m.group(1)
                remainder = m.group(2).strip()
                cur_paras = []
                if remainder:
                    cur_paras.append(remainder)
                prev_nonempty = ln
                i += 1
                continue

            # Otherwise the line is paragraph content — append to current subsection (or create default)
            if cur_section is None:
                # No section detected yet — buffer prev_nonempty and skip until first section found
                prev_nonempty = ln
                i += 1
                continue

            if cur_subsection is None:
                cur_subsection = "0"
            cur_paras.append(ln)
            prev_nonempty = ln
            i += 1

        # end of lines for this part: flush remaining
        flush_current()

    logger.info(f"Parsed {len(out_subsections)} subsection-level units from {len(part_blocks)} parts")
    return out_subsections

# -----------------------------
# Chunking (token-aware) and metadata assembly
# -----------------------------


SENTENCE_SPLIT_RE = re.compile(r'(?<=[\.\?!\n])\s+')

def gen_chunk_id(law_short, part, section, subsection, idx):
    # placeholder: replace with your real id generation
    return f"{law_short}-{section or 'sec'}-{subsection or ''}-{idx}"

CANON_FMT = "{law_short} s.{section}{subsection}"

def chunk_section_texts(sections: List[Dict],
                                 tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                                 law_short: str = "PDPA") -> List[Dict]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    out_chunks = []
    chunk_idx = 0

    for sec in tqdm(sections, desc="Chunking sections"):
        part = sec.get("part")
        section = sec.get("section")
        subsection = sec.get("subsection")
        title = sec.get("section_title") or ""
        paras = sec.get("paragraphs", [])  # expected list[str]

        # assemble full text with a single header
        header = (title.strip() + "\n") if title else ""
        full_text = header + "\n".join(paras)
        if not full_text.strip():
            continue

        # quick token check for entire section
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if len(full_ids) <= CHUNK_TOKEN_TARGET:
            # single chunk for small sections
            canonical = CANON_FMT.format(law_short=law_short, section=(section or ""), subsection=("("+subsection+")") if subsection else "")
            out_chunks.append({
                "part": part, "section": section, "subsection": subsection, "title": title,
                "canonical_citation": canonical, "text": full_text.strip(),
                "token_count": len(full_ids),
                "start_pos": 0, "end_pos": len(full_text),
                "references": [], "source": f"{law_short}-{section}"
            })
            continue

        # If section huge, split into paragraph-level pieces first (but keep header only on first piece)
        # Build pieces as text blocks composed of sentences (avoid cutting sentences)
        # We'll split paragraphs into sentences and group sentences to reach token targets.
        # Pre-split all paragraphs into sentences.
        sentences = []
        for p in paras:
            p = p.strip()
            if not p:
                continue
            sents = SENTENCE_SPLIT_RE.split(p)
            # keep original paragraph boundary marker by appending newline to last sentence in paragraph
            if sents:
                sents = [s.strip() for s in sents if s.strip()]
                if sents:
                    sents[-1] = sents[-1] + "\n"
                    sentences.extend(sents)

        # batch-encode sentences to get token lengths (speeds up counting)
        # tokenizer can process a list
        enc = tokenizer(sentences, add_special_tokens=False)
        sent_token_counts = [len(ids) for ids in enc["input_ids"]]

        # now assemble chunks by adding sentences until near CHUNK_TOKEN_TARGET
        pieces = []
        current_piece_sents = []
        current_piece_tokens = 0
        for sent, toks in zip(sentences, sent_token_counts):
            # if a single sentence is very large (> CHUNK_TOKEN_TARGET), split by token-slice fallback
            if toks >= CHUNK_TOKEN_TARGET:
                # flush current piece first
                if current_piece_sents:
                    pieces.append("".join(current_piece_sents).strip())
                    current_piece_sents = []
                    current_piece_tokens = 0
                # break the long sentence into token slices (decode safe)
                sent_ids = tokenizer(sent, add_special_tokens=False)["input_ids"]
                start = 0
                while start < len(sent_ids):
                    end = min(start + CHUNK_TOKEN_TARGET, len(sent_ids))
                    slice_text = tokenizer.decode(sent_ids[start:end], skip_special_tokens=True).strip()
                    pieces.append(slice_text)
                    start = max(0, end - OVERLAP_TOKENS)
                continue

            if current_piece_tokens + toks > CHUNK_TOKEN_TARGET:
                # flush current piece
                pieces.append("".join(current_piece_sents).strip())
                current_piece_sents = [sent]
                current_piece_tokens = toks
            else:
                current_piece_sents.append(sent)
                current_piece_tokens += toks

        if current_piece_sents:
            pieces.append("".join(current_piece_sents).strip())

        # Now produce final token-overlapped chunks from each piece (sliding window)
        section_char_index = 0  # track char offsets inside full_text for start/end mapping
        # compute full_text positions mapping to help offsets: we'll just use find to locate piece occurrence.
        for i_piece, piece in enumerate(pieces):
            if not piece:
                continue
            # ensure header only on first piece (attach header to first piece's first chunk)
            attach_header = (i_piece == 0)
            piece_text = (header + piece) if attach_header and header else piece

            # token ids for piece
            piece_enc = tokenizer(piece_text, add_special_tokens=False)
            toks = piece_enc["input_ids"]
            n = len(toks)
            if n <= CHUNK_TOKEN_TARGET:
                # small piece -> single chunk
                chunk_text = piece_text.strip()
                # find char offsets (first occurrence) - ok because pieces are from full_text in order
                # try to find next occurrence from section_char_index to preserve order
                start_pos = full_text.find(chunk_text, section_char_index)
                if start_pos == -1:
                    start_pos = None
                    end_pos = None
                else:
                    end_pos = start_pos + len(chunk_text)
                    section_char_index = end_pos
                canonical = CANON_FMT.format(law_short=law_short, section=(section or ""), subsection=("("+subsection+")") if subsection else "")
                out_chunks.append({
                    "part": part, "section": section, "subsection": subsection, "title": title if attach_header else "",
                    "canonical_citation": canonical, "text": chunk_text,
                    "token_count": n,
                    "start_pos": start_pos, "end_pos": end_pos,
                    "references": [], "source": f"{law_short}-{section}"
                })
                chunk_idx += 1
                continue

            # sliding window
            start = 0
            while start < n:
                end = min(start + CHUNK_TOKEN_TARGET, n)
                slice_ids = toks[start:end]
                chunk_text = tokenizer.decode(slice_ids, skip_special_tokens=True).strip()
                # attach header only to the very first chunk of first piece
                if start == 0 and attach_header and title:
                    chunk_text = title.strip() + "\n" + chunk_text

                # find offsets (best effort) starting from section_char_index
                search_from = section_char_index if section_char_index is not None else 0
                start_pos = full_text.find(chunk_text, search_from)
                if start_pos == -1:
                    # fallback: attempt find without header
                    alt = chunk_text
                    if title and chunk_text.startswith(title.strip() + "\n"):
                        alt = chunk_text[len(title.strip())+1:]
                    start_pos = full_text.find(alt, search_from)
                    if start_pos == -1:
                        start_pos = None
                        end_pos = None
                    else:
                        end_pos = start_pos + len(alt)
                        section_char_index = end_pos
                else:
                    end_pos = start_pos + len(chunk_text)
                    section_char_index = end_pos

                canonical = CANON_FMT.format(law_short=law_short, section=(section or ""), subsection=("("+subsection+")") if subsection else "")
                out_chunks.append({
                    "part": part, "section": section, "subsection": subsection,
                    "title": title if (start == 0 and attach_header) else "",
                    "canonical_citation": canonical, "text": chunk_text,
                    "token_count": len(slice_ids),
                    "start_pos": start_pos, "end_pos": end_pos,
                    "references": [], "source": f"{law_short}-{section}"
                })
                chunk_idx += 1
                if end == n:
                    break
                start = max(0, end - OVERLAP_TOKENS)

        # optional: merge last tiny chunk into previous one (reduce tiny fragments)
        if len(out_chunks) >= 2 and out_chunks[-1]["token_count"] < MIN_TOKENS_PER_CHUNK:
            tiny = out_chunks.pop()
            prev = out_chunks.pop()
            merged_text = (prev["text"] + "\n" + tiny["text"]).strip()
            merged_ids = tokenizer(merged_text, add_special_tokens=False)["input_ids"]
            prev["text"] = merged_text
            prev["token_count"] = len(merged_ids)
            prev["end_pos"] = tiny["end_pos"]
            out_chunks.append(prev)

    # assign chunk_ids
    for i, c in enumerate(out_chunks):
        c["chunk_id"] = gen_chunk_id(law_short, c.get("part"), c.get("section"), c.get("subsection"), i)

    return out_chunks

# -----------------------------
# Cross-ref detection (heuristic-based)
# -----------------------------


def detect_references_per_text(text: str, section_name: str, law_name: str = "PDPA") -> List[str]:
    """
    Detect references in `text` and return canonical strings like:
      - "PDPA Part 3"
      - "PDPA s.4B(1)(a)"
      - "PDPA s.{section_name}(3)", etc.
    Uses `section_name` when the textual fragment gives only subsection parentheses like "(3)".
    """
    seen = []
    def add(item):
        if item not in seen:
            seen.append(item)

    txt = text

    # 1) PARTS: e.g. "Parts 3, 4, 5, 6, 6A, and 6B"
    # parts_pattern = re.compile(r'\b[Pp]arts?\s+([0-9A-Za-z\-\s,ands]+?)(?:[\.;:]|$)', re.UNICODE)
    # for m in parts_pattern.finditer(txt):
    #     capture = m.group(1)
    #     tokens = re.split(r'\s*(?:,|and|or)\s*', capture)
    #     for t in tokens:
    #         t = t.strip()
    #         if not t:
    #             continue
    #         if '-' in t:
    #             a,b = [x.strip() for x in t.split('-',1)]
    #             try:
    #                 ai = int(re.match(r'\d+', a).group(0))
    #                 bi = int(re.match(r'\d+', b).group(0))
    #                 for n in range(ai, bi+1):
    #                     add(f"{law_name} Part {n}")
    #             except Exception:
    #                 add(f"{law_name} Part {t}")
    #         else:
    #             add(f"{law_name} Part {t}")

    # 2) SECTIONS (plural/combined) - new: handles "sections 14(1)(a) and 18(b)" and "section 15 or 15A"
    sections_group_pattern = re.compile(
        r'\b[Ss]ections?\b\s*'                             # 'section' or 'sections'
        r'('
        r'(?:[0-9]+[A-Za-z]?(?:\(\s*[0-9A-Za-z]+\s*\))*)'   # a section id with optional parenthesis groups
        r'(?:\s*(?:,|and|or)\s*'                           # separators
        r'(?:[0-9]+[A-Za-z]?(?:\(\s*[0-9A-Za-z]+\s*\))*)'
        r')*'
        r')'
    )
    for m in sections_group_pattern.finditer(txt):
        group = m.group(1)
        # extract each section-like token (keeps trailing parenthetical groups)
        sect_tokens = re.findall(r'([0-9]+[A-Za-z]?(?:\(\s*[0-9A-Za-z]+\s*\))*)', group)
        for tok in sect_tokens:
            # split number/letters from parentheses
            match = re.match(r'^([0-9]+[A-Za-z]?)(.*)$', tok)
            if match:
                sect_id = match.group(1)
                parens = match.group(2) or ""
                # normalize parentheses spacing
                parens_clean = ''.join(re.findall(r'\(\s*([0-9A-Za-z]+)\s*\)', parens))
                if parens:
                    # re-extract tokens to keep structure (e.g. (1)(a))
                    paren_list = re.findall(r'\(\s*([0-9A-Za-z]+)\s*\)', parens)
                    tail = ''.join(f"({p})" for p in paren_list)
                    add(f"{law_name} s.{sect_id}{tail}")
                else:
                    add(f"{law_name} s.{sect_id}")

    # 3) EXPLICIT SECTION with possible parentheses right after the word 'Section'
    #    (covers "Section 4B(1)(a)" and also "Section 15 or 15A" where only one id followed - the plural pattern handles multi-ids)
    section_full_pattern = re.compile(
        r'\b[Ss]ection\s+'             # 'Section'
        r'([0-9]+[A-Za-z]?)'           # single section id like 4B or 15
        r'((?:\(\s*[0-9A-Za-z]+\s*\))*)'  # zero or more (...) groups
    )
    for m in section_full_pattern.finditer(txt):
        sect = m.group(1)
        parens = m.group(2) or ""
        tokens = re.findall(r'\(\s*([0-9A-Za-z]+)\s*\)', parens)
        if not tokens:
            add(f"{law_name} s.{sect}")
        else:
            tail = ''.join(f"({t})" for t in tokens)
            add(f"{law_name} s.{sect}{tail}")

    # 4) SUBSECTIONS and nested parenthetical groups after words 'subsection' or 'subsections'
    subsection_pattern = re.compile(
        r'\b[Ss]ubsection[s]?\b\s*'                       # 'subsection' or 'subsections'
        r'((?:\(\s*[0-9A-Za-z]+\s*\)(?:\s*(?:,|and|or)?\s*)?)+)',  # one or more parenthesis groups possibly separated by commas/and/or
    )
    for m in subsection_pattern.finditer(txt):
        paren_seq = m.group(1)
        tokens = re.findall(r'\(\s*([0-9A-Za-z]+)\s*\)', paren_seq)
        cur_numeric = None
        out_tmp = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if re.match(r'^\d+[A-Za-z]*$', tok):  # numeric or numeric+alpha (like 6A)
                if i + 1 < len(tokens) and re.match(r'^[A-Za-z][A-Za-z0-9]*$', tokens[i+1]):
                    out_tmp.append(f"{law_name} s.{section_name}({tok})({tokens[i+1]})")
                    i += 2
                else:
                    out_tmp.append(f"{law_name} s.{section_name}({tok})")
                    i += 1
            else:
                # alpha token without preceding number -> attach to last numeric if exists else to provided section_name
                j = i-1
                prev_num = None
                while j >= 0:
                    if re.match(r'^\d+[A-Za-z]*$', tokens[j]):
                        prev_num = tokens[j]
                        break
                    j -= 1
                if prev_num:
                    out_tmp.append(f"{law_name} s.{section_name}({prev_num})({tok})")
                else:
                    out_tmp.append(f"{law_name} s.{section_name}({tok})")
                i += 1
        for r in out_tmp:
            add(r)

    # 5) fallback single 'subsection (2)' if missed above
    fallback_single_sub = re.compile(r'\b[Ss]ubsection\b\s*\(\s*([0-9A-Za-z]+)\s*\)')
    for m in fallback_single_sub.finditer(txt):
        num = m.group(1)
        add(f"{law_name} s.{section_name}({num})")

    return seen


def detect_references(chunks: List[Dict]) -> None:
    """Populate 'references' in-place by heuristically finding 'see Section X' patterns and mapping them to chunk_ids."""
    logger.info("Detecting cross-references among chunks...")

    count_refs = 0
    for c in chunks:
        text = c.get("text", "")
        c['references'] = detect_references_per_text(text, section_name=c.get("section"))
        count_refs += len(c['references'])
    logger.info(f"Detected {count_refs} cross-reference links across chunks")


# -----------------------------
# Writer: produce corpus.jsonl
# -----------------------------

def write_corpus_jsonl(chunks: List[Dict], output_dir: str, file_name: str, doc_id: str):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{file_name}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            line = {
                "doc_id": doc_id,
                "chunk_id": c.get("chunk_id"),
                "law_short": doc_id,
                "part": c.get("part"),
                "section": c.get("section"),
                "subsection": c.get("subsection"),
                "canonical_citation": c.get("canonical_citation"),
                "text": c.get("text"),
                "token_count": c.get("token_count"),
                "references": c.get("references", [])
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(chunks)} chunks to {out_path}")


# -----------------------------
# CLI
# -----------------------------

def build_from_file(input_path: str, output_dir: str, file_name: str, law_short: str, tokenizer_name: str):
    p = Path(input_path)
    if not p.exists():
        logger.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(input_path)
    ext = p.suffix.lower()
    if ext in [".docx", ".doc"]:
        logger.info(f"Parsing DOCX input: {input_path}")
        # blocks = parse_docx(str(p))
    else:
        logger.info(f"Parsing plain text input: {input_path}")
        text = p.read_text(encoding='utf-8')

    blocks = split_raw_txt_into_blocks(text)
    logger.info(f"Parsed {len(blocks)} blocks from {input_path}")
    sections = parse_parts_to_subsections(blocks, law_short=law_short)
    logger.info(f"Extracted {len(sections)} structured sections/subsections")
    chunks = chunk_section_texts(sections, tokenizer_name=tokenizer_name, law_short=law_short)
    detect_references(chunks)
    write_corpus_jsonl(chunks, output_dir, file_name=file_name, doc_id=law_short)

def main():
    parser = argparse.ArgumentParser(description="Chunk legal doc and produce corpus.jsonl with metadata")
    sub = parser.add_subparsers(dest='cmd', required=True)

    pb = sub.add_parser('build', help='Build corpus.jsonl from a docx/html/plaintext file')
    pb.add_argument('--input', required=True, help='Path to input file (.docx, .html, .txt)')
    pb.add_argument('--output_dir', required=True, help='Output folder for corpus.jsonl')
    pb.add_argument('--filename', required=True, help='Name for the corpus')
    pb.add_argument('--law_short', default='PDPA', help='Short name for canonical citations (e.g., PDPA)')
    pb.add_argument('--tokenizer', default=DEFAULT_TOKENIZER, help='Tokenizer / model name for token-aware chunking')

    args = parser.parse_args()
    if args.cmd == 'build':
        logger.info(f"Starting build process for {args.input} (law_short={args.law_short})")
        build_from_file(args.input, args.output_dir, args.filename, args.law_short, args.tokenizer)
        logger.info("Build completed successfully.")


if __name__ == '__main__':
    main()
