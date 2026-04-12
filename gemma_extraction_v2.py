# -*- coding: utf-8 -*-
"""
gemma_extraction_v3_gemma_compatible.py

Stable entity/relation extraction for Studium Parisiense notices with
Google GenAI + Gemma (`gemma-3-27b-it`).

Compared with earlier versions:
- NO JSON mode (`response_mime_type`) because Gemma 3 27B IT rejects it
- smaller prompts for more stable JSON output
- recursive re-splitting of failing chunks
- explicit handling of finish reasons such as MAX_TOKENS and RECITATION
- chunk-level reflection by default, whole-record reflection optional
- checkpoint + append-only outputs
- canonical name normalization + chunk merge

Usage:
    export GEMINI_API_KEY="..."
    python gemma_extraction_v3_gemma_compatible.py \
        --input-file test_suite.jsonl \
        --entities-file entities_v5.jsonl \
        --relations-file relations_v5.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import List

from google import genai
from google.genai import types as genai_types

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_NAME = "gemma-3-27b-it"
DEFAULT_INPUT_FILE = "test_suite.jsonl"
DEFAULT_ENTITIES_FILE = "entities_v5.jsonl"
DEFAULT_RELATIONS_FILE = "relations_v5.jsonl"
DEFAULT_CHECKPOINT_FILE = "checkpoint_v5.json"

DEFAULT_MAX_RETRIES = 4
DEFAULT_BASE_SLEEP = 4.0
DEFAULT_TOKEN_BUDGET_PER_MIN = 14000
DEFAULT_TARGET_PROMPT_TOKENS = 2200
DEFAULT_MAX_PROMPT_TOKENS = 3000
DEFAULT_REFLECT_MAX_PROMPT_TOKENS = 2600
DEFAULT_MAX_CHUNK_CHARS = 3500
DEFAULT_MIN_CHUNK_CHARS = 1200
DEFAULT_MAX_SPLIT_DEPTH = 5
DEFAULT_CHARS_PER_TOKEN = 4
DEFAULT_MAX_HINT_ITEMS = 12
DEFAULT_MAX_EVIDENCE = 8
DEFAULT_RELATION_NOTE_MAX = 300
DEFAULT_RECORD_REFLECTION_MAX_TEXT_CHARS = 5000

# ──────────────────────────────────────────────────────────────────────────────
# Relation enum
# ──────────────────────────────────────────────────────────────────────────────
RELATION_TYPES = [
    "STUDIED_AT", "TAUGHT_AT", "OBTAINED_DEGREE", "LECTURED_AT",
    "LECTURED_SUBJECT", "ENROLLED_IN", "AFFILIATED_WITH", "MEMBER_OF",
    "BELONGS_TO", "FOUNDED", "HELD_POSITION_IN", "SERVED", "PHYSICIAN_OF",
    "BORN_IN", "DIED_IN", "ACTIVE_IN", "FROM_DIOCESE", "STUDENT_OF",
    "MENTOR_OF", "DEDICATED_TO", "ADDRESSED", "OPPOSED_TO",
    "COLLABORATED_WITH", "SPOUSE_OF", "PARENT_OF", "SIBLING_OF",
    "ATTESTED_BY", "AUTHORED", "TRANSLATED",
]
RELATION_SET = set(RELATION_TYPES)
RELATION_ENUM_STR = ", ".join(RELATION_TYPES)

RELATION_ALIASES = {
    "TEACHES_AT": "TAUGHT_AT",
    "TAUGHTS_AT": "TAUGHT_AT",
    "TEACHING_AT": "TAUGHT_AT",
    "STUDY_AT": "STUDIED_AT",
    "STUDIES_AT": "STUDIED_AT",
    "STUDIEDIN": "STUDIED_AT",
    "LECTURES_AT": "LECTURED_AT",
    "LECTUREDIN": "LECTURED_AT",
    "MEMBER": "MEMBER_OF",
    "BELONGS": "BELONGS_TO",
    "AUTHOR_OF": "AUTHORED",
    "WRITES": "AUTHORED",
    "WRITTEN_BY": "AUTHORED",
    "TRANSLATOR_OF": "TRANSLATED",
    "MARRIED_TO": "SPOUSE_OF",
    "CHILD_OF": "PARENT_OF",
    "BROTHER_OF": "SIBLING_OF",
    "SISTER_OF": "SIBLING_OF",
    "COLLABORATES_WITH": "COLLABORATED_WITH",
}

# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────
EXTRACTION_PROMPT = (
    "Tu es un extracteur d'information specialise dans les corpus historiques medievaux.\n"
    "Tu recois un FRAGMENT d'une notice biographique du corpus Studium Parisiense.\n\n"
    "TACHE : extraire toutes les entites et toutes les relations presentes dans ce fragment, sans invention.\n\n"
    "REGLES ABSOLUES :\n"
    "- Reponds UNIQUEMENT par un objet JSON valide, sans markdown ni explication.\n"
    "- L'entite principale DOIT avoir l'id exact : \"subject_person\".\n"
    "- Les autres ids doivent etre stables, courts, uniques et en snake_case.\n"
    "- Les relations doivent utiliser UNIQUEMENT les ids presents dans entities.\n"
    "- Types de relations autorises (liste FERMEE) : " + RELATION_ENUM_STR + "\n"
    "- N'invente jamais un type de relation hors de cette liste.\n"
    "- evidence = extraits verbatim du fragment.\n"
    "- Si une information est absente, n'invente rien.\n"
    "- Si tu n'es pas sur, baisse confidence.\n\n"
    "TYPES D'ENTITES suggeres : PERSON, PLACE, UNIVERSITY, INSTITUTION, DIOCESE, DEGREE, ROLE, DATE, SOURCE, WORK, NATION.\n\n"
    "FORMAT DE SORTIE :\n"
    "{\n"
    '  "record_id": "<reference>",\n'
    '  "subject": "<nom canonique>",\n'
    '  "entities": [\n'
    '    {"id": "subject_person", "name": "<nom>", "type": "PERSON", "confidence": 0.0, "evidence": ["<citation>"]}\n'
    "  ],\n"
    '  "relations": [\n'
    '    {"source": "<id>", "target": "<id>", "type": "<TYPE>", "confidence": 0.0, "evidence": ["<citation>"], "attributes": {"date": "", "note": ""}}\n'
    "  ]\n"
    "}\n\n"
    "FRAGMENT :\n__DATA__\n"
)

REFLECTION_PROMPT = (
    "Tu es un verificateur d'extraction d'information medievale.\n\n"
    "Tu recois :\n"
    "1. Le FRAGMENT SOURCE original\n"
    "2. Une EXTRACTION INITIALE JSON\n\n"
    "TACHE : produire une extraction corrigee et complete du FRAGMENT.\n"
    "Ne traite QUE ce fragment, pas une notice entiere.\n\n"
    "CORRIGER :\n"
    "- entites manquantes\n"
    "- noms incomplets\n"
    "- relations manquantes\n"
    "- types de relations incorrects\n\n"
    "REGLES :\n"
    "- Reponds UNIQUEMENT par un objet JSON valide.\n"
    "- Conserve tous les ids existants si possible.\n"
    "- subject_person ne change jamais d'id.\n"
    "- Types de relations autorises : " + RELATION_ENUM_STR + "\n"
    "- N'invente rien.\n"
    "- evidence = citations verbatim du fragment.\n\n"
    "FRAGMENT SOURCE :\n__SOURCE__\n\n"
    "EXTRACTION INITIALE :\n__EXTRACTION__\n\n"
    "RENVOIE L'OBJET JSON CORRIGE :\n"
)

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def normalize_key(text: str) -> str:
    text = normalize_space(text).lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[-\s]+", "_", text)
    return text.strip("_")


def slugify(text: str, fallback: str = "item") -> str:
    s = normalize_key(text)
    return s[:80] if s else fallback


def unique_preserve_order(items, max_items=None):
    seen, out = set(), []
    for item in items:
        item = normalize_space(item)
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
        if max_items and len(out) >= max_items:
            break
    return out


def fsync_and_flush(fh):
    fh.flush()
    try:
        os.fsync(fh.fileno())
    except Exception:
        pass


def safe_json_load(text: str):
    try:
        return json.loads(text)
    except Exception:
        try:
            import json_repair  # type: ignore
            return json_repair.loads(text)
        except Exception:
            return None


def extract_first_json_object(text: str):
    depth = 0
    in_str = False
    esc = False
    start = None
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i + 1]
    return None


def parse_retry_delay_seconds(err, attempt):
    s = str(err)
    m = re.search(r"retryDelay.*?(\d+)s", s) or re.search(r"retry in ([\d.]+)s", s)
    return int(float(m.group(1))) + 3 if m else min(60, 2 ** attempt)


def load_checkpoint(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_line": -1, "processed_ids": []}


def save_checkpoint(path, state):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def build_hint_lines(parsed_line):
    meta = parsed_line.get("meta_entities") or {}
    hints = []
    if meta.get("names"):
        hints.append("Variantes : " + ", ".join(unique_preserve_order(meta["names"], DEFAULT_MAX_HINT_ITEMS)))
    if meta.get("places"):
        hints.append("Lieux : " + ", ".join(unique_preserve_order(meta["places"], DEFAULT_MAX_HINT_ITEMS)))
    if meta.get("institutions"):
        hints.append("Institutions : " + ", ".join(unique_preserve_order(meta["institutions"], DEFAULT_MAX_HINT_ITEMS)))
    return hints


def build_slim_components(parsed_line):
    text_body = normalize_space(parsed_line.get("text", ""))
    subject_hint = normalize_space(parsed_line.get("name") or parsed_line.get("subject") or "")
    return text_body, build_hint_lines(parsed_line), subject_hint


def build_fragment_payload(record_id, subject_hint, text_chunk, hint_lines, chunk_index, total_chunks):
    parts = [f"record_id: {record_id}", f"chunk: {chunk_index + 1}/{total_chunks}"]
    if subject_hint:
        parts.append(f"subject_hint: {subject_hint}")
    if hint_lines:
        parts.append("hints:\n- " + "\n- ".join(hint_lines))
    parts.append("text:\n" + text_chunk)
    return "\n\n".join(parts)


def split_long_text(text, max_chunk_chars=DEFAULT_MAX_CHUNK_CHARS):
    text = text.strip()
    if not text:
        return [""]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()] or [text]
    chunks, current, current_len = [], [], 0

    def flush():
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current).strip())
            current[:] = []
            current_len = 0

    for para in paragraphs:
        if len(para) > max_chunk_chars:
            flush()
            sentences = re.split(r"(?<=[.!?;:])\s+", para)
            buf, buf_len = [], 0
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                add_len = len(sent) + 1
                if buf and buf_len + add_len > max_chunk_chars:
                    chunks.append(" ".join(buf).strip())
                    buf, buf_len = [sent], len(sent)
                else:
                    buf.append(sent)
                    buf_len += add_len
            if buf:
                chunks.append(" ".join(buf).strip())
            continue
        add_len = len(para) + 2
        if current and current_len + add_len > max_chunk_chars:
            flush()
        current.append(para)
        current_len += add_len

    flush()
    return chunks or [text]


def split_text_in_half(text: str) -> List[str]:
    text = text.strip()
    if len(text) <= DEFAULT_MIN_CHUNK_CHARS:
        return [text]
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) >= 2:
        mid = max(1, len(paras) // 2)
        return ["\n\n".join(paras[:mid]).strip(), "\n\n".join(paras[mid:]).strip()]
    mid = len(text) // 2
    left = text[:mid].rsplit(" ", 1)[0].strip() or text[:mid].strip()
    right = text[len(left):].strip()
    return [left, right] if left and right else [text]


def make_client(api_key=None):
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY environment variable missing.")
    return genai.Client(api_key=key)


def count_tokens_safe(client, model_name, contents):
    try:
        resp = client.models.count_tokens(model=model_name, contents=contents)
        total = getattr(resp, "total_tokens", None)
        if isinstance(total, int) and total > 0:
            return total
    except Exception:
        pass
    return max(1, len(contents) // DEFAULT_CHARS_PER_TOKEN)


def safe_response_text(response) -> str:
    text = getattr(response, "text", None)
    return text if isinstance(text, str) else ""


def response_finish_reason(response) -> str:
    try:
        cands = getattr(response, "candidates", None) or []
        if cands:
            fr = getattr(cands[0], "finish_reason", None)
            return str(fr) if fr is not None else ""
    except Exception:
        pass
    return ""


def paced_generate_content(client, model_name, prompt, max_retries, base_sleep, token_budget_per_min):
    tokens = count_tokens_safe(client, model_name, prompt)
    wait = max(base_sleep, 60.0 * tokens / max(1, token_budget_per_min))
    if wait > base_sleep:
        print(f"  [token guard] {tokens} tokens -> sleep {wait:.1f}s")
    time.sleep(wait)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.3,
                    top_p=1.0,
                    candidate_count=1,
                    max_output_tokens=4096,
                ),
            )
            return response
        except Exception as e:
            last_error = e
            print(f"  attempt {attempt} failed: {e}")
            if attempt >= max_retries:
                break
            backoff = parse_retry_delay_seconds(e, attempt)
            print(f"   -> retry in {backoff}s")
            time.sleep(backoff)

    raise RuntimeError(f"Failed after {max_retries} attempts: {last_error}")


def parse_model_output_from_response(response):
    raw_text = safe_response_text(response)
    finish_reason = response_finish_reason(response).upper()

    if finish_reason.endswith("RECITATION"):
        raise RuntimeError("RECITATION")
    if finish_reason.endswith("MAX_TOKENS"):
        raise RuntimeError("MAX_TOKENS")
    if finish_reason.endswith("MALFORMED_RESPONSE"):
        raise RuntimeError("MALFORMED_RESPONSE")

    cleaned = re.sub(r"```(?:json)?", "", raw_text, flags=re.IGNORECASE).strip()
    obj = safe_json_load(cleaned)
    if obj is not None:
        return obj

    cand = extract_first_json_object(cleaned)
    tail = f"\n[finish_reason={finish_reason}]" if finish_reason else ""
    if cand is None:
        raise ValueError(f"No JSON found:{tail}\n{cleaned[:1200]}")

    obj = safe_json_load(cand)
    if obj is None:
        raise ValueError(f"JSON parse failed:{tail}\n{cand[:1200]}")
    return obj


def normalize_relation_type(rel_type):
    rt = re.sub(r"[^A-Z_]", "_", normalize_space(rel_type).upper())
    rt = re.sub(r"_+", "_", rt).strip("_")
    rt = RELATION_ALIASES.get(rt, rt)
    return rt if rt in RELATION_SET else None


def sanitize_entity(ent):
    name = normalize_space(str(ent.get("name", "")))
    if not name:
        return None
    ent_type = normalize_space(str(ent.get("type", ""))).upper() or "ENTITY"
    ent_id = normalize_space(str(ent.get("id", ""))) or slugify(f"{name}_{ent_type}")
    try:
        confidence = max(0.0, min(1.0, float(ent.get("confidence", 0.5) or 0.5)))
    except Exception:
        confidence = 0.5
    return {
        "id": ent_id,
        "name": name,
        "type": ent_type,
        "confidence": confidence,
        "evidence": unique_preserve_order(ent.get("evidence") or [], DEFAULT_MAX_EVIDENCE),
    }


def sanitize_relation(rel, known_ids):
    source = normalize_space(str(rel.get("source", "")))
    target = normalize_space(str(rel.get("target", "")))
    rel_type = normalize_relation_type(str(rel.get("type", "")))
    if not source or not target or not rel_type:
        return None
    if source not in known_ids or target not in known_ids:
        return None
    attrs = rel.get("attributes") or {}
    if not isinstance(attrs, dict):
        attrs = {}
    try:
        confidence = max(0.0, min(1.0, float(rel.get("confidence", 0.5) or 0.5)))
    except Exception:
        confidence = 0.5
    return {
        "source": source,
        "target": target,
        "type": rel_type,
        "confidence": confidence,
        "evidence": unique_preserve_order(rel.get("evidence") or [], DEFAULT_MAX_EVIDENCE),
        "attributes": {
            "date": normalize_space(str(attrs.get("date", ""))),
            "note": normalize_space(str(attrs.get("note", "")))[:DEFAULT_RELATION_NOTE_MAX],
        },
    }


def ensure_subject_entity(obj, subject_hint, record_id):
    for ent in obj.setdefault("entities", []):
        if ent.get("id") == "subject_person":
            ent.setdefault("type", "PERSON")
            ent["type"] = str(ent.get("type") or "PERSON").upper()
            if not ent.get("name") and subject_hint:
                ent["name"] = subject_hint
            ent["confidence"] = float(ent.get("confidence", 0.6) or 0.6)
            ent["evidence"] = unique_preserve_order(ent.get("evidence") or [], DEFAULT_MAX_EVIDENCE)
            return
    obj["entities"].insert(0, {
        "id": "subject_person",
        "name": subject_hint or obj.get("subject") or record_id,
        "type": "PERSON",
        "confidence": 0.5,
        "evidence": [],
    })


def sanitize_extraction_object(obj, record_id, subject_hint):
    if not isinstance(obj, dict):
        obj = {}
    obj["record_id"] = normalize_space(str(obj.get("record_id") or record_id))
    obj["subject"] = normalize_space(str(obj.get("subject") or subject_hint or record_id))
    obj.setdefault("entities", [])
    obj.setdefault("relations", [])
    ensure_subject_entity(obj, subject_hint, record_id)

    seen_subj, final_entities = False, []
    for ent in obj.get("entities", []):
        if not isinstance(ent, dict):
            continue
        clean = sanitize_entity(ent)
        if clean is None:
            continue
        if clean["id"] == "subject_person":
            if seen_subj:
                clean["id"] = slugify(clean["name"], "person")
            else:
                seen_subj = True
        final_entities.append(clean)

    known_ids = {e["id"] for e in final_entities}
    final_relations = [
        r for rel in obj.get("relations", [])
        if isinstance(rel, dict)
        for r in [sanitize_relation(rel, known_ids)] if r
    ]

    return {
        "record_id": obj["record_id"],
        "subject": obj["subject"],
        "entities": final_entities,
        "relations": final_relations,
    }


def build_canonical_name_index(subject_hint, hint_lines):
    index = {}

    def add(name):
        key = normalize_key(name)
        if not key:
            return
        if key not in index or len(name) > len(index[key]):
            index[key] = name

    if subject_hint:
        add(subject_hint)
        for part in subject_hint.split():
            if len(part) > 2:
                add(part)

    for line in hint_lines:
        if ":" in line:
            for name in line.split(":", 1)[1].split(","):
                name = name.strip()
                if name:
                    add(name)

    return index


def apply_canonical_names(extraction, canonical_index):
    for ent in extraction.get("entities", []):
        key = normalize_key(ent.get("name", ""))
        if key in canonical_index:
            ent["name"] = canonical_index[key]
    for ent in extraction.get("entities", []):
        if ent.get("id") == "subject_person":
            extraction["subject"] = ent["name"]
            break
    return extraction


def entity_merge_key(ent):
    if ent.get("id") == "subject_person":
        return "subject_person"
    return f"{normalize_key(ent.get('name', ''))}::{normalize_key(ent.get('type', 'ENTITY'))}"


def relation_merge_key(rel):
    attrs = rel.get("attributes") or {}
    return "||".join([
        rel.get("source", ""), rel.get("target", ""), rel.get("type", ""),
        normalize_space(str(attrs.get("date", ""))), normalize_space(str(attrs.get("note", ""))),
    ])


def merge_chunk_objects(record_id, subject_hint, chunk_objects):
    merged_subject = subject_hint or record_id
    entity_store, relation_store = {}, {}

    for obj in chunk_objects:
        merged_subject = obj.get("subject") or merged_subject
        local_id_to_key = {}
        for ent in obj.get("entities", []):
            key = entity_merge_key(ent)
            local_id_to_key[ent["id"]] = key
            if key not in entity_store:
                entity_store[key] = {
                    "id": "subject_person" if key == "subject_person" else ent["id"],
                    "name": ent.get("name", ""),
                    "type": ent.get("type", "ENTITY"),
                    "confidence": float(ent.get("confidence", 0.5)),
                    "evidence": list(ent.get("evidence", [])),
                }
            else:
                cur = entity_store[key]
                cur["confidence"] = max(float(cur.get("confidence", 0)), float(ent.get("confidence", 0)))
                cur["evidence"] = unique_preserve_order(cur["evidence"] + ent.get("evidence", []), DEFAULT_MAX_EVIDENCE)
                if len(ent.get("name", "")) > len(cur.get("name", "")):
                    cur["name"] = ent["name"]

        canonical_by_local = {lid: entity_store[key]["id"] for lid, key in local_id_to_key.items() if key in entity_store}
        for rel in obj.get("relations", []):
            rel2 = {
                **rel,
                "source": canonical_by_local.get(rel.get("source", ""), rel.get("source", "")),
                "target": canonical_by_local.get(rel.get("target", ""), rel.get("target", "")),
            }
            key = relation_merge_key(rel2)
            if key not in relation_store:
                relation_store[key] = {
                    "source": rel2["source"],
                    "target": rel2["target"],
                    "type": rel2.get("type", ""),
                    "confidence": float(rel2.get("confidence", 0.5)),
                    "evidence": list(rel2.get("evidence", [])),
                    "attributes": dict(rel2.get("attributes", {}) or {}),
                }
            else:
                cur = relation_store[key]
                cur["confidence"] = max(float(cur.get("confidence", 0)), float(rel2.get("confidence", 0)))
                cur["evidence"] = unique_preserve_order(cur["evidence"] + rel2.get("evidence", []), DEFAULT_MAX_EVIDENCE)

    merged_entities = list(entity_store.values())
    known_ids = {e["id"] for e in merged_entities}
    merged_relations = [r for r in [sanitize_relation(v, known_ids) for v in relation_store.values()] if r]
    merged_entities.sort(key=lambda e: (e["id"] != "subject_person", e["type"], e["name"]))
    merged_relations.sort(key=lambda r: (r["type"], r["source"], r["target"]))
    return {"record_id": record_id, "subject": merged_subject, "entities": merged_entities, "relations": merged_relations}


def build_extraction_prompt(record_id, subject_hint, text_chunk, hint_lines, chunk_index, total_chunks):
    payload = build_fragment_payload(record_id, subject_hint, text_chunk, hint_lines, chunk_index, total_chunks)
    return EXTRACTION_PROMPT.replace("__DATA__", payload)


def recursive_extract_chunk(client, model_name, record_id, subject_hint, text_chunk, hint_lines,
                            chunk_index, total_chunks, max_retries, base_sleep,
                            token_budget_per_min, max_prompt_tokens, split_depth):
    prompt = build_extraction_prompt(record_id, subject_hint, text_chunk, hint_lines, chunk_index, total_chunks)
    prompt_tokens = count_tokens_safe(client, model_name, prompt)
    print(f"    chunk {chunk_index + 1}/{total_chunks} | tokens={prompt_tokens} | depth={split_depth}")

    if prompt_tokens > max_prompt_tokens:
        if split_depth >= DEFAULT_MAX_SPLIT_DEPTH or len(text_chunk) <= DEFAULT_MIN_CHUNK_CHARS:
            raise RuntimeError(f"Chunk too large and cannot split further ({prompt_tokens} tokens).")
        parts = split_text_in_half(text_chunk)
        objs = [recursive_extract_chunk(client, model_name, record_id, subject_hint, part, hint_lines,
                                        i, len(parts), max_retries, base_sleep, token_budget_per_min,
                                        max_prompt_tokens, split_depth + 1)
                for i, part in enumerate(parts)]
        return merge_chunk_objects(record_id, subject_hint, objs)

    try:
        response = paced_generate_content(client, model_name, prompt, max_retries, base_sleep, token_budget_per_min)
        obj = parse_model_output_from_response(response)
        return sanitize_extraction_object(obj, record_id=record_id, subject_hint=subject_hint)
    except Exception as e:
        msg = str(e)
        if split_depth >= DEFAULT_MAX_SPLIT_DEPTH or len(text_chunk) <= DEFAULT_MIN_CHUNK_CHARS:
            raise RuntimeError(f"Chunk failed even at max split depth: {msg}")
        print(f"      -> split and retry due to: {msg}")
        parts = split_text_in_half(text_chunk)
        if len(parts) <= 1:
            raise
        objs = [recursive_extract_chunk(client, model_name, record_id, subject_hint, part, hint_lines,
                                        i, len(parts), max_retries, base_sleep, token_budget_per_min,
                                        max_prompt_tokens, split_depth + 1)
                for i, part in enumerate(parts)]
        return merge_chunk_objects(record_id, subject_hint, objs)


def maybe_reflect_chunk(client, model_name, source_text, extraction, record_id, subject_hint,
                        max_retries, base_sleep, token_budget_per_min):
    extraction_json = json.dumps(extraction, ensure_ascii=False, indent=2)
    prompt = REFLECTION_PROMPT.replace("__SOURCE__", source_text).replace("__EXTRACTION__", extraction_json)
    tokens = count_tokens_safe(client, model_name, prompt)

    if tokens > DEFAULT_REFLECT_MAX_PROMPT_TOKENS:
        print(f"    [reflect] skipped (prompt too large: {tokens} tokens)")
        return extraction

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            wait = max(base_sleep, 60.0 * tokens / max(1, token_budget_per_min))
            if wait > base_sleep:
                print(f"    [reflect token guard] {tokens} tokens -> sleep {wait:.1f}s")
            time.sleep(wait)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.5,
                    top_p=1.0,
                    candidate_count=1,
                    max_output_tokens=3072,
                ),
            )
            obj = parse_model_output_from_response(response)
            reflected = sanitize_extraction_object(obj, record_id=record_id, subject_hint=subject_hint)
            print(f"    [reflect] entities={len(reflected['entities'])} relations={len(reflected['relations'])}")
            return reflected
        except Exception as e:
            last_error = e
            print(f"    [reflect] attempt {attempt} failed: {e}")
            if attempt >= max_retries:
                break
            time.sleep(parse_retry_delay_seconds(e, attempt))

    print(f"    [reflect] keeping initial extraction after failures: {last_error}")
    return extraction


def build_initial_chunks(client, model_name, record_id, subject_hint, text_body, hint_lines, target_prompt_tokens):
    if not text_body.strip():
        return [""]
    initial_prompt = build_extraction_prompt(record_id, subject_hint, text_body, hint_lines, 0, 1)
    if count_tokens_safe(client, model_name, initial_prompt) <= target_prompt_tokens:
        return [text_body]

    chunks = split_long_text(text_body, max_chunk_chars=DEFAULT_MAX_CHUNK_CHARS)
    changed = True
    while changed:
        changed = False
        refined = []
        for chunk in chunks:
            prompt = build_extraction_prompt(record_id, subject_hint, chunk, hint_lines, 0, 1)
            if count_tokens_safe(client, model_name, prompt) <= target_prompt_tokens:
                refined.append(chunk)
            else:
                parts = split_text_in_half(chunk)
                if len(parts) == 1:
                    refined.append(chunk)
                else:
                    refined.extend(parts)
                    changed = True
        chunks = refined
    return chunks


def process_record(client, model_name, record_id, subject_hint, text_body, hint_lines,
                   max_retries, base_sleep, token_budget_per_min, max_prompt_tokens,
                   target_prompt_tokens, canonical_index, enable_chunk_reflection,
                   enable_record_reflection):
    raw_chunks = build_initial_chunks(client, model_name, record_id, subject_hint, text_body, hint_lines, target_prompt_tokens)
    chunk_objects = []
    total = len(raw_chunks)
    for i, chunk_text in enumerate(raw_chunks):
        obj = recursive_extract_chunk(client, model_name, record_id, subject_hint, chunk_text, hint_lines,
                                      i, total, max_retries, base_sleep, token_budget_per_min,
                                      max_prompt_tokens, 0)
        if enable_chunk_reflection:
            obj = maybe_reflect_chunk(client, model_name, chunk_text, obj, record_id, subject_hint,
                                      max_retries, base_sleep, token_budget_per_min)
        chunk_objects.append(obj)
        time.sleep(base_sleep)

    extraction = chunk_objects[0] if len(chunk_objects) == 1 else merge_chunk_objects(record_id, subject_hint, chunk_objects)
    extraction = apply_canonical_names(extraction, canonical_index)
    print(f"    [merged] entities={len(extraction['entities'])} relations={len(extraction['relations'])}")

    if enable_record_reflection and len(text_body) <= DEFAULT_RECORD_REFLECTION_MAX_TEXT_CHARS:
        extraction = maybe_reflect_chunk(client, model_name, text_body, extraction, record_id, subject_hint,
                                         max_retries, base_sleep, token_budget_per_min)
        extraction = apply_canonical_names(extraction, canonical_index)
    return extraction


def derive_record_id(parsed_line, fallback_idx):
    if isinstance(parsed_line, dict):
        return str(parsed_line.get("reference") or parsed_line.get("id") or parsed_line.get("reference_id") or f"{parsed_line.get('name', 'unknown')}_{fallback_idx}")
    return f"line_{fallback_idx}"


def parse_and_process(input_file, entities_file, relations_file, checkpoint_file, model_name,
                      limit, max_retries, base_sleep, token_budget_per_min,
                      max_prompt_tokens, target_prompt_tokens,
                      enable_chunk_reflection, enable_record_reflection):
    checkpoint = load_checkpoint(checkpoint_file)
    last_line = checkpoint.get("last_line", -1)
    processed_ids = set(checkpoint.get("processed_ids", []))

    Path(os.path.dirname(entities_file) or ".").mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(relations_file) or ".").mkdir(parents=True, exist_ok=True)
    client = make_client()

    with open(input_file, "r", encoding="utf-8") as infile, \
         open(entities_file, "a", encoding="utf-8") as ent_out, \
         open(relations_file, "a", encoding="utf-8") as rel_out:

        for idx, raw in enumerate(infile):
            if limit is not None and idx >= limit:
                print("Reached limit, stopping.")
                break
            if idx <= last_line:
                continue

            line = raw.strip()
            if not line:
                checkpoint["last_line"] = idx
                save_checkpoint(checkpoint_file, checkpoint)
                continue

            parsed_line = None
            try:
                parsed_line = json.loads(line)
            except Exception:
                pass

            record_id = derive_record_id(parsed_line, idx)
            if record_id in processed_ids:
                print(f"[{idx}] {record_id} already processed -> skip")
                checkpoint["last_line"] = idx
                save_checkpoint(checkpoint_file, checkpoint)
                continue

            try:
                if parsed_line is not None:
                    text_body, hint_lines, subject_hint = build_slim_components(parsed_line)
                else:
                    text_body, hint_lines, subject_hint = normalize_space(line), [], ""
                if not text_body:
                    raise ValueError("Empty text field.")

                canonical_index = build_canonical_name_index(subject_hint, hint_lines)
                print(f"[{idx}] record_id={record_id}")
                extraction = process_record(client, model_name, record_id, subject_hint, text_body, hint_lines,
                                            max_retries, base_sleep, token_budget_per_min,
                                            max_prompt_tokens, target_prompt_tokens,
                                            canonical_index, enable_chunk_reflection,
                                            enable_record_reflection)

                ent_obj = {"record_id": extraction["record_id"], "subject": extraction["subject"], "entities": extraction["entities"]}
                rel_obj = {"record_id": extraction["record_id"], "subject": extraction["subject"], "relations": extraction["relations"]}

                ent_out.write(json.dumps(ent_obj, ensure_ascii=False) + "\n")
                fsync_and_flush(ent_out)
                rel_out.write(json.dumps(rel_obj, ensure_ascii=False) + "\n")
                fsync_and_flush(rel_out)

                processed_ids.add(record_id)
                checkpoint["last_line"] = idx
                checkpoint["processed_ids"] = sorted(processed_ids)
                save_checkpoint(checkpoint_file, checkpoint)
                print(f"[{idx}] OK {record_id} | entities={len(ent_obj['entities'])} relations={len(rel_obj['relations'])}")
            except Exception as e:
                print(f"[{idx}] FAILED {record_id}: {e}")
                err = {"record_id": record_id, "error": str(e)}
                ent_out.write(json.dumps(err, ensure_ascii=False) + "\n")
                fsync_and_flush(ent_out)
                rel_out.write(json.dumps(err, ensure_ascii=False) + "\n")
                fsync_and_flush(rel_out)
                checkpoint["last_line"] = idx
                checkpoint["processed_ids"] = sorted(processed_ids)
                save_checkpoint(checkpoint_file, checkpoint)

            time.sleep(base_sleep)
    print("Done.")


def main():
    p = argparse.ArgumentParser(description="Studium KG extraction with stable recursive chunking + reflection")
    p.add_argument("--model", default=DEFAULT_MODEL_NAME)
    p.add_argument("--input-file", default=DEFAULT_INPUT_FILE)
    p.add_argument("--entities-file", default=DEFAULT_ENTITIES_FILE)
    p.add_argument("--relations-file", default=DEFAULT_RELATIONS_FILE)
    p.add_argument("--checkpoint-file", default=DEFAULT_CHECKPOINT_FILE)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    p.add_argument("--base-sleep", type=float, default=DEFAULT_BASE_SLEEP)
    p.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET_PER_MIN)
    p.add_argument("--target-prompt-tokens", type=int, default=DEFAULT_TARGET_PROMPT_TOKENS)
    p.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    p.add_argument("--no-chunk-reflection", action="store_true")
    p.add_argument("--record-reflection", action="store_true")
    args = p.parse_args()

    parse_and_process(
        input_file=args.input_file,
        entities_file=args.entities_file,
        relations_file=args.relations_file,
        checkpoint_file=args.checkpoint_file,
        model_name=args.model,
        limit=args.limit,
        max_retries=args.max_retries,
        base_sleep=args.base_sleep,
        token_budget_per_min=args.token_budget,
        max_prompt_tokens=args.max_prompt_tokens,
        target_prompt_tokens=args.target_prompt_tokens,
        enable_chunk_reflection=not args.no_chunk_reflection,
        enable_record_reflection=args.record_reflection,
    )


if __name__ == "__main__":
    main()
