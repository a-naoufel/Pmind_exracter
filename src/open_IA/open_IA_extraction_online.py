# -*- coding: utf-8 -*-
"""
Extraction d'entités / relations depuis un fichier JSONL de notices biographiques
avec un modèle local servi par vLLM (OpenAI-compatible API).

Testé avec : Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 sur ppti-gpu-4 (V100 32 GB)

Installation :
    pip install openai json-repair

Usage :
    python qwen_extraction_local.py \
        --input-file /chemin/vers/test_suite.jsonl \
        --entities-file entities_local.jsonl \
        --relations-file relations_local.jsonl

Options importantes :
    --base-url        URL du serveur vLLM  (défaut: http://127.0.0.1:8000/v1)
    --model           Nom du modèle tel que déclaré par vLLM (défaut: auto-détecté)
    --max-tokens      Tokens max en sortie  (défaut: 2048)
    --max-prompt-chars  Nombre max de caractères dans le texte d'un chunk (défaut: 12000)
    --concurrency     Nombre de requêtes parallèles (défaut: 1, monte prudemment)
    --limit           Nombre max de notices à traiter (debug)
    --no-reflection   Désactive la boucle de self-reflection (plus rapide, moins stable)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# Configuration par défaut
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_BASE_URL              = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL_NAME            = None          # auto-détecté via /v1/models
DEFAULT_INPUT_FILE            = "test_suite.jsonl"
DEFAULT_ENTITIES_FILE         = "entities_local.jsonl"
DEFAULT_RELATIONS_FILE        = "relations_local.jsonl"
DEFAULT_CHECKPOINT_FILE       = "checkpoint_local.json"
DEFAULT_MAX_RETRIES           = 4
DEFAULT_BASE_SLEEP            = 0.5           # local → pas de quota, pause légère
DEFAULT_MAX_TOKENS            = 2048          # tokens de sortie max par appel
DEFAULT_MAX_PROMPT_CHARS      = 12000         # caractères max de texte par chunk
DEFAULT_MAX_HINT_ITEMS        = 12
DEFAULT_MAX_EVIDENCE          = 8
DEFAULT_RELATION_NOTE_MAX     = 300
DEFAULT_TEMPERATURE           = 0.1           # bas pour cohérence JSON
DEFAULT_REFLECTION_TEMPERATURE = 0.0

# ──────────────────────────────────────────────────────────────────────────────
# Enum verrouillé des relations
# ──────────────────────────────────────────────────────────────────────────────
RELATION_TYPES = [
    "STUDIED_AT", "TAUGHT_AT", "OBTAINED_DEGREE", "LECTURED_AT", "LECTURED_SUBJECT",
    "ENROLLED_IN", "AFFILIATED_WITH", "MEMBER_OF", "BELONGS_TO", "FOUNDED",
    "HELD_POSITION_IN", "SERVED", "PHYSICIAN_OF", "BORN_IN", "DIED_IN", "ACTIVE_IN",
    "FROM_DIOCESE", "STUDENT_OF", "MENTOR_OF", "DEDICATED_TO", "ADDRESSED",
    "OPPOSED_TO", "COLLABORATED_WITH", "SPOUSE_OF", "PARENT_OF", "SIBLING_OF",
    "ATTESTED_BY", "AUTHORED", "TRANSLATED",
]
RELATION_SET     = set(RELATION_TYPES)
RELATION_ENUM_STR = ", ".join(RELATION_TYPES)

RELATION_ALIASES = {
    "TEACHES_AT": "TAUGHT_AT", "TAUGHTS_AT": "TAUGHT_AT", "TEACHING_AT": "TAUGHT_AT",
    "STUDY_AT": "STUDIED_AT", "STUDIES_AT": "STUDIED_AT", "STUDIEDIN": "STUDIED_AT",
    "LECTURES_AT": "LECTURED_AT", "LECTUREDIN": "LECTURED_AT",
    "MEMBER": "MEMBER_OF", "BELONGS": "BELONGS_TO",
    "AUTHOR_OF": "AUTHORED", "WRITES": "AUTHORED", "WRITTEN_BY": "AUTHORED",
    "TRANSLATOR_OF": "TRANSLATED",
    "MARRIED_TO": "SPOUSE_OF", "CHILD_OF": "PARENT_OF",
    "BROTHER_OF": "SIBLING_OF", "SISTER_OF": "SIBLING_OF",
    "COLLABORATES_WITH": "COLLABORATED_WITH",
}

# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Tu es un extracteur d'information spécialisé dans les corpus historiques médiévaux. "
    "Tu réponds UNIQUEMENT par un objet JSON valide, sans markdown, sans explication."
)

EXTRACTION_TEMPLATE = """Tu reçois un FRAGMENT d'une notice biographique du corpus Studium Parisiense.

TÂCHE : extraire toutes les entités et toutes les relations présentes dans ce fragment, sans invention.

RÈGLES :
- Réponds UNIQUEMENT par un objet JSON valide. Aucun markdown. Aucune explication.
- L'entité représentant l'individu principal DOIT avoir l'id exact : "subject_person".
- Les autres ids doivent être stables, simples et uniques dans ce fragment.
- Les relations doivent utiliser UNIQUEMENT les ids présents dans `entities`.
- Les types de relations autorisés sont STRICTEMENT : {relation_enum}
- N'invente jamais un type hors de cette liste.
- Si une relation est seulement faible ou ambiguë, baisse `confidence` au lieu d'inventer.
- `evidence` doit contenir des extraits verbatim du fragment.
- Si une information n'est pas présente dans ce fragment, n'invente rien.

TYPES D'ENTITÉS suggérés (liste ouverte, en UPPER_SNAKE_CASE) :
PERSON, PLACE, UNIVERSITY, INSTITUTION, DIOCESE, DEGREE, ROLE, DATE, SOURCE, WORK, NATION.

FORMAT DE SORTIE OBLIGATOIRE :
{{
  "record_id": "<référence>",
  "subject": "<nom canonique ou meilleur nom disponible>",
  "entities": [
    {{
      "id": "subject_person",
      "name": "<nom>",
      "type": "PERSON",
      "confidence": 0.0,
      "evidence": ["<citation verbatim>"]
    }}
  ],
  "relations": [
    {{
      "source": "subject_person",
      "target": "<id_entité>",
      "type": "<TYPE_RELATION>",
      "confidence": 0.0,
      "evidence": ["<citation verbatim>"],
      "attributes": {{"date": "<si disponible>", "note": "<si utile>"}}
    }}
  ]
}}

Si aucune relation n'est trouvée, renvoie `relations: []`.
Si une seule entité est trouvée, renvoie cette entité et `relations: []`.

FRAGMENT À ANALYSER :
{data}
""".format(relation_enum=RELATION_ENUM_STR, data="{data}")

REFLECTION_TEMPLATE = """Tu viens d'extraire des entités et relations depuis un fragment historique.
Voici ta première extraction :

{first_extraction}

Voici le fragment source original :

{source_text}

Révise ton extraction en corrigeant UNIQUEMENT :
[A] Les entités manquantes clairement mentionnées dans le texte
[B] Les noms inconsistants ou abrégés (utilise la forme la plus complète)
[C] Les types de relations manquants ou erronés

Types de relations autorisés : {relation_enum}

Renvoie UNIQUEMENT l'objet JSON corrigé complet, sans explication ni markdown.
""".format(first_extraction="{first_extraction}", source_text="{source_text}", relation_enum=RELATION_ENUM_STR)


# ──────────────────────────────────────────────────────────────────────────────
# Utilitaires généraux
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

def unique_preserve_order(items: List[str], max_items: Optional[int] = None) -> List[str]:
    seen = set()
    out = []
    for item in items:
        item = normalize_space(item)
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
        if max_items is not None and len(out) >= max_items:
            break
    return out

def fsync_and_flush(fh) -> None:
    fh.flush()
    try:
        os.fsync(fh.fileno())
    except Exception:
        pass

def safe_json_load(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        try:
            import json_repair  # type: ignore
            return json_repair.loads(text)
        except Exception:
            return None

def extract_first_json_object(text: str) -> Optional[str]:
    start = None
    depth = 0
    in_str = False
    esc = False
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

def stable_hash(text: str, n: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ──────────────────────────────────────────────────────────────────────────────
def load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_line": -1, "processed_ids": []}

def save_checkpoint(path: str, state: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# Construction de l'input slim
# ──────────────────────────────────────────────────────────────────────────────
def build_hint_lines(parsed_line: dict, max_hint_items: int = DEFAULT_MAX_HINT_ITEMS) -> List[str]:
    meta = parsed_line.get("meta_entities", {}) or {}
    def cap(values: List[str]) -> str:
        return ", ".join(unique_preserve_order(values, max_hint_items))
    hints = []
    names = meta.get("names") or []
    places = meta.get("places") or []
    institutions = meta.get("institutions") or []
    if names:
        hints.append(f"Variantes de noms connues : {cap(names)}")
    if places:
        hints.append(f"Lieux identifiés : {cap(places)}")
    if institutions:
        hints.append(f"Institutions identifiées : {cap(institutions)}")
    return hints

def build_slim_components(parsed_line: dict) -> Tuple[str, List[str], str]:
    text_body = normalize_space(parsed_line.get("text", ""))
    subject_hint = (
        parsed_line.get("name")
        or parsed_line.get("subject")
        or parsed_line.get("canonical_name")
        or ""
    )
    hints = build_hint_lines(parsed_line)
    return text_body, hints, normalize_space(subject_hint)

def build_fragment_payload(
    record_id: str,
    subject_hint: str,
    text_chunk: str,
    hint_lines: List[str],
    chunk_index: int,
    total_chunks: int,
) -> str:
    parts = [
        f"record_id: {record_id}",
        f"chunk_index: {chunk_index + 1}/{total_chunks}",
    ]
    if subject_hint:
        parts.append(f"subject_hint: {subject_hint}")
    if hint_lines:
        parts.append("hints:\n- " + "\n- ".join(hint_lines))
    parts.append("text:\n" + text_chunk)
    return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Découpage des longues notices (basé sur les caractères — pas de token counter remote)
# ──────────────────────────────────────────────────────────────────────────────
def split_long_text(text: str, max_chunk_chars: int = DEFAULT_MAX_PROMPT_CHARS) -> List[str]:
    text = text.strip()
    if not text:
        return [""]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush_current():
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_len = 0

    for para in paragraphs:
        if len(para) > max_chunk_chars:
            flush_current()
            sentences = re.split(r"(?<=[\.!?;:])\s+", para)
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
            flush_current()
        current.append(para)
        current_len += add_len

    flush_current()
    return chunks or [text]

def chunk_notice_to_prompts(
    record_id: str,
    subject_hint: str,
    text_body: str,
    hint_lines: List[str],
    max_prompt_chars: int,
) -> List[str]:
    if not text_body.strip():
        payload = build_fragment_payload(record_id, subject_hint, "", hint_lines, 0, 1)
        return [EXTRACTION_TEMPLATE.replace("{data}", payload)]

    # essai sans découpage
    initial_payload = build_fragment_payload(record_id, subject_hint, text_body, hint_lines, 0, 1)
    if len(text_body) <= max_prompt_chars:
        return [EXTRACTION_TEMPLATE.replace("{data}", initial_payload)]

    chunks = split_long_text(text_body, max_chunk_chars=max_prompt_chars)
    total = len(chunks)
    prompts = []
    for i, chunk in enumerate(chunks):
        payload = build_fragment_payload(record_id, subject_hint, chunk, hint_lines, i, total)
        prompts.append(EXTRACTION_TEMPLATE.replace("{data}", payload))
    return prompts


# ──────────────────────────────────────────────────────────────────────────────
# Client vLLM (OpenAI-compatible)
# ──────────────────────────────────────────────────────────────────────────────
def make_client(base_url: str) -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY", "dummy")
    return OpenAI(base_url=base_url, api_key=api_key)

def detect_model_name(client: OpenAI) -> str:
    """Récupère le premier modèle disponible depuis /v1/models."""
    try:
        models = client.models.list()
        names = [m.id for m in models.data]
        if names:
            print(f"  Modèles disponibles : {names}")
            return names[0]
    except Exception as e:
        print(f"  Avertissement : impossible de détecter le modèle ({e})")
    return "default"


# ──────────────────────────────────────────────────────────────────────────────
# Appel modèle
# ──────────────────────────────────────────────────────────────────────────────
def call_model(
    client: OpenAI,
    model_name: str,
    user_content: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    base_sleep: float,
) -> str:
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            last_error = e
            wait = base_sleep * (2 ** (attempt - 1))
            print(f"  Erreur attempt {attempt}/{max_retries}: {e} → retry dans {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"Échec après {max_retries} tentatives : {last_error}")


def parse_model_output(raw_text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", raw_text, flags=re.IGNORECASE).strip()
    obj = safe_json_load(cleaned)
    if obj is not None:
        return obj
    candidate = extract_first_json_object(cleaned)
    if candidate is None:
        raise ValueError(f"Impossible d'extraire un objet JSON valide :\n{cleaned[:1200]}")
    obj = safe_json_load(candidate)
    if obj is None:
        raise ValueError(f"Parsing JSON échoué :\n{candidate[:1200]}")
    return obj


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation / validation de la sortie modèle
# ──────────────────────────────────────────────────────────────────────────────
def normalize_relation_type(rel_type: str) -> Optional[str]:
    rel_type = normalize_space(rel_type).upper()
    rel_type = re.sub(r"[^A-Z_]", "_", rel_type)
    rel_type = re.sub(r"_+", "_", rel_type).strip("_")
    rel_type = RELATION_ALIASES.get(rel_type, rel_type)
    if rel_type in RELATION_SET:
        return rel_type
    return None

def ensure_subject_entity(obj: dict, subject_hint: str, record_id: str) -> None:
    entities = obj.setdefault("entities", [])
    for ent in entities:
        if ent.get("id") == "subject_person":
            ent.setdefault("type", "PERSON")
            ent["type"] = str(ent.get("type") or "PERSON").upper()
            if not ent.get("name") and subject_hint:
                ent["name"] = subject_hint
            ent["confidence"] = float(ent.get("confidence", 0.6) or 0.6)
            ent["evidence"] = unique_preserve_order(ent.get("evidence") or [], DEFAULT_MAX_EVIDENCE)
            return
    name = subject_hint or obj.get("subject") or record_id
    entities.insert(0, {
        "id": "subject_person",
        "name": name,
        "type": "PERSON",
        "confidence": 0.5,
        "evidence": [],
    })

def sanitize_entity(ent: dict) -> Optional[dict]:
    name = normalize_space(str(ent.get("name", "")))
    ent_type = normalize_space(str(ent.get("type", ""))).upper() or "ENTITY"
    ent_id = normalize_space(str(ent.get("id", "")))
    if not name:
        return None
    if not ent_id:
        ent_id = slugify(f"{name}_{ent_type}")
    evidence = unique_preserve_order(ent.get("evidence") or [], DEFAULT_MAX_EVIDENCE)
    try:
        confidence = float(ent.get("confidence", 0.5) or 0.5)
    except Exception:
        confidence = 0.5
    return {
        "id": ent_id,
        "name": name,
        "type": ent_type,
        "confidence": max(0.0, min(1.0, confidence)),
        "evidence": evidence,
    }

def sanitize_relation(rel: dict, known_ids: set) -> Optional[dict]:
    source   = normalize_space(str(rel.get("source", "")))
    target   = normalize_space(str(rel.get("target", "")))
    rel_type = normalize_relation_type(str(rel.get("type", "")))
    if not source or not target or not rel_type:
        return None
    if source not in known_ids or target not in known_ids:
        return None
    attrs = rel.get("attributes") or {}
    if not isinstance(attrs, dict):
        attrs = {}
    evidence = unique_preserve_order(rel.get("evidence") or [], DEFAULT_MAX_EVIDENCE)
    try:
        confidence = float(rel.get("confidence", 0.5) or 0.5)
    except Exception:
        confidence = 0.5
    return {
        "source": source,
        "target": target,
        "type": rel_type,
        "confidence": max(0.0, min(1.0, confidence)),
        "evidence": evidence,
        "attributes": {
            "date": normalize_space(str(attrs.get("date", ""))),
            "note": normalize_space(str(attrs.get("note", "")))[:DEFAULT_RELATION_NOTE_MAX],
        },
    }

def sanitize_extraction_object(obj: dict, record_id: str, subject_hint: str) -> dict:
    if not isinstance(obj, dict):
        obj = {}
    obj["record_id"] = normalize_space(str(obj.get("record_id") or record_id))
    obj["subject"]   = normalize_space(str(obj.get("subject") or subject_hint or record_id))
    obj.setdefault("entities", [])
    obj.setdefault("relations", [])
    ensure_subject_entity(obj, subject_hint, record_id)

    entities_clean = []
    for ent in obj.get("entities", []):
        if isinstance(ent, dict):
            clean = sanitize_entity(ent)
            if clean is not None:
                entities_clean.append(clean)

    subject_inserted = False
    final_entities = []
    for ent in entities_clean:
        if ent["id"] == "subject_person":
            if subject_inserted:
                ent["id"] = slugify(ent["name"], "person")
            else:
                subject_inserted = True
        final_entities.append(ent)

    known_ids = {ent["id"] for ent in final_entities}
    relations_clean = []
    for rel in obj.get("relations", []):
        if isinstance(rel, dict):
            clean_rel = sanitize_relation(rel, known_ids)
            if clean_rel is not None:
                relations_clean.append(clean_rel)

    return {
        "record_id": obj["record_id"],
        "subject":   obj["subject"],
        "entities":  final_entities,
        "relations": relations_clean,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Self-reflection (second appel à température 0)
# ──────────────────────────────────────────────────────────────────────────────
def run_reflection(
    client: OpenAI,
    model_name: str,
    first_extraction: dict,
    source_text: str,
    max_tokens: int,
    max_retries: int,
    base_sleep: float,
) -> Optional[dict]:
    prompt = REFLECTION_TEMPLATE.format(
        first_extraction=json.dumps(first_extraction, ensure_ascii=False, indent=2),
        source_text=source_text,
    )
    try:
        raw = call_model(
            client=client,
            model_name=model_name,
            user_content=prompt,
            max_tokens=max_tokens,
            temperature=DEFAULT_REFLECTION_TEMPERATURE,
            max_retries=max_retries,
            base_sleep=base_sleep,
        )
        return parse_model_output(raw)
    except Exception as e:
        print(f"  [reflection] échec, on garde l'extraction originale : {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Merge inter-chunks
# ──────────────────────────────────────────────────────────────────────────────
def entity_merge_key(ent: dict) -> str:
    if ent.get("id") == "subject_person":
        return "subject_person"
    return f"{normalize_key(ent.get('name', ''))}::{normalize_key(ent.get('type', 'ENTITY'))}"

def relation_merge_key(rel: dict) -> str:
    attrs = rel.get("attributes") or {}
    return "||".join([
        rel.get("source", ""),
        rel.get("target", ""),
        rel.get("type", ""),
        normalize_space(str(attrs.get("date", ""))),
        normalize_space(str(attrs.get("note", ""))),
    ])

def merge_chunk_objects(record_id: str, subject_hint: str, chunk_objects: List[dict]) -> dict:
    merged_subject = subject_hint or record_id
    entity_store:   Dict[str, dict] = {}
    relation_store: Dict[str, dict] = {}

    for obj in chunk_objects:
        merged_subject = obj.get("subject") or merged_subject
        local_id_to_key: Dict[str, str] = {}

        for ent in obj.get("entities", []):
            key = entity_merge_key(ent)
            local_id_to_key[ent["id"]] = key
            if key not in entity_store:
                canonical_id = "subject_person" if key == "subject_person" else ent["id"]
                if canonical_id == "subject_person" and ent.get("type") != "PERSON":
                    ent["type"] = "PERSON"
                entity_store[key] = {
                    "id":         canonical_id,
                    "name":       ent.get("name", ""),
                    "type":       ent.get("type", "ENTITY"),
                    "confidence": float(ent.get("confidence", 0.5)),
                    "evidence":   list(ent.get("evidence", [])),
                }
            else:
                cur = entity_store[key]
                cur["confidence"] = max(float(cur.get("confidence", 0.0)), float(ent.get("confidence", 0.0)))
                cur["evidence"]   = unique_preserve_order(cur.get("evidence", []) + ent.get("evidence", []), DEFAULT_MAX_EVIDENCE)
                if len(ent.get("name", "")) > len(cur.get("name", "")):
                    cur["name"] = ent.get("name", cur.get("name", ""))

        canonical_id_by_local_id = {
            local_id: entity_store[key]["id"]
            for local_id, key in local_id_to_key.items()
            if key in entity_store
        }

        for rel in obj.get("relations", []):
            src = canonical_id_by_local_id.get(rel.get("source", ""), rel.get("source", ""))
            tgt = canonical_id_by_local_id.get(rel.get("target", ""), rel.get("target", ""))
            rel2 = {**rel, "source": src, "target": tgt}
            key  = relation_merge_key(rel2)
            if key not in relation_store:
                relation_store[key] = {
                    "source":     src,
                    "target":     tgt,
                    "type":       rel2.get("type", ""),
                    "confidence": float(rel2.get("confidence", 0.5)),
                    "evidence":   list(rel2.get("evidence", [])),
                    "attributes": dict(rel2.get("attributes", {}) or {}),
                }
            else:
                cur = relation_store[key]
                cur["confidence"] = max(float(cur.get("confidence", 0.0)), float(rel2.get("confidence", 0.0)))
                cur["evidence"]   = unique_preserve_order(cur.get("evidence", []) + rel2.get("evidence", []), DEFAULT_MAX_EVIDENCE)

    merged_entities = list(entity_store.values())
    merged_known_ids = {e["id"] for e in merged_entities}
    merged_relations = [
        r for r in (sanitize_relation(rel, merged_known_ids) for rel in relation_store.values())
        if r is not None
    ]
    merged_entities.sort(key=lambda e: (e["id"] != "subject_person", e["type"], e["name"]))
    merged_relations.sort(key=lambda r: (r["type"], r["source"], r["target"], r["attributes"].get("date", "")))

    return {
        "record_id": record_id,
        "subject":   merged_subject,
        "entities":  merged_entities,
        "relations": merged_relations,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Traitement d'un record
# ──────────────────────────────────────────────────────────────────────────────
def derive_record_id(parsed_line: Optional[dict], fallback_idx: int) -> str:
    if isinstance(parsed_line, dict):
        return str(
            parsed_line.get("reference")
            or parsed_line.get("id")
            or parsed_line.get("reference_id")
            or f"{parsed_line.get('name', 'unknown')}_{fallback_idx}"
        )
    return f"line_{fallback_idx}"


def process_record(
    client: OpenAI,
    model_name: str,
    record_id: str,
    subject_hint: str,
    text_body: str,
    hint_lines: List[str],
    max_tokens: int,
    max_retries: int,
    base_sleep: float,
    max_prompt_chars: int,
    use_reflection: bool,
) -> dict:
    prompts = chunk_notice_to_prompts(
        record_id=record_id,
        subject_hint=subject_hint,
        text_body=text_body,
        hint_lines=hint_lines,
        max_prompt_chars=max_prompt_chars,
    )

    chunk_objects = []
    total = len(prompts)

    for i, prompt in enumerate(prompts, start=1):
        print(f"    chunk {i}/{total} | ~{len(prompt)} chars")
        raw = call_model(
            client=client,
            model_name=model_name,
            user_content=prompt,
            max_tokens=max_tokens,
            temperature=DEFAULT_TEMPERATURE,
            max_retries=max_retries,
            base_sleep=base_sleep,
        )
        obj = parse_model_output(raw)
        obj = sanitize_extraction_object(obj, record_id=record_id, subject_hint=subject_hint)

        if use_reflection:
            # source text for this chunk only (strip the prompt header, keep the text section)
            chunk_text = text_body if total == 1 else prompts[i - 1]
            reflected = run_reflection(
                client=client,
                model_name=model_name,
                first_extraction=obj,
                source_text=chunk_text,
                max_tokens=max_tokens,
                max_retries=max_retries,
                base_sleep=base_sleep,
            )
            if reflected is not None:
                reflected = sanitize_extraction_object(reflected, record_id=record_id, subject_hint=subject_hint)
                obj = reflected

        chunk_objects.append(obj)
        if i < total:
            time.sleep(base_sleep)

    if len(chunk_objects) == 1:
        return chunk_objects[0]
    return merge_chunk_objects(record_id=record_id, subject_hint=subject_hint, chunk_objects=chunk_objects)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────────────────────────────────────
def parse_and_process(
    base_url: str,
    model_name: Optional[str],
    input_file: str,
    entities_file: str,
    relations_file: str,
    checkpoint_file: str,
    limit: Optional[int],
    max_tokens: int,
    max_retries: int,
    base_sleep: float,
    max_prompt_chars: int,
    use_reflection: bool,
) -> None:
    checkpoint    = load_checkpoint(checkpoint_file)
    last_line     = checkpoint.get("last_line", -1)
    processed_ids = set(checkpoint.get("processed_ids", []))

    Path(os.path.dirname(entities_file)  or ".").mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(relations_file) or ".").mkdir(parents=True, exist_ok=True)

    client = make_client(base_url)

    if not model_name:
        model_name = detect_model_name(client)
    print(f"Modèle utilisé : {model_name}")
    print(f"Endpoint       : {base_url}")
    print(f"Reflection     : {'activée' if use_reflection else 'désactivée'}")

    with open(input_file, "r", encoding="utf-8") as infile, \
         open(entities_file,  "a", encoding="utf-8") as ent_out, \
         open(relations_file, "a", encoding="utf-8") as rel_out:

        for idx, raw in enumerate(infile):
            if limit is not None and idx >= limit:
                print("Limite atteinte, arrêt.")
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
                print(f"[{idx}] record_id={record_id} déjà traité → skip")
                checkpoint["last_line"] = idx
                save_checkpoint(checkpoint_file, checkpoint)
                continue

            try:
                if parsed_line is not None:
                    text_body, hint_lines, subject_hint = build_slim_components(parsed_line)
                else:
                    text_body, hint_lines, subject_hint = normalize_space(line), [], ""

                if not text_body:
                    raise ValueError("Champ `text` vide ou notice illisible.")

                print(f"[{idx}] traitement record_id={record_id} | texte={len(text_body)} chars")
                extraction = process_record(
                    client=client,
                    model_name=model_name,
                    record_id=record_id,
                    subject_hint=subject_hint,
                    text_body=text_body,
                    hint_lines=hint_lines,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    base_sleep=base_sleep,
                    max_prompt_chars=max_prompt_chars,
                    use_reflection=use_reflection,
                )

                ent_obj = {
                    "record_id": extraction["record_id"],
                    "subject":   extraction["subject"],
                    "entities":  extraction["entities"],
                }
                rel_obj = {
                    "record_id": extraction["record_id"],
                    "subject":   extraction["subject"],
                    "relations": extraction["relations"],
                }

                ent_out.write(json.dumps(ent_obj, ensure_ascii=False) + "\n")
                fsync_and_flush(ent_out)
                rel_out.write(json.dumps(rel_obj, ensure_ascii=False) + "\n")
                fsync_and_flush(rel_out)

                processed_ids.add(record_id)
                checkpoint["last_line"]     = idx
                checkpoint["processed_ids"] = sorted(processed_ids)
                save_checkpoint(checkpoint_file, checkpoint)

                print(
                    f"[{idx}] OK record_id={record_id} | "
                    f"entities={len(ent_obj['entities'])} relations={len(rel_obj['relations'])}"
                )

            except Exception as e:
                print(f"[{idx}] ÉCHEC record_id={record_id}: {e}")
                error_obj = {"record_id": record_id, "error": str(e)}
                ent_out.write(json.dumps(error_obj, ensure_ascii=False) + "\n")
                fsync_and_flush(ent_out)
                rel_out.write(json.dumps(error_obj, ensure_ascii=False) + "\n")
                fsync_and_flush(rel_out)
                checkpoint["last_line"]     = idx
                checkpoint["processed_ids"] = sorted(processed_ids)
                save_checkpoint(checkpoint_file, checkpoint)

            time.sleep(base_sleep)

    print("Traitement terminé.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extraction KG locale — vLLM OpenAI-compatible endpoint"
    )
    parser.add_argument("--base-url",         default=DEFAULT_BASE_URL,
                        help=f"URL de base vLLM  (défaut: {DEFAULT_BASE_URL})")
    parser.add_argument("--model",            default=DEFAULT_MODEL_NAME,
                        help="Nom du modèle (auto-détecté si absent)")
    parser.add_argument("--input-file",       default=DEFAULT_INPUT_FILE)
    parser.add_argument("--entities-file",    default=DEFAULT_ENTITIES_FILE)
    parser.add_argument("--relations-file",   default=DEFAULT_RELATIONS_FILE)
    parser.add_argument("--checkpoint-file",  default=DEFAULT_CHECKPOINT_FILE)
    parser.add_argument("--limit",            type=int,   default=None,
                        help="Nombre max de notices (debug)")
    parser.add_argument("--max-tokens",       type=int,   default=DEFAULT_MAX_TOKENS,
                        help=f"Tokens max en sortie (défaut: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--max-prompt-chars", type=int,   default=DEFAULT_MAX_PROMPT_CHARS,
                        help=f"Chars max de texte par chunk (défaut: {DEFAULT_MAX_PROMPT_CHARS})")
    parser.add_argument("--max-retries",      type=int,   default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--base-sleep",       type=float, default=DEFAULT_BASE_SLEEP)
    parser.add_argument("--no-reflection",    action="store_true",
                        help="Désactive la boucle de self-reflection")
    args = parser.parse_args()

    parse_and_process(
        base_url        = args.base_url,
        model_name      = args.model,
        input_file      = args.input_file,
        entities_file   = args.entities_file,
        relations_file  = args.relations_file,
        checkpoint_file = args.checkpoint_file,
        limit           = args.limit,
        max_tokens      = args.max_tokens,
        max_retries     = args.max_retries,
        base_sleep      = args.base_sleep,
        max_prompt_chars= args.max_prompt_chars,
        use_reflection  = not args.no_reflection,
    )


if __name__ == "__main__":
    main()
