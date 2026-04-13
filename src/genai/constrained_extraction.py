"""
Constrained relation extraction for the Studium KG pipeline.
Uses Outlines (https://github.com/outlines-dev/outlines) to enforce
a fixed relation-type enum at the logit level — the model cannot
produce any relation type outside the canonical list.

Install:
    pip install outlines pydantic transformers bitsandbytes accelerate
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Optional

import outlines
import outlines.models as models
import outlines.generate as generate
from pydantic import BaseModel, Field


# ── 1. Canonical relation types ──────────────────────────────────────────────
#
# Built from your corpus observations (report sections 1-4).
# Edit this list — adding or removing a value automatically updates
# every downstream validator and the constrained decoder.

class RelationType(str, Enum):
    # Academic trajectory
    STUDIED_AT        = "STUDIED_AT"
    TAUGHT_AT         = "TAUGHT_AT"
    OBTAINED_DEGREE   = "OBTAINED_DEGREE"
    LECTURED_AT       = "LECTURED_AT"
    LECTURED_SUBJECT  = "LECTURED_SUBJECT"
    ENROLLED_IN       = "ENROLLED_IN"

    # Institutional affiliation
    AFFILIATED_WITH   = "AFFILIATED_WITH"
    MEMBER_OF         = "MEMBER_OF"
    BELONGS_TO        = "BELONGS_TO"
    FOUNDED           = "FOUNDED"

    # Roles and positions
    HELD_POSITION_IN  = "HELD_POSITION_IN"
    SERVED            = "SERVED"
    PHYSICIAN_OF      = "PHYSICIAN_OF"

    # Geography and origin
    BORN_IN           = "BORN_IN"
    DIED_IN           = "DIED_IN"
    ACTIVE_IN         = "ACTIVE_IN"
    FROM_DIOCESE      = "FROM_DIOCESE"

    # Social / intellectual network
    STUDENT_OF        = "STUDENT_OF"
    MENTOR_OF         = "MENTOR_OF"
    DEDICATED_TO      = "DEDICATED_TO"
    ADDRESSED         = "ADDRESSED"
    OPPOSED_TO        = "OPPOSED_TO"
    COLLABORATED_WITH = "COLLABORATED_WITH"

    # Family
    SPOUSE_OF         = "SPOUSE_OF"
    PARENT_OF         = "PARENT_OF"
    SIBLING_OF        = "SIBLING_OF"

    # Documentary
    ATTESTED_BY       = "ATTESTED_BY"
    AUTHORED          = "AUTHORED"
    TRANSLATED        = "TRANSLATED"


# ── 2. Pydantic output schema ─────────────────────────────────────────────────

class Entity(BaseModel):
    id:         str  = Field(description="Unique slug, e.g. 'robertus_gervasii'")
    label:      str  = Field(description="Human-readable name as it appears in the text")
    type:       str  = Field(description="UPPER_SNAKE_CASE entity type, e.g. PERSON, PLACE")
    confidence: float = Field(ge=0.0, le=1.0)
    evidence:   str  = Field(description="Verbatim excerpt justifying this entity")


class Relation(BaseModel):
    subject:    str          = Field(description="Entity id of the subject")
    relation:   RelationType = Field(description="Canonical relation type from the fixed enum")
    object:     str          = Field(description="Entity id of the object")
    confidence: float        = Field(ge=0.0, le=1.0)
    evidence:   str          = Field(description="Verbatim excerpt justifying this relation")
    start_date: Optional[str] = None
    end_date:   Optional[str] = None
    note:       Optional[str] = None


class ExtractionOutput(BaseModel):
    entities:  list[Entity]
    relations: list[Relation]


# ── 3. Prompts ────────────────────────────────────────────────────────────────

ENTITY_SYSTEM = """You are a medieval history expert extracting structured data
from biographical notices of university scholars (Studium Parisiense corpus).

Extract ALL relevant entities: people, places, institutions, universities,
degrees, roles, dates, and sources. Use UPPER_SNAKE_CASE for entity types.
Be exhaustive — prefer high recall over high precision at this stage."""

RELATION_SYSTEM = """You are a medieval history expert building a knowledge graph.
Given a biographical notice and a list of already-identified entities,
extract all relations between those entities.

CRITICAL RULES:
- Only use entity ids from the provided entity list — never invent new ones.
- Only use relation types from the fixed enum provided in the schema.
- If no canonical type fits well, choose the closest one rather than inventing a new label.
- Provide a verbatim evidence string for every relation."""

def build_entity_prompt(notice_text: str) -> str:
    return f"{ENTITY_SYSTEM}\n\n--- NOTICE ---\n{notice_text}\n"

def build_relation_prompt(notice_text: str, entities: list[Entity]) -> str:
    entity_list = "\n".join(
        f"  - {e.id} ({e.type}): {e.label}" for e in entities
    )
    allowed = ", ".join(r.value for r in RelationType)
    return (
        f"{RELATION_SYSTEM}\n\n"
        f"Allowed relation types: {allowed}\n\n"
        f"--- ENTITIES ---\n{entity_list}\n\n"
        f"--- NOTICE ---\n{notice_text}\n"
    )


# ── 4. Model loading (reuse your existing quantised model) ───────────────────

def load_model(model_name: str = "google/gemma-3-27b-it"):
    """
    Load via Outlines' HuggingFace backend.
    Outlines wraps the model; your existing bitsandbytes config still applies
    via the model_kwargs passed here.
    """
    from transformers import BitsAndBytesConfig
    import torch

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = models.transformers(
        model_name,
        device="auto",
        model_kwargs={
            "quantization_config": bnb_config,
            "device_map": "auto",
        },
    )
    return model


# ── 5. Two-step constrained extraction ───────────────────────────────────────

class ConstrainedExtractor:
    """
    Drop-in replacement for your current LLM extraction step.
    Generates JSON that is guaranteed to match ExtractionOutput's schema.
    """

    def __init__(self, model):
        self.model = model
        # One generator per schema — Outlines compiles the FSM once and reuses it
        self._entity_gen  = generate.json(model, Entity)
        self._relation_gen = generate.json(model, Relation)
        self._full_gen     = generate.json(model, ExtractionOutput)

    def extract(self, notice_text: str) -> ExtractionOutput:
        """Full two-step pipeline: entities first, then relations."""

        # Step 1 — entity extraction (free types, constrained JSON structure)
        entity_prompt = build_entity_prompt(notice_text)
        # We use a list-of-Entity schema by wrapping in a container
        entities_raw: list[dict] = self._extract_entity_list(entity_prompt)
        entities = [Entity(**e) for e in entities_raw]

        # Step 2 — relation extraction (relation type is enum-constrained)
        relation_prompt = build_relation_prompt(notice_text, entities)
        relations_raw: list[dict] = self._extract_relation_list(relation_prompt, entities)
        relations = [Relation(**r) for r in relations_raw]

        return ExtractionOutput(entities=entities, relations=relations)

    def _extract_entity_list(self, prompt: str) -> list[dict]:
        """
        Outlines generates one object at a time; we loop until the model
        signals completion (empty label) or we hit max_entities.
        For simplicity here we generate the full output in one shot using
        the ExtractionOutput schema restricted to entities only.
        """
        class EntityList(BaseModel):
            entities: list[Entity]

        gen = generate.json(self.model, EntityList)
        result = gen(prompt, max_tokens=2048, temperature=0.7)
        return [e.model_dump() for e in result.entities]

    def _extract_relation_list(
        self, prompt: str, entities: list[Entity]
    ) -> list[dict]:
        """
        Relation types are enum-constrained: the FSM built by Outlines
        only allows tokens that spell out a valid RelationType value.
        """
        valid_ids = {e.id for e in entities}

        class RelationList(BaseModel):
            relations: list[Relation]

        gen = generate.json(self.model, RelationList)
        result = gen(prompt, max_tokens=2048, temperature=0.35)

        # Secondary guard: drop relations referencing unknown entity ids
        clean = [
            r for r in result.relations
            if r.subject in valid_ids and r.object in valid_ids
        ]
        return [r.model_dump() for r in clean]


# ── 6. Pipeline integration ───────────────────────────────────────────────────

def process_jsonl(
    input_path: str,
    output_path: str,
    model_name: str = "google/gemma-3-27b-it",
    cache_path: str  = "llm_cache.sqlite",
) -> None:
    """
    Process your studium_llm_ready_people.jsonl file notice by notice,
    writing constrained ExtractionOutput objects to a new JSONL.

    Your existing SQLite cache logic can wrap calls to `extractor.extract()`
    with (model, prompt, temperature) as the cache key — nothing changes there.
    """
    import sqlite3, hashlib, pathlib

    model     = load_model(model_name)
    extractor = ConstrainedExtractor(model)

    # Minimal cache (extend with your existing llm_cache logic)
    conn = sqlite3.connect(cache_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cache "
        "(key TEXT PRIMARY KEY, value TEXT)"
    )

    with open(input_path)  as fin, \
         open(output_path, "w") as fout:

        for line in fin:
            notice = json.loads(line)
            text   = notice.get("text") or notice.get("raw_text", "")
            key    = hashlib.sha256(
                f"{model_name}|{text}".encode()
            ).hexdigest()

            row = conn.execute(
                "SELECT value FROM cache WHERE key=?", (key,)
            ).fetchone()

            if row:
                result_dict = json.loads(row[0])
            else:
                result      = extractor.extract(text)
                result_dict = result.model_dump()
                conn.execute(
                    "INSERT INTO cache VALUES (?,?)",
                    (key, json.dumps(result_dict, ensure_ascii=False)),
                )
                conn.commit()

            fout.write(json.dumps(
                {"id": notice.get("reference", ""), **result_dict},
                ensure_ascii=False,
            ) + "\n")

    conn.close()
    print(f"Done. Results written to {output_path}")


# ── 7. Quick smoke test (no GPU needed) ──────────────────────────────────────

if __name__ == "__main__":
    # Validate the schema without loading the model
    sample = ExtractionOutput(
        entities=[
            Entity(
                id="robertus_gervasii",
                label="ROBERTUS Gervasii",
                type="PERSON",
                confidence=0.99,
                evidence="ROBERTUS Gervasii, dominicain...",
            ),
            Entity(
                id="paris",
                label="Paris",
                type="PLACE",
                confidence=0.99,
                evidence="étudié à Paris",
            ),
        ],
        relations=[
            Relation(
                subject="robertus_gervasii",
                relation=RelationType.STUDIED_AT,  # enum — no string drift
                object="paris",
                confidence=0.90,
                evidence="étudié à Paris",
            )
        ],
    )
    print(sample.model_dump_json(indent=2))

    # Uncomment to run the full pipeline:
    # process_jsonl(
    #     "studium_llm_ready_people.jsonl",
    #     "studium_kg_constrained.jsonl",
    # )
