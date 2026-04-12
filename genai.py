# -*- coding: utf-8 -*-
"""
Extraire entités/relations depuis un fichier jsonl d'individus avec
le modèle Google Generative API (gemma).

Améliorations :
 - prompt non restrictif (le LLM propose ses propres types)
 - écriture incrémentale (append + fsync)
 - parsing robuste des objets JSON retournés
 - prise en charge de deux arguments CLI :
      1) ligne de début
      2) ligne de fin
"""

import json
import time
import re
import os
import sys
from pathlib import Path

import google.generativeai as genai

# --- CONFIG ---
API_KEY = "AIzaSyDjWVz_stDcg2g-C3jTK06VshXTtnVh1Oo"
MODEL_NAME = "gemma-3-27b-it"   # ou autre modèle que tu utilises
INPUT_FILE = "../studium_llm_ready_people.jsonl"
ENTITIES_FILE = "entities_1000.jsonl"
RELATIONS_FILE = "relations_1000.jsonl"
MAX_RETRIES = 5
SLEEP_BETWEEN_REQUESTS = 1.5  # secondes (ajuste selon quota)
# ----------------

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

PROMPT_TEMPLATE = """
Tu es un extracteur d'information. On te fournit ci-dessous les données textuelles
d'un enregistrement (un individu du fichier). TA TÂCHE : EXTRAIRE TOUTES les
ENTITÉS possibles et TOUTES les RELATIONS possibles **présentes ou implicites**
dans ce texte. Ne limite pas les types d'entités ni les types de relations :
laisse le modèle proposer des types (ex : PERSON, PLACE, UNIVERSITY, DEGREE,
DATE, ROLE, SOURCE, NOTE, COLLEGE, DIOCESE, etc. — mais d'autres types sont
acceptables). Fournis aussi pour chaque item des éléments de preuve ("evidence")
extraits du texte.

Format de sortie demandé (obligatoire) :
Réponds **SEULEMENT** par deux objets JSON (sans markdown, sans explication),
séparés par exactement une ligne vide. Le premier objet correspond aux entités,
le deuxième aux relations.

Exemples (illustratifs — tu peux ajouter n'importe quel champ) :

1) Objet "entités" :
{
  "record_id": "<id_ou_reference_fournie>",
  "subject": "<champ principal/nom si présent>",
  "entities": [
    {
      "name": "Anselinus GALLI",
      "type": "PERSON",
      "confidence": 0.92,
      "evidence": ["name: ANCELINUS Galli", "nameVariant: Anselinus GALLI"],
      "attributes": {"gender": "male", "datesOfActivity": "1435"}
    },
    ...
  ]
}

2) Objet "relations" :
{
  "record_id": "<id_ou_reference_fournie>",
  "relations": [
    {
      "source": "Anselinus GALLI",
      "target": "Paris",
      "type": "STUDIED_AT",
      "confidence": 0.88,
      "evidence": ["university: Paris 1435-1435", "Bachelier en décret (Paris) en 1435"],
      "attributes": {"date": "1435"}
    },
    ...
  ]
}

Contraintes importantes :
- N'invente PAS d'informations : n'ajoute que ce qui est raisonnablement inféré
  ou explicitement présent dans le texte. Si incertain, indique "confidence" faible.
- Les deux objets JSON doivent être valides. Le premier objet = entités,
  le deuxième = relations.
- Inclure "record_id" (utilise la clé "reference" si elle existe dans l'entrée,
  sinon fournis un identifiant basé sur la position/ligne).
- Pour les dates, fournis la forme la plus explicite extraite (ex : "1435",
  "27 avril 1435" si disponible).
- Fournis des preuves textuelles ("evidence") pour chaque entité / relation.

Voici l'enregistrement (fournis tel quel ci-dessous) :
{data}
"""

# --- utilitaires ---

def balanced_json_objects(text, expected=2):
    """
    Extrait les premiers `expected` objets JSON complets du texte, en cherchant
    accolades équilibrées. Retourne liste de strings (ou [] si pas trouvés).
    """
    objs = []
    start = None
    depth = 0
    in_str = False
    esc = False

    for i, ch in enumerate(text):
        if ch == '"' and not esc:
            in_str = not in_str

        if ch == "\\" and not esc:
            esc = True
        else:
            esc = False

        if in_str:
            continue

        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                objs.append(text[start:i + 1])
                start = None
                if len(objs) >= expected:
                    break

    return objs


def clean_model_text(text):
    text = re.sub(r'```(?:json)?', '', text)
    return text.strip()


def safe_json_load(s):
    try:
        return json.loads(s)
    except Exception:
        return None


def fsync_and_flush(fh):
    fh.flush()
    try:
        os.fsync(fh.fileno())
    except Exception:
        pass


# --- main ---
def parse_and_process(input_file, entities_file, relations_file, start_line, end_line):
    Path(os.path.dirname(entities_file) or ".").mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(relations_file) or ".").mkdir(parents=True, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(entities_file, 'a', encoding='utf-8') as ent_out, \
         open(relations_file, 'a', encoding='utf-8') as rel_out:

        for idx, raw in enumerate(infile):
            if idx < start_line:
                continue
            if idx > end_line:
                print(f"Fin atteinte à la ligne {idx}.")
                break

            line = raw.strip()
            if not line:
                print(f"[{idx}] Ligne vide -> skip")
                continue

            record_id = f"line_{idx}"
            try:
                parsed_line = json.loads(line)
                if isinstance(parsed_line, dict):
                    if "reference" in parsed_line:
                        record_id = str(parsed_line["reference"])
                    elif "id" in parsed_line:
                        record_id = str(parsed_line["id"])
                    elif "reference_id" in parsed_line:
                        record_id = str(parsed_line["reference_id"])
                    elif "name" in parsed_line:
                        record_id = f"{parsed_line.get('name')}_{idx}"
                pretty_input = json.dumps(parsed_line, ensure_ascii=False, indent=2)
            except Exception:
                parsed_line = None
                pretty_input = line

            prompt = PROMPT_TEMPLATE.replace("{data}", pretty_input)

            attempt = 0
            success = False

            while attempt < MAX_RETRIES and not success:
                attempt += 1
                try:
                    response = model.generate_content(prompt)
                    text = response.text if hasattr(response, 'text') else str(response)
                    text = clean_model_text(text)

                    objs = balanced_json_objects(text, expected=2)

                    if len(objs) < 2:
                        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
                        if len(parts) >= 2:
                            candidates = []
                            for p in parts[:3]:
                                j = None
                                if p.startswith('{'):
                                    j = p
                                else:
                                    found = balanced_json_objects(p, expected=1)
                                    if found:
                                        j = found[0]
                                if j:
                                    candidates.append(j)
                                if len(candidates) >= 2:
                                    break
                            objs = candidates

                    if len(objs) < 2:
                        raise ValueError(
                            f"Impossible d'extraire 2 objets JSON à partir de la réponse "
                            f"(attempt {attempt}). Réponse brute:\n{text[:1000]}"
                        )

                    ent_obj_raw, rel_obj_raw = objs[0], objs[1]

                    ent_obj = safe_json_load(ent_obj_raw)
                    rel_obj = safe_json_load(rel_obj_raw)

                    if ent_obj is None or rel_obj is None:
                        raise ValueError("Parsing JSON échoué après extraction.")

                    if "record_id" not in ent_obj:
                        ent_obj["record_id"] = record_id
                    if "record_id" not in rel_obj:
                        rel_obj["record_id"] = record_id

                    ent_out.write(json.dumps(ent_obj, ensure_ascii=False) + "\n")
                    fsync_and_flush(ent_out)

                    rel_out.write(json.dumps(rel_obj, ensure_ascii=False) + "\n")
                    fsync_and_flush(rel_out)

                    print(f"[{idx}] OK record_id={ent_obj['record_id']} (attempt {attempt})")
                    success = True

                except Exception as e:
                    print(f"[{idx}] Erreur attempt {attempt}: {e}")
                    if attempt < MAX_RETRIES:
                        backoff = 2 ** attempt
                        print(f" -> retry après {backoff}s")
                        time.sleep(backoff)
                    else:
                        ent_out.write(json.dumps({
                            "record_id": record_id,
                            "error": str(e)
                        }, ensure_ascii=False) + "\n")
                        fsync_and_flush(ent_out)

                        rel_out.write(json.dumps({
                            "record_id": record_id,
                            "error": str(e)
                        }, ensure_ascii=False) + "\n")
                        fsync_and_flush(rel_out)

                        print(f"[{idx}] Échec définitif, on avance au suivant.")
                        break

            time.sleep(SLEEP_BETWEEN_REQUESTS)

    print("Traitement terminé.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <start_line> <end_line>")
        sys.exit(1)

    try:
        start_line = int(sys.argv[1])
        end_line = int(sys.argv[2])
    except ValueError:
        print("Erreur : start_line et end_line doivent être des entiers.")
        sys.exit(1)

    if start_line < 0 or end_line < 0:
        print("Erreur : start_line et end_line doivent être >= 0.")
        sys.exit(1)

    if start_line > end_line:
        print("Erreur : start_line doit être <= end_line.")
        sys.exit(1)

    parse_and_process(INPUT_FILE, ENTITIES_FILE, RELATIONS_FILE, start_line, end_line)
