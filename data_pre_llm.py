import json
import os
import re

def clean(text):
    """Remove markup like %1504%, $Name$, *Place, £Institution, &Title&"""
    text = re.sub(r'%[\d\s:]*%', '', text)       # dates
    text = re.sub(r'[\$\*£&=]', '', text)          # special markers
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_values(field_list):
    if not field_list:
        return []
    return [clean(item['value']) for item in field_list if item.get('value')]

def record_to_text(record):
    parts = []
    
    # --- Identity ---
    name = extract_values(record.get('identity', {}).get('name', []))
    variants = extract_values(record.get('identity', {}).get('nameVariant', []))
    desc = extract_values(record.get('identity', {}).get('shortDescription', []))
    dates_activity = extract_values(record.get('identity', {}).get('datesOfActivity', []))
    status = extract_values(record.get('identity', {}).get('status', []))
    gender = extract_values(record.get('identity', {}).get('gender', []))

    primary_name = name[0] if name else record.get('title', 'Unknown')
    parts.append(f"PERSON: {primary_name}")
    if variants:
        parts.append(f"Also known as: {'; '.join(variants)}.")
    if desc:
        parts.append(f"Description: {desc[0]}.")
    if status:
        parts.append(f"Status: {status[0]}.")
    if gender:
        parts.append(f"Gender: {gender[0]}.")
    if dates_activity:
        parts.append(f"Active during: {dates_activity[0]}.")

    # --- Origin ---
    origin = record.get('origin', {})
    birthplace = extract_values(origin.get('birthPlace', []))
    diocese = extract_values(origin.get('diocese', []))
    if birthplace:
        parts.append(f"Born in: {birthplace[0]}.")
    if diocese:
        parts.append(f"Diocese of origin: {diocese[0]}.")

    # --- Curriculum ---
    curriculum = record.get('curriculum', {})
    universities = extract_values(curriculum.get('university', []))
    grades = extract_values(curriculum.get('grades', []))
    colleges = extract_values(curriculum.get('universityCollege', []))
    if universities:
        parts.append(f"Studied at: {'; '.join(universities)}.")
    if grades:
        parts.append(f"Academic grades obtained: {'; '.join(grades)}.")
    if colleges:
        parts.append(f"College affiliation: {'; '.join(colleges)}.")

    # --- Relational insertion ---
    relational = record.get('relationalInsertion', {})
    family = extract_values(relational.get('familyNetwork', []))
    student_prof = extract_values(relational.get('studentProfessorRelationships', []))
    social = extract_values(relational.get('socialClassOrigin', []))
    if family:
        parts.append(f"Family relationships: {' '.join(family)}.")
    if student_prof:
        parts.append(f"Academic relationships: {' '.join(student_prof)}.")
    if social:
        parts.append(f"Social origin: {social[0]}.")

    # --- Ecclesiastical career ---
    eccl = record.get('ecclesiasticalCareer', {})
    positions = extract_values(eccl.get('secularPosition', []))
    eccl_status = extract_values(eccl.get('ecclesiasticalStatus', []))
    regular_order = extract_values(eccl.get('regularOrder', []))
    regular_fn = extract_values(eccl.get('regularFunctions', []))
    other_fn = extract_values(eccl.get('otherFunctions', []))
    if eccl_status:
        parts.append(f"Ecclesiastical status: {'; '.join(eccl_status)}.")
    if regular_order:
        parts.append(f"Religious order: {'; '.join(regular_order)}.")
    if regular_fn:
        parts.append(f"Religious functions: {'; '.join(regular_fn)}.")
    if positions:
        parts.append(f"Ecclesiastical positions held: {'; '.join(positions)}.")
    if other_fn:
        parts.append(f"Other ecclesiastical functions: {'; '.join(other_fn)}.")

    # --- Professional career ---
    prof = record.get('professionalCareer', {})
    univ_fn = extract_values(prof.get('universityFunction', []))
    representation = extract_values(prof.get('representation', []))
    if univ_fn:
        parts.append(f"University functions: {'; '.join(univ_fn)}.")
    if representation:
        parts.append(f"Representation roles: {'; '.join(representation)}.")

    # --- Textual production ---
    textual = record.get('textualProduction', {})
    for field_key in textual:
        field = textual[field_key]
        if isinstance(field, dict) and 'opus' in field:
            for opus in field['opus']:
                titles = extract_values(opus.get('title', []))
                date_place = extract_values(opus.get('dateAndPlace', []))
                dedic = extract_values(opus.get('dedicaceOrAdress', []))
                if titles:
                    work_text = f"Authored work: '{titles[0]}'"
                    if date_place:
                        work_text += f", composed around {date_place[0]}"
                    if dedic:
                        work_text += f", dedicated to {dedic[0]}"
                    parts.append(work_text + ".")

    # --- Bibliography ---
    bib = record.get('bibliography', {})
    other_bases = extract_values(bib.get('otherBases', []))
    if other_bases:
        parts.append(f"Source: {other_bases[0]}.")

    return "\n".join(parts)


# --- Main conversion ---
input_path = "data/studium_parisiense_dataset.jsonl"
output_dir = "ragdata/input"
os.makedirs(output_dir, exist_ok=True)

count = 0
with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        text = record_to_text(record)
        ref = record.get('reference', record.get('_id', str(count)))
        out_path = os.path.join(output_dir, f"person_{ref}.txt")
        with open(out_path, 'w', encoding='utf-8') as out:
            out.write(text)
        count += 1

print(f"✅ Converted {count} records to {output_dir}/")
