# shared_logic.py

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel

# --- Project-specific Imports (ensure they are in PYTHONPATH) ---
try:
    import config
    from data_loader import load_patient_data
    from logging_setup import setup_logging
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please run this script from the project root or ensure 'llmRecom' is in your PYTHONPATH.")
    exit(1)

# --- Shared Configuration ---
LLM_TEMPERATURE = 0.0

# Data Directories
BASE_PROJECT_DIR = Path("/home/pia/projects/netTubo")
EVAL_DATA_DIR = BASE_PROJECT_DIR / "data_for_evaluation/single_prompt"
GUIDELINE_SOURCE_DIR = BASE_PROJECT_DIR / "data/guidelines/mds/"

# Patient data fields to include in the prompt
PATIENT_FIELDS_FOR_PROMPT = [
    "id", "beschreibung", "tumorboard_datum", "main_diagnosis_text", "Fragestellung"
]

# Constants for prompt tags
TAG_ASSESSMENT = "<beurteilung>"
TAG_RECOMMENDATION = "<therapieempfehlung>"
TAG_RATIONALE = "<begründung>"

# --- Logging Setup ---
setup_logging()
logger = logging.getLogger("shared_logic")


def _sanitize_tag_name(filename: str) -> str:
    """Converts a filename into a valid, clean XML-like tag name."""
    name = Path(filename).stem
    name = re.sub(r'[\s-]', '_', name)
    name = re.sub(r'[^\w_]', '', name)
    return name.lower()


def load_structured_guidelines(guideline_dir: Path) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """Recursively finds guideline files and organizes them by their source subdirectory."""
    structured_docs: Dict[str, Dict[str, str]] = {}
    loaded_files: List[str] = []
    for item in sorted(guideline_dir.iterdir()):
        source_name = item.name.lower()
        files_to_load = []
        if item.is_dir():
            structured_docs[source_name] = {}
            files_to_load = list(item.glob("*.md")) + list(item.glob("*.mds"))
        elif item.is_file() and item.suffix in ['.md', '.mds']:
            source_name = "root"
            if source_name not in structured_docs:
                structured_docs[source_name] = {}
            files_to_load = [item]
        for file in sorted(files_to_load):
            try:
                content = file.read_text(encoding='utf-8')
                structured_docs[source_name][file.name] = content
                loaded_files.append(str(file.relative_to(guideline_dir.parent)))
            except Exception as e:
                logger.error(f"Error reading file {file}: {e}")
    return structured_docs, loaded_files


def format_guidelines_for_prompt(structured_docs: Dict[str, Dict[str, str]]) -> str:
    """Formats the structured guideline dictionary into a nested XML-like string."""
    if not structured_docs: return ""
    context_parts = ["<guidelines_context>"]
    for source, files in structured_docs.items():
        source_tag = f"{source}_guidelines" if source != "root" else "general_guidelines"
        context_parts.append(f"  <{source_tag}>")
        for filename, content in files.items():
            file_tag = _sanitize_tag_name(filename)
            context_parts.append(f"    <{file_tag}>\n{content}\n    </{file_tag}>")
        context_parts.append(f"  </{source_tag}>")
    context_parts.append("</guidelines_context>")
    return "\n".join(context_parts)


def format_patient_data_for_prompt(patient_row: Dict, fields: List[str]) -> str:
    """Formats patient data into a string for the LLM prompt."""
    lines = ["Patienteninformationen:"]
    for field in fields:
        value = patient_row.get(field)
        if value and str(value).strip():
            field_name_pretty = field.replace("_", " ").title()
            lines.append(f"- {field_name_pretty}: {str(value)}")
    return "\n".join(lines)


def _parse_llm_response(response_text: str) -> Dict[str, Optional[str]]:
    """Extracts content from specified tags in the LLM's response."""
    def extract_tag_content(tag: str, text: str) -> Optional[str]:
        pattern = f"{tag}(.*?)</{tag[1:]}"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None
    return {
        "assessment": extract_tag_content(TAG_ASSESSMENT, response_text),
        "recommendation": extract_tag_content(TAG_RECOMMENDATION, response_text),
        "rationale": extract_tag_content(TAG_RATIONALE, response_text),
    }


def _sanitize_filename(name: str) -> str:
    """Replaces characters in a string to make it a valid filename component."""
    return name.replace(":", "_").replace("/", "_").replace(".", "_")


def generate_single_recommendation(
    patient_data_dict: Dict,
    llm: BaseLanguageModel
) -> Tuple[Optional[Dict], Optional[str], Optional[str], Optional[float], Dict]:
    """Generates a single therapy recommendation for a patient using a provided LLM instance."""
    patient_data_string = format_patient_data_for_prompt(patient_data_dict, PATIENT_FIELDS_FOR_PROMPT)
    structured_guidelines, loaded_files = load_structured_guidelines(GUIDELINE_SOURCE_DIR)
    guidelines_context_string = format_guidelines_for_prompt(structured_guidelines)

    if not guidelines_context_string:
        logger.warning(f"No guideline context could be loaded for patient {patient_data_dict.get('id')}. Proceeding without it.")

    prompt_template_str = """
Du bist ein KI-Assistent, der eine Beurteilung und Therapieempfehlung für Patienten eines Tumorboards erstellen soll.
Deine Aufgabe ist es, die gegebenen Patienteninformationen zu analysieren, die bereitgestellten medizinischen Leitlinien zu konsultieren und eine fundierte Empfehlung auf Deutsch abzugeben.

**Wichtige Regeln für deine Antwort:**
1.  **Antworte ausschließlich auf Basis der Informationen in `<patient_information>` und `<guidelines_context>`. Verwende kein externes Wissen.**
2.  **Erfinde niemals Fakten, Diagnosen oder Testergebnisse, die nicht explizit im Kontext erwähnt werden.**
3.  Wenn eine für die Empfehlung wichtige Information fehlt, weise in deiner Beurteilung explizit darauf hin

<patient_information>
{patient_data_string}
</patient_information>
{guidelines_context_string}

<scratchpad>
- Wichtigste Punkte aus den Patienteninformationen
- Relevante Abschnitte aus Leitlinien
- Vorläufige Beurteilung der Patientensituation
- Mögliche Therapieoptionen
- Argumente für und gegen verschiedene Therapieansätze
</scratchpad>

<beurteilung>
[Hier deine ausführliche Beurteilung der Patientensituation einfügen]
</beurteilung>
<therapieempfehlung>
[Hier deine detaillierte Therapieempfehlung einfügen]
</therapieempfehlung>
<begründung>
[Hier eine Begründung für deine Empfehlung basierend auf den Leitlinien und Patientenfaktoren einfügen]
</begründung>
"""
    prompt = PromptTemplate(
        template=prompt_template_str,
        input_variables=["patient_data_string", "guidelines_context_string"]
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    template_input = {
    "patient_data_string": patient_data_string,
    "guidelines_context_string": guidelines_context_string
    }

    llm_input_for_log = {
        "prompt_text": prompt.format(**template_input),
        "attachments_used": loaded_files,
        "llm_kwargs": config.MODEL_KWARGS
    }

    start_time = time.perf_counter()
    
    try:
        logger.info(f"Generating recommendation for Patient ID {patient_data_dict.get('ID')}. Attached files: {', '.join(loaded_files) or 'None'}")
        response = chain.invoke(template_input, **config.MODEL_KWARGS)
        duration = time.perf_counter() - start_time
        raw_response = response.get("text", "").strip()
        parsed_response = _parse_llm_response(raw_response)
        # if not parsed_response.get("recommendation"):
        #      logger.warning(f"Could not extract recommendation from response for patient {patient_data_dict.get('ID')}.")
        return parsed_response, raw_response, None, duration, llm_input_for_log
    except Exception as e:
        duration = time.perf_counter() - start_time
        error_msg = f"LLM generation failed: {e}"
        logger.error(error_msg, exc_info=True)
        return None, None, error_msg, duration, llm_input_for_log


def run_processing_pipeline(
    llm: BaseLanguageModel,
    llm_model_name: str,
    patient_data_file: Path,
    output_file: Optional[Path] = None,
    is_clinical_info_modified: bool = False
):
    """Main function to run the single-prompt processing pipeline with a given LLM."""
    logger.info(f"Starting processing with LLM: {llm_model_name}, Patient Data: {patient_data_file.name}")

    if not output_file:
        sanitized_model_name = _sanitize_filename(llm_model_name)
        filename = f"singleprompt_{sanitized_model_name}_modified_{is_clinical_info_modified}.json"
        output_file = EVAL_DATA_DIR / filename
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output will be saved to: {output_file}")

    df_patients = load_patient_data(str(patient_data_file))
    if df_patients is None or df_patients.empty:
        logger.error(f"No patient data loaded from {patient_data_file}. Exiting.")
        return

    all_results = []
    patient_ids = [pid for pid in df_patients["ID"].unique() if pid and str(pid).strip()]
    total_patients = len(patient_ids)
    logger.info(f"Found {total_patients} unique patients to process.")

    for i, patient_id in enumerate(patient_ids, 1):
        logger.info(f"--- Processing patient {i}/{total_patients} (ID: {patient_id}) ---")
        patient_row = df_patients[df_patients["ID"] == patient_id].iloc[0]
        patient_dict = patient_row.to_dict()
        parsed_rec, raw_resp, error, duration, llm_input = generate_single_recommendation(patient_dict, llm)
        all_results.append({
            "patient_id": str(patient_id),
            "patient_data_source_file": patient_data_file.name,
            "timestamp_processed": datetime.now().isoformat(),
            "llm_model_used": llm_model_name,
            "clinical_info_modified": is_clinical_info_modified,
            "llm_input": llm_input,
            "llm_raw_output": raw_resp,
            "llm_parsed_output": parsed_rec,
            "llm_generation_time_s": f"{duration:.4f}" if duration else None,
            "error": error
        })
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Processing complete. All {len(all_results)} results saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to write results to {output_file}: {e}", exc_info=True)