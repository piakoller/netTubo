import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# --- Project-specific Imports (ensure they are in PYTHONPATH) ---
try:
    import config
    from data_loader import load_patient_data
    from logging_setup import setup_logging
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please run this script from the project root or ensure 'llmRecom' is in your PYTHONPATH.")
    exit(1)

# --- Configuration ---
# LLM Defaults
DEFAULT_LLM_MODEL = "gemma3:27b"
LLM_TEMPERATURE = 0.0

# Data Directories
BASE_PROJECT_DIR = Path("/home/pia/projects/netTubo")
EVAL_DATA_DIR = BASE_PROJECT_DIR / "data_for_evaluation/single_prompt"
# Source directory for all guidelines
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
logger = logging.getLogger("single_prompt_processor")


def _sanitize_tag_name(filename: str) -> str:
    """Converts a filename into a valid, clean XML-like tag name."""
    # Remove file extension
    name = Path(filename).stem
    # Replace spaces and hyphens with underscores
    name = re.sub(r'[\s-]', '_', name)
    # Remove any other non-alphanumeric characters (except underscores)
    name = re.sub(r'[^\w_]', '', name)
    # Convert to lowercase
    return name.lower()


def load_structured_guidelines(guideline_dir: Path) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """
    Recursively finds guideline files and organizes them by their source subdirectory.

    Returns:
        A tuple containing:
        - A dictionary structured by source: {'source_name': {'filename': 'content', ...}}
        - A flat list of all loaded filenames for logging.
    """
    structured_docs: Dict[str, Dict[str, str]] = {}
    loaded_files: List[str] = []

    for item in sorted(guideline_dir.iterdir()):
        source_name = item.name.lower()
        files_to_load = []

        if item.is_dir():
            # It's a subdirectory (e.g., ENET, ESMO)
            structured_docs[source_name] = {}
            files_to_load = list(item.glob("*.md")) + list(item.glob("*.mds"))
        elif item.is_file() and item.suffix in ['.md', '.mds']:
            # It's a file in the root directory (e.g., NCCN Guidelines.md)
            source_name = "root" # Special key for top-level files
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
    if not structured_docs:
        return ""

    context_parts = ["<guidelines_context>"]
    for source, files in structured_docs.items():
        # Use 'root' for top-level files, otherwise 'source_guideline'
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
    llm: OllamaLLM
) -> Tuple[Optional[Dict], Optional[str], Optional[str], Optional[float], Dict]:
    """Generates a single therapy recommendation for a patient using structured guidelines."""
    patient_data_string = format_patient_data_for_prompt(patient_data_dict, PATIENT_FIELDS_FOR_PROMPT)

    structured_guidelines, loaded_files = load_structured_guidelines(GUIDELINE_SOURCE_DIR)
    guidelines_context_string = format_guidelines_for_prompt(structured_guidelines)

    if not guidelines_context_string:
        logger.warning(f"No guideline context could be loaded for patient {patient_data_dict.get('id')}. Proceeding without it.")

    prompt_template_str = f"""
Du bist ein KI-Assistent, der eine Beurteilung und Therapieempfehlung für Patienten eines Tumorboards erstellen soll.
Deine Aufgabe ist es, die gegebenen Patienteninformationen zu analysieren, die bereitgestellten medizinischen Leitlinien zu konsultieren und eine fundierte Empfehlung auf Deutsch abzugeben.

Hier sind die Eingabedaten, die du analysieren sollst:

<patient_information>
{{patient_data_string}}
</patient_information>

{{guidelines_context_string}}

Folge diesen Schritten, um deine Aufgabe zu erfüllen:
1. Analysiere sorgfältig die Patienteninformationen.
2. Konsultiere die bereitgestellten Leitlinien im `<guidelines_context>`. Die Leitlinien sind nach ihrer Quelle (z.B. `<enet_guidelines>`, `<esmo_guidelines>`) strukturiert.
3. Erstelle eine Beurteilung der Patientensituation.
4. Entwickle eine detaillierte Therapieempfehlung.
5. Begründe deine Empfehlung mit klaren Verweisen auf die relevanten Leitlinien (nenne die Quelle und das spezifische Dokument) und individuelle Patientenfaktoren.

Strukturiere deine finale Antwort ausschließlich mit den folgenden Tags. Gib keine weiteren Erklärungen oder den Scratchpad aus.

{TAG_ASSESSMENT}
[Hier deine ausführliche Beurteilung der Patientensituation einfügen]
</{TAG_ASSESSMENT[1:]}

{TAG_RECOMMENDATION}
[Hier deine detaillierte Therapieempfehlung einfügen]
</{TAG_RECOMMENDATION[1:]}

{TAG_RATIONALE}
[Hier eine Begründung für deine Empfehlung basierend auf den Leitlinien und Patientenfaktoren einfügen]
</{TAG_RATIONALE[1:]}
"""

    prompt = PromptTemplate.from_template(prompt_template_str)
    chain = LLMChain(llm=llm, prompt=prompt)

    template_input = {
        "patient_data_string": patient_data_string,
        "guidelines_context_string": guidelines_context_string,
    }
    
    llm_input_for_log = {
        "prompt_text": prompt.format(**template_input),
        "attachments_used": loaded_files,
        "llm_kwargs": config.MODEL_KWARGS
    }

    start_time = time.perf_counter()
    try:
        logger.info(f"Generating recommendation for Patient ID {patient_data_dict.get('id')}. Attached files: {', '.join(loaded_files) or 'None'}")
        
        response = chain.invoke(template_input, **config.MODEL_KWARGS)
        duration = time.perf_counter() - start_time
        
        raw_response = response.get("text", "").strip()
        parsed_response = _parse_llm_response(raw_response)

        if not parsed_response.get("recommendation"):
             logger.warning(f"Could not extract recommendation from response for patient {patient_data_dict.get('id')}.")

        return parsed_response, raw_response, None, duration, llm_input_for_log

    except Exception as e:
        duration = time.perf_counter() - start_time
        error_msg = f"LLM generation failed: {e}"
        logger.error(error_msg, exc_info=True)
        return None, None, error_msg, duration, llm_input_for_log


def run_single_prompt_processing(
    llm_model: str,
    patient_data_file: Path,
    output_file: Optional[Path] = None,
    is_clinical_info_modified: bool = False
):
    """Main function to run the single-prompt processing pipeline."""
    logger.info(f"Starting processing with LLM: {llm_model}, Patient Data: {patient_data_file.name}")
    
    try:
        llm = OllamaLLM(model=llm_model, temperature=LLM_TEMPERATURE)
        llm.invoke("Hi") # Check if the model is available
    except Exception as e:
        logger.error(f"Failed to initialize or connect to LLM '{llm_model}'. Please ensure Ollama is running and the model is available. Error: {e}", exc_info=True)
        return

    if not output_file:
        sanitized_model_name = _sanitize_filename(llm_model)
        filename = f"structured_guideline_{sanitized_model_name}_modified_{is_clinical_info_modified}.json"
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

        parsed_rec, raw_resp, error, duration, llm_input = generate_single_recommendation(
            patient_dict, llm
        )

        all_results.append({
            "patient_id": str(patient_id),
            "patient_data_source_file": patient_data_file.name,
            "timestamp_processed": datetime.now().isoformat(),
            "llm_model_used": llm_model,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate therapy recommendations using a structured set of guidelines.")
    parser.add_argument(
        "--patient_data_file", type=Path, default=None,
        help="Optional: Path to the patient data Excel file. Overrides the default from config."
    )
    parser.add_argument(
        "--llm_model", type=str, default=DEFAULT_LLM_MODEL,
        help=f"LLM model to use. Default: {DEFAULT_LLM_MODEL}"
    )
    parser.add_argument(
        "--output_file", type=Path, default=None,
        help="Path to save the JSON results. If not set, a filename is auto-generated."
    )
    parser.add_argument(
        "--clinical_info_modified", action="store_true",
        help="Flag that 'clinical_info' or other context was modified. Is set automatically if --patient_data_file is used."
    )
    args = parser.parse_args()

    is_modified = args.clinical_info_modified
    if args.patient_data_file:
        patient_file = args.patient_data_file
        if not is_modified:
            logger.info("Using a custom patient data file, so 'clinical_info_modified' is automatically set to True.")
            is_modified = True
    else:
        patient_file = Path(config.TUBO_EXCEL_FILE_PATH)
    
    run_single_prompt_processing(
        llm_model=args.llm_model,
        patient_data_file=patient_file,
        output_file=args.output_file,
        is_clinical_info_modified=is_modified
    )