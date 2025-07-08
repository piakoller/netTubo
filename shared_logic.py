# shared_logic.py

import json
import logging
import warnings
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel

warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
logging.getLogger("streamlit").setLevel(logging.ERROR)

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
BASE_PROJECT_DIR = Path("C:/Users/pia/OneDrive - Universitaet Bern/Projects/NetTubo")
# BASE_PROJECT_DIR = Path("/home/pia/projects/netTubo")

EVAL_DATA_DIR = BASE_PROJECT_DIR / "netTubo/data_for_evaluation/single_prompt"

# 1.0 using only one ESMO and one ENET Guideline as Context!!
GUIDELINE_SOURCE_DIR = BASE_PROJECT_DIR / "netTubo/data/guidelines/1-0_data_singleprompt/mds"

# 1.1 adding one press release and one newer study to the context
ADDITIONAL_CONTEXT = False
# ADDITIONAL_CONTEXT = BASE_PROJECT_DIR / "data/guidelines/1-1_data_singleprompt"

# 1.2 Manual search of relevant studies of N studies <= context size
# NEW_NET_EVIDENCE = False
NEW_NET_EVIDENCE = BASE_PROJECT_DIR / "New_NET_evidence/New_NET_evidence/mds"

# PROMPT_FILE_PATH = BASE_PROJECT_DIR / "prompts/prompt_v3_1-1.txt"
PROMPT_FILE_PATH = BASE_PROJECT_DIR / "prompts/prompt_v3_1-2.txt"
print(f'Promptversion: {PROMPT_FILE_PATH}')

# Patient data fields to include in the prompt
PATIENT_FIELDS_FOR_PROMPT = [
    "id", "tumorboard_datum", "main_diagnosis_text", "Fragestellung"
]

# Constants for prompt tags
TAG_ASSESSMENT = "<beurteilung>"
TAG_RECOMMENDATION = "<therapieempfehlung>"
TAG_RATIONALE = "<begründung>"

# --- Logging Setup ---
setup_logging()
logger = logging.getLogger("shared_logic")

def get_prompt_version_from_path(prompt_path: Path) -> str:
    """Extract prompt version from prompt file path."""
    match = re.search(r'prompt_(v\d+(?:_[\d-]+)?)\.txt$', str(prompt_path))
    return match.group(1) if match else "unknown"

PROMPT_VERSION = get_prompt_version_from_path(PROMPT_FILE_PATH)

def _sanitize_tag_name(filename: str) -> str:
    """Converts a filename into a valid, clean XML-like tag name."""
    name = Path(filename).stem
    name = re.sub(r'[\s-]', '_', name)
    name = re.sub(r'[^\w_]', '', name)
    return name.lower()

def load_structured_guidelines(guideline_dir: Path, additional_dir: Optional[Path] = None) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """
    Recursively finds guideline files and organizes them by their source subdirectory.
    
    Args:
        guideline_dir: Primary directory containing guideline files
        additional_dir: Optional additional directory with more guidelines
    """
    structured_docs: Dict[str, Dict[str, str]] = {}
    loaded_files: List[str] = []
    
    # Helper function to process a directory
    def process_directory(dir_path: Path, base_dir: Path) -> None:
        dir_path = Path(dir_path)
        if not dir_path.exists():
            logger.warning(f"Guidelines directory {dir_path} does not exist")
            return
            
        for item in sorted(dir_path.iterdir()):
            source_name = item.name.lower()
            files_to_load = []
            if item.is_dir():
                if source_name not in structured_docs:
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
                    loaded_files.append(str(file.relative_to(base_dir.parent)))
                except Exception as e:
                    logger.error(f"Error reading file {file}: {e}")
    
    # Process primary directory
    process_directory(guideline_dir, guideline_dir)
    
    # Process additional directory if provided
    if additional_dir:
        process_directory(additional_dir, additional_dir)
    
    return structured_docs, loaded_files

def format_patient_data_for_prompt(patient_row: Dict, fields: List[str]) -> str:
    """Formats patient data into a string for the LLM prompt."""
    lines = ["Patienteninformationen:"]
    field_map = {k.lower(): k for k in patient_row.keys()}  # Create case-insensitive field mapping
    
    for field in fields:
        actual_field = field_map.get(field.lower())  # Try to find the actual field name
        if actual_field:
            value = patient_row.get(actual_field)
            if value and str(value).strip():
                field_name_pretty = field.replace("_", " ").title()
                lines.append(f"- {field_name_pretty}: {str(value)}")
    return "\n".join(lines)

def build_prompt(patient_data_string: str, guidelines_context_string: str) -> str:
    """Builds the complete prompt with patient data and guidelines."""

    # Read the prompt template from the file
    try:
        with open(PROMPT_FILE_PATH, "r", encoding="utf-8") as f:
            prompt_template = f.read()
    except Exception as e:
        logger.error(f"Failed to read the prompt file {PROMPT_FILE_PATH}: {e}")
        raise

    # Format the prompt with the provided variables
    formatted_prompt = prompt_template.format(
        patient_data_string=patient_data_string,
        guidelines_context_string=guidelines_context_string
    )
    return formatted_prompt

def generate_single_recommendation(
    patient_data_dict: Dict,
    llm: BaseLanguageModel
) -> Tuple[Optional[Dict], Optional[str], Optional[str], Optional[float], Dict]:
    """Generates a single therapy recommendation for a patient using a provided LLM instance."""
    # Get patient ID in a case-insensitive way
    patient_id = next((str(v) for k, v in patient_data_dict.items() 
                      if k.lower() == "id"), "unknown")
    
    patient_data_string = format_patient_data_for_prompt(patient_data_dict, PATIENT_FIELDS_FOR_PROMPT)
    structured_guidelines, loaded_files = load_structured_guidelines(GUIDELINE_SOURCE_DIR)

    additional_structured = None
    if NEW_NET_EVIDENCE:
        additional_structured, additional_loaded_files = load_structured_guidelines(Path(NEW_NET_EVIDENCE))
        loaded_files.extend(additional_loaded_files)

    guidelines_context_string = format_guidelines_for_prompt(
        structured_docs=structured_guidelines,
        additional_structured_docs=additional_structured,
        additional_dir=Path(NEW_NET_EVIDENCE) if NEW_NET_EVIDENCE else None
    )

    if not guidelines_context_string:
        logger.warning(f"No guideline context could be loaded for patient {patient_id}. Proceeding without it.")

    # Use the build_prompt function instead of the template string
    formatted_prompt = build_prompt(patient_data_string, guidelines_context_string)
    
    prompt = PromptTemplate(
        template=formatted_prompt,
        input_variables=[]  # No input variables needed as we've already formatted the prompt
    )

    chain = prompt | llm

    llm_input_for_log = {
        "prompt_text": formatted_prompt,
        "attachments_used": loaded_files,
        "llm_kwargs": getattr(config, "MODEL_KWARGS", {})
    }

    start_time = time.perf_counter()
    
    try:
        logger.info(f"Generating recommendation for Patient ID {patient_id}. Attached files: {', '.join(loaded_files) or 'None'}")
        
        response = chain.invoke({})  # Empty dict since we've pre-formatted the prompt
        duration = time.perf_counter() - start_time
        
        if hasattr(response, 'content'):
            raw_response = response.content.strip()
        elif isinstance(response, str):
            raw_response = response.strip()
        elif isinstance(response, dict) and 'text' in response:
            raw_response = response['text'].strip()
        else:
            raw_response = str(response).strip()
        
        parsed_response = _parse_llm_response(raw_response)
        
        return parsed_response, raw_response, None, duration, llm_input_for_log
        
    except Exception as e:
        duration = time.perf_counter() - start_time
        error_msg = f"LLM generation failed: {e}"
        logger.error(error_msg, exc_info=True)
        return None, None, error_msg, duration, llm_input_for_log

def format_guidelines_for_prompt(
    structured_docs: Dict[str, Dict[str, str]],
    additional_structured_docs: Optional[Dict[str, Dict[str, str]]] = None,
    additional_dir: Optional[Path] = None
) -> str:
    """Formats both main and additional structured guideline dictionaries into XML-like tagged text."""
    context_parts = ["<guidelines_context>"]
    
    # Add guidelines from the main guideline_dir
    for source, files in structured_docs.items():
        for filename, content in files.items():
            file_tag = _sanitize_tag_name(filename)
            context_parts.append(f"<{file_tag}>\n{content}\n</{file_tag}>")
    
    context_parts.append("</guidelines_context>")

    # Add NEW_NET_EVIDENCE context with individual file tags
    if additional_structured_docs and additional_dir:
        context_parts.append("<new_net_evidence>")
        for source, files in additional_structured_docs.items():
            for filename, content in files.items():
                # Use the original filename (without extension) as tag, preserving spaces and hyphens
                file_tag = Path(filename).stem
                context_parts.append(f"<{file_tag}>\n{content}\n</{file_tag}>")
        context_parts.append("</new_net_evidence>")

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

    return all_results