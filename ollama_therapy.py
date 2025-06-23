# ollama_therapy.py

import argparse
import logging
from pathlib import Path
import requests
import json
import time
import pandas as pd

import config
from shared_logic import (
    format_patient_data_for_prompt,
    load_structured_guidelines,
    format_guidelines_for_prompt,
    build_prompt,
    PATIENT_FIELDS_FOR_PROMPT,
    GUIDELINE_SOURCE_DIR,
    _sanitize_filename
)

DEFAULT_OLLAMA_MODEL = "gemma3:27b"
OLLAMA_HOST = "http://localhost:11434"
logger = logging.getLogger("run_ollama")
logging.basicConfig(level=logging.INFO)

def call_ollama_api(model: str, prompt: str, temperature: float = 0.0) -> str:
    data = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature
    }
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            headers={"Content-Type": "application/json"},
            data=json.dumps(data),
            stream=True  # Enable streaming to handle response properly
        )
        response.raise_for_status()
        
        # Handle streaming response
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_response = json.loads(line)
                    if 'response' in json_response:
                        full_response += json_response['response']
                    if json_response.get('done', False):
                        break
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON line: {e}. Line content: {line}")
                    continue
        
        return full_response
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API request failed: {e}")
        return f"ERROR: API request failed - {str(e)}"
    except Exception as e:
        logger.error(f"Ollama API call failed: {e}")
        return f"ERROR: {str(e)}"
    
def main():
    parser = argparse.ArgumentParser(description="Generate recommendations using a local Ollama model.")
    parser.add_argument("--llm_model", type=str, default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--patient_data_file", type=Path, default=None)
    parser.add_argument("--output_file", type=Path, default=None)
    parser.add_argument("--clinical_info_modified", action="store_true")
    args = parser.parse_args()

    is_modified = args.clinical_info_modified
    patient_file = args.patient_data_file or Path(config.TUBO_EXCEL_FILE_PATH)
    if args.patient_data_file and not is_modified:
        logger.info("Custom patient file used -> setting clinical_info_modified=True")
        is_modified = True

    structured_guidelines, loaded_files = load_structured_guidelines(GUIDELINE_SOURCE_DIR)
    guidelines_context_string = format_guidelines_for_prompt(structured_guidelines)

    df_patients = pd.read_excel(str(patient_file))
    if df_patients is None or df_patients.empty:
        logger.error(f"No patient data loaded from {patient_file}. Exiting.")
        return

    all_results = []
    patient_ids = [pid for pid in df_patients["ID"].unique() if pid and str(pid).strip()]
    total_patients = len(patient_ids)
    logger.info(f"Found {total_patients} unique patients to process.")

    for i, patient_id in enumerate(patient_ids, 1):
        logger.info(f"--- Processing patient {i}/{total_patients} (ID: {patient_id}) ---")
        patient_row = df_patients[df_patients["ID"] == patient_id].iloc[0]
        patient_dict = patient_row.to_dict()
        patient_data_string = format_patient_data_for_prompt(patient_dict, PATIENT_FIELDS_FOR_PROMPT)
        structured_guidelines, loaded_files = load_structured_guidelines(GUIDELINE_SOURCE_DIR)
        guidelines_context_string = format_guidelines_for_prompt(structured_guidelines)
        prompt = build_prompt(patient_data_string, guidelines_context_string)
        raw_response = call_ollama_api(
            model=args.llm_model,
            prompt=prompt
        )
        all_results.append({
            "patient_id": str(patient_id),
            "patient_data_source_file": patient_file.name,
            "timestamp_processed": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "llm_model_used": args.llm_model,
            "clinical_info_modified": is_modified,
            "llm_input": {
                "prompt_text": prompt,
                "attachments_used": loaded_files
            },
            "llm_raw_output": raw_response,
            "error": None if not raw_response.startswith("ERROR") else raw_response
        })
        time.sleep(1)

    output_file = args.output_file or (Path("data_for_evaluation/singleprompt") / f"singleprompt_{_sanitize_filename(args.llm_model)}_modified_{is_modified}.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Processing complete. All {len(all_results)} results saved to: {output_file}")

if __name__ == "__main__":
    main()