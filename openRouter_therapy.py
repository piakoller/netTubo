# openRouter_therapy.py

import argparse
import json
import logging
import os
import requests
import time
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

from config import LLM_TEMPERATURE, MAX_TOKENS, TUBO_EXCEL_FILE_PATH
from shared_logic import (
    format_patient_data_for_prompt,
    load_structured_guidelines,
    format_guidelines_for_prompt,
    build_prompt,
    PATIENT_FIELDS_FOR_PROMPT,
    GUIDELINE_SOURCE_DIR,
    ADDITIONAL_CONTEXT,
    # NEW_NET_EVIDENCE,
    _sanitize_filename,
    PROMPT_VERSION
)

load_dotenv()
logger = logging.getLogger("run_openrouter")
logging.basicConfig(level=logging.INFO)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-pro"
API_KEY = os.getenv("OPENROUTER_API_KEY")
SYSTEM_MESSAGE = "Du bist ein hilfreicher medizinischer Assistent für ein Tumorboard."

def call_openrouter_api(model: str, prompt: str, api_key: str, temperature: float, max_tokens: int, system_message: str = SYSTEM_MESSAGE, max_retries: int = 3) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/piakoller/netTubo",
        "X-Title": "NET Tumorboard Assistant",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,  # Füge eine maximale Token-Länge hinzu
        "temperature": temperature,
        "stop": ["</begründung>"] 
    }
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Attempt {attempt + 1}: Calling OpenRouter API")
            # Debug-Logging hinzufügen
            logger.debug(f"Request URL: {OPENROUTER_URL}")
            logger.debug(f"Request Headers: {headers}")
            logger.debug(f"Request Data: {json.dumps(data, indent=2)}")
            
            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=data,
                timeout=30  # Timeout hinzufügen
            )
            
            # # Debug-Logging für die Antwort
            # logger.info(f"Response Status Code: {response.status_code}")
            # logger.debug(f"Response Headers: {dict(response.headers)}")
            
            if response.status_code == 400:
                error_message = response.json() if response.text else "No error details available"
                logger.error(f"Bad Request Error. Details: {error_message}")
                return f"ERROR: Bad Request - {error_message}"
            
            if response.status_code == 401:
                logger.error("Authentication failed. Please check your API key.")
                return "ERROR: Authentication failed - invalid API key"
            
            response.raise_for_status()
            response_json = response.json()
            
            if "choices" in response_json and len(response_json["choices"]) > 0:
                return response_json["choices"][0]["message"]["content"]
            else:
                logger.error(f"Unexpected API response structure: {response_json}")
                return "ERROR: Unexpected API response structure"
                
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API request failed (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response (attempt {attempt+1}): {e}")
            logger.error(f"Raw response: {response.text}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"Unexpected error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    
    return "ERROR: OpenRouter API call failed after all retries"

def main():
    parser = argparse.ArgumentParser(description="Generate therapy recommendations using OpenRouter API (no LangChain).")
    parser.add_argument("--llm_model", type=str, default=DEFAULT_OPENROUTER_MODEL)
    parser.add_argument("--patient_data_file", type=Path, default=None)
    parser.add_argument("--output_file", type=Path, default=None)
    parser.add_argument("--clinical_info_modified", action="store_true")
    args = parser.parse_args()

    if not API_KEY:
        logger.error("OPENROUTER_API_KEY not set in .env")
        return

    patient_file = args.patient_data_file or Path(TUBO_EXCEL_FILE_PATH)
    if ADDITIONAL_CONTEXT:
        structured_guidelines, loaded_files = load_structured_guidelines(
            guideline_dir=GUIDELINE_SOURCE_DIR,
            additional_dir=ADDITIONAL_CONTEXT,
        )
    # if NEW_NET_EVIDENCE:
    #     structured_guidelines, loaded_files = load_structured_guidelines(
    #         guideline_dir=GUIDELINE_SOURCE_DIR,
    #         new_net_evidence=NEW_NET_EVIDENCE
    #     )
    else:
        structured_guidelines, loaded_files = load_structured_guidelines(
            guideline_dir=GUIDELINE_SOURCE_DIR
        )
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
        prompt = build_prompt(patient_data_string, guidelines_context_string)
        guidelines_context_string = format_guidelines_for_prompt(structured_guidelines)
        prompt = build_prompt(patient_data_string, guidelines_context_string)
        raw_response = call_openrouter_api(
            model=args.llm_model,
            prompt=prompt,
            api_key=API_KEY,
            temperature=LLM_TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        all_results.append({
            "patient_id": str(patient_id),
            "patient_data_source_file": patient_file.name,
            "timestamp_processed": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "llm_model_used": args.llm_model,
            "prompt_version": PROMPT_VERSION,
            "llm_input": {
                "prompt_text": prompt,
                "attachments_used": loaded_files
            },
            "llm_raw_output": raw_response,
            "error": None if not raw_response.startswith("ERROR") else raw_response
        })
        time.sleep(1)

    output_file = args.output_file or (Path("./data_for_evaluation/singleprompt") / f"singleprompt_{_sanitize_filename(args.llm_model)}_prompt_{PROMPT_VERSION}.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Processing complete. All {len(all_results)} results saved to: {output_file}")

if __name__ == "__main__":
    main()