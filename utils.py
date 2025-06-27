import json
import os
import pandas as pd
from datetime import datetime
import logging
import glob
from pymongo import MongoClient
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional

load_dotenv()

MONGO_URI = os.environ.get("MONGO_URI")
# MONGO_URI = os.getenv("MONGO_URI")

try:
    client = MongoClient(MONGO_URI) if MONGO_URI else None
    db = client["llmTubo_eval_db"] if client is not None else None
    collection = db["expert_evaluations_net"] if db is not None else None
except Exception as e:
    client = None
    db = None
    collection = None
    logging.warning(f"MongoDB connection failed: {e}. Evaluations will only be saved locally.")

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

script_dir = os.path.dirname(os.path.abspath(__file__))

# NET specific configuration
PATIENT_DATA = os.path.join(script_dir, "data", "NET Tubo v2.xlsx")
RESULTS_DIRS = [
    os.path.join(script_dir, "data_for_evaluation", "singleprompt"),
    os.path.join(script_dir, "data_for_evaluation", "multiagent")
]
EVALUATION_RESULTS_SAVE_DIR = os.path.join(script_dir, "evaluations_completed_comparative")

try:
    os.makedirs(EVALUATION_RESULTS_SAVE_DIR, exist_ok=True)
except OSError as e:
    logger.error(f"Could not create directory {EVALUATION_RESULTS_SAVE_DIR}: {e}")

# --- Caching for loaded data ---
_all_json_data_cache: Dict[str, List[dict]] = {}
_patient_ids_cache: List[str] = []

def load_all_json_files() -> Dict[str, List[dict]]:
    """
    Load all JSON files from the results directories.
    Returns a dictionary with filename as key and list of patient entries as value.
    """
    global _all_json_data_cache
    if _all_json_data_cache:
        return _all_json_data_cache
    
    all_json_files = []
    for res_dir in RESULTS_DIRS:
        if os.path.isdir(res_dir):
            # Use recursive=True to find files in subdirectories if needed, though current structure doesn't require it.
            all_json_files.extend(glob.glob(os.path.join(res_dir, "*.json")))
    
    if not all_json_files:
        logger.error(f"No JSON files found in directories: {RESULTS_DIRS}")
        return {}
    
    data_by_file = {}
    
    for json_file_path in all_json_files:
        filename_only = os.path.basename(json_file_path)
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    data_by_file[filename_only] = data
                    logger.info(f"Loaded {len(data)} entries from {filename_only}")
                else:
                    logger.warning(f"File {filename_only} does not contain a list. Skipping.")
                    
        except Exception as e:
            logger.error(f"Error loading file {filename_only}: {e}")
            continue
    
    _all_json_data_cache = data_by_file
    return data_by_file

def get_patient_ids_for_selection() -> List[str]:
    """
    Get all unique patient IDs from all JSON files.
    """
    global _patient_ids_cache
    if _patient_ids_cache:
        return _patient_ids_cache
    
    all_data = load_all_json_files()
    patient_ids = set()
    
    for filename, data_list in all_data.items():
        for entry in data_list:
            if isinstance(entry, dict) and "patient_id" in entry:
                patient_ids.add(str(entry["patient_id"]))
    
    try:
        # Sort numerically if possible, otherwise alphabetically
        _patient_ids_cache = sorted(list(patient_ids), key=lambda x: int(x) if x.isdigit() else float('inf'))
    except ValueError:
        _patient_ids_cache = sorted(list(patient_ids))
    
    return _patient_ids_cache

def get_data_for_patient(patient_id: str) -> Dict[str, dict]:
    all_data = load_all_json_files()
    patient_data = {}
    
    for filename, data_list in all_data.items():
        for entry in data_list:
            # Ensure patient_id is compared as a string
            if isinstance(entry, dict) and str(entry.get("patient_id")) == patient_id:
                patient_data[filename] = entry
                break
    
    return patient_data

def parse_filename_to_components(filename: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    try:
        name_without_ext = filename.rsplit('.json', 1)[0]
        
        # 1. Determine Approach based on directory
        approach = None
        if 'singleprompt' in filename.lower():
            approach = "SinglePrompt"
        elif 'multiagent' in filename.lower():
            approach = "MultiAgent"
        
        # 2. Extract prompt version (e.g. "prompt_v1")
        prompt_version = None
        if '_prompt_' in name_without_ext:
            prompt_version = name_without_ext.split('_prompt_')[-1]
            prompt_version = f"prompt_v{prompt_version}" if not prompt_version.startswith("v") else prompt_version
        else:
            # Optional fallback for older modified files
            if '_modified_True' in name_without_ext:
                prompt_version = "modified"
            elif '_modified_False' in name_without_ext:
                prompt_version = "standard"

        # 3. Extract LLM model
        llm_model = None
        prefixes = ['structured_guideline_', 'singleprompt_', 'multiagent_']
        
        for prefix in prefixes:
            if name_without_ext.startswith(prefix):
                start_idx = len(prefix)
                if '_prompt_' in name_without_ext:
                    end_idx = name_without_ext.find('_prompt_', start_idx)
                elif '_modified_' in name_without_ext:
                    end_idx = name_without_ext.find('_modified_', start_idx)
                else:
                    end_idx = len(name_without_ext)
                llm_model = name_without_ext[start_idx:end_idx]
                break

        if not all([llm_model, approach, prompt_version]):
            logger.warning(f"Could not fully parse filename: {filename}. Got: model={llm_model}, approach={approach}, prompt_version={prompt_version}")
        
        return llm_model, approach, prompt_version

    except Exception as e:
        logger.error(f"Error parsing filename {filename}: {e}")
        return None, None, None

def get_available_llm_models_for_patient(patient_id: str) -> List[str]:
    patient_data = get_data_for_patient(patient_id)
    llm_models = set()
    
    for filename in patient_data.keys():
        llm_model, _, _ = parse_filename_to_components(filename)
        if llm_model:
            llm_models.add(llm_model)
    
    return sorted(list(llm_models))

def get_variants_for_patient_and_model(patient_id: str, llm_model: str) -> Dict[str, dict]:
    patient_data = get_data_for_patient(patient_id)
    variants = {}
    
    for filename, entry in patient_data.items():
        parsed_llm, approach, prompt_version = parse_filename_to_components(filename)
        
        if parsed_llm == llm_model and approach and prompt_version:
            variant_key = f"{approach}_{prompt_version}"
            variants[variant_key] = {
                'filename': filename,
                'entry': entry,
                'approach': approach,
                'prompt_version': prompt_version
            }
    
    return variants

def extract_recommendation_from_entry(entry: dict) -> Tuple[str, str, str, str]:
    try:
        raw_output = entry.get("llm_raw_output", "")
        parsed_output = entry.get("llm_parsed_output", {})
        
        # Build the formatted recommendation string from the parsed components
        final_parts = []
        # Use the correct keys ('assessment', 'recommendation', 'rationale') from the JSON
        if parsed_output and parsed_output.get("assessment"):
            final_parts.append(f"<beurteilung>\n{parsed_output['assessment']}\n</beurteilung>")
        
        if parsed_output and parsed_output.get("recommendation"):
            final_parts.append(f"<therapieempfehlung>\n{parsed_output['recommendation']}\n</therapieempfehlung>")
        
        if parsed_output and parsed_output.get("rationale"):
            final_parts.append(f"<begründung>\n{parsed_output['rationale']}\n</begründung>")
        
        # If parsing failed or produced no parts, fall back to the raw output
        formatted_recommendation = "\n\n".join(final_parts) if final_parts else raw_output
        
        # The "think block" is the entire raw output for this structure
        think_block = raw_output
        
        # Format the LLM input for display
        llm_input = json.dumps(entry.get("llm_input", {}), indent=2, ensure_ascii=False)
        
        return formatted_recommendation, think_block, raw_output, llm_input
        
    except Exception as e:
        logger.error(f"Error extracting recommendation from entry: {e}")
        return "Error extracting recommendation", "", "", ""

def save_comparative_evaluation(patient_id: str, llm_model_evaluated: str, evaluation_data: dict, expert_name: str) -> Tuple[bool, Optional[str]]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_patient_id = patient_id.replace('/', '_').replace('\\', '_')
    safe_expert_name = expert_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    safe_llm_model = llm_model_evaluated.replace(':', '_').replace('/', '_').replace('.', '_')

    eval_filename = f"eval_{safe_patient_id}_llm_{safe_llm_model}_{safe_expert_name}_{timestamp}.json"
    filepath = os.path.join(EVALUATION_RESULTS_SAVE_DIR, eval_filename)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Comparative evaluation saved locally to: {filepath}")
        
        if client is not None and db is not None and collection is not None:
            try:
                mongo_document = {
                    "patient_id": patient_id,
                    "llm_model": llm_model_evaluated,
                    "expert_name": expert_name,
                    "timestamp": timestamp,
                    "evaluation_data": evaluation_data,
                    "created_at": datetime.utcnow()
                }
                collection.insert_one(mongo_document)
                logger.info(f"Comparative evaluation also saved to MongoDB (Cloud).")
            except Exception as mongo_e:
                logger.error(f"Failed to save evaluation to MongoDB: {mongo_e}")
        else:
            logger.warning("MongoDB connection not established; skipping cloud save.")

        return True, eval_filename

    except Exception as e:
        logger.error(f"Error saving comparative evaluation: {e}", exc_info=True)
        return False, None

def check_if_evaluated(patient_id: str, llm_model: str, expert_name: str) -> bool:
    """Checks if an evaluation file already exists for this patient-LLM-expert combo."""
    safe_patient_id = patient_id.replace('/', '_').replace('\\', '_')
    safe_expert_name = expert_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    safe_llm_model = llm_model.replace(':', '_').replace('/', '_').replace('.', '_')
    
    pattern = f"eval_{safe_patient_id}_llm_{safe_llm_model}_{safe_expert_name}_*.json"
    search_path = os.path.join(EVALUATION_RESULTS_SAVE_DIR, pattern)
    existing_files = glob.glob(search_path)
    
    return len(existing_files) > 0