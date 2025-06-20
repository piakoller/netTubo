# run_ollama.py

import argparse
import logging
from pathlib import Path

from langchain_ollama import OllamaLLM

import config
from shared_logic import LLM_TEMPERATURE, run_processing_pipeline

# --- Configuration for this specific script ---
DEFAULT_OLLAMA_MODEL = "hf.co/unsloth/medgemma-27b-text-it-GGUF:Q4_K_M"
logger = logging.getLogger("run_ollama")


def main():
    """Main execution function for the Ollama approach."""
    parser = argparse.ArgumentParser(description="Generate recommendations using a local Ollama model.")
    parser.add_argument(
        "--llm_model", type=str, default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama model to use. Default: {DEFAULT_OLLAMA_MODEL}"
    )
    parser.add_argument(
        "--patient_data_file", type=Path, default=None,
        help="Optional: Path to the patient data Excel file. Overrides the default from config."
    )
    parser.add_argument(
        "--output_file", type=Path, default=None,
        help="Path to save the JSON results. If not set, a filename is auto-generated."
    )
    parser.add_argument(
        "--clinical_info_modified", action="store_true",
        help="Flag that context was modified. Set automatically if --patient_data_file is used."
    )
    args = parser.parse_args()

    # --- Determine final file paths and flags ---
    is_modified = args.clinical_info_modified
    if args.patient_data_file:
        patient_file = args.patient_data_file
        if not is_modified:
            logger.info("Using a custom patient data file, so 'clinical_info_modified' is automatically set to True.")
            is_modified = True
    else:
        patient_file = Path(config.TUBO_EXCEL_FILE_PATH)

    # --- Initialize the LLM ---
    try:
        logger.info(f"Initializing Ollama model: {args.llm_model}")
        llm = OllamaLLM(
            model=args.llm_model,
            temperature=LLM_TEMPERATURE,
            model_kwargs={
                "num_ctx": 131072
            }
        )
        llm.invoke("Hi")
    except Exception as e:
        logger.error(f"Failed to initialize or connect to Ollama model '{args.llm_model}'.", exc_info=True)
        logger.error("Please ensure Ollama is running and the model is downloaded (e.g., 'ollama pull {args.llm_model}').")
        return

    # --- Run the shared processing pipeline ---
    run_processing_pipeline(
        llm=llm,
        llm_model_name=args.llm_model,
        patient_data_file=patient_file,
        output_file=args.output_file,
        is_clinical_info_modified=is_modified
    )


if __name__ == "__main__":
    main()