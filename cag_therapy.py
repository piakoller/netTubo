import logging
from pathlib import Path
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
from langchain.chat_models import ChatOllama  # or ChatOpenAI, etc.
from langchain.schema.language_model import BaseLanguageModel

from shared_logic import (
    generate_single_recommendation,
    run_processing_pipeline,
    GUIDELINE_SOURCE_DIR,
    BASE_PROJECT_DIR,
)

# --- Setup Langchain Cache (Cache-Augmented Generation) ---
CACHE_PATH = BASE_PROJECT_DIR / ".langchain_cache.db"
set_llm_cache(SQLiteCache(database_path=str(CACHE_PATH)))

# --- Load LLM with deterministic parameters ---
def get_llm(model_name: str = "mistral", temperature: float = 0.0) -> BaseLanguageModel:
    return ChatOllama(
        model=model_name,
        temperature=temperature
    )

# --- Main Pipeline Runner ---
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run therapy recommendation using CAG.")
    parser.add_argument(
        "--model",
        type=str,
        default="mistral",
        help="Model name for LLM backend (e.g., mistral, llama3)"
    )
    parser.add_argument(
        "--patients",
        type=str,
        required=True,
        help="Path to the patient data CSV or JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to store output results JSON"
    )
    parser.add_argument(
        "--clinical_modified",
        action="store_true",
        help="Mark the data as clinically modified (adds flag in filename)"
    )

    args = parser.parse_args()

    # Set logging level
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("cag_runner")
    logger.info(f"Using model: {args.model}")

    # Load model
    llm = get_llm(args.model)

    # Run main processing
    run_processing_pipeline(
        llm=llm,
        llm_model_name=args.model,
        patient_data_file=Path(args.patients),
        output_file=Path(args.output) if args.output else None,
        is_clinical_info_modified=args.clinical_modified
    )


if __name__ == "__main__":
    main()
