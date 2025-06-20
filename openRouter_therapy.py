import argparse
import logging
import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, List, Optional, Dict
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation

import config
from shared_logic import LLM_TEMPERATURE, run_processing_pipeline

load_dotenv()

DEFAULT_OPENROUTER_MODEL = "google/gemma-2-9b-it:free"
API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# --- Configuration for this specific script ---
logger = logging.getLogger("run_openrouter")


class OpenRouterLLM(BaseLanguageModel):
    """LangChain-compatible wrapper for calling OpenRouter API."""

    def __init__(self, model_name: str, temperature: float, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.model = model_name
        self.temperature = temperature
        self.api_key = api_key

    @property
    def _llm_type(self) -> str:
        """Return identifier of llm type."""
        return "openrouter"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate responses for the given prompts."""
        generations = []
        
        for prompt in prompts:
            try:
                response_text = self._call_api(prompt)
                generations.append([Generation(text=response_text)])
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                generations.append([Generation(text=f"Error: {str(e)}")])
        
        return LLMResult(generations=generations)

    def _call_api(self, prompt: str) -> str:
        """Make the actual API call to OpenRouter."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://yourproject.local",
            "X-Title": "NetTubo Therapy"
        }

        data = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": prompt}
            ]
        }

        try:
            response = requests.post(OPENROUTER_URL, headers=headers, data=json.dumps(data))
            
            # Debug logging
            logger.debug(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                try:
                    error_data = response.json()
                    logger.error(f"Error details: {json.dumps(error_data, indent=2)}")
                except:
                    pass
                raise Exception(f"API call failed with status {response.status_code}: {response.text}")
            
            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise Exception(f"Request failed: {e}")

    def invoke(self, prompt: str) -> str:
        """Invoke method for direct string prompts (backward compatibility)."""
        return self._call_api(prompt)

    # Required for BaseLanguageModel but not used in our case
    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Async generate - not implemented for this simple wrapper."""
        raise NotImplementedError("Async generation not implemented")


def main():
    parser = argparse.ArgumentParser(description="Generate recommendations using the OpenRouter API.")
    parser.add_argument(
        "--llm_model", type=str, default=DEFAULT_OPENROUTER_MODEL,
        help=f"OpenRouter model to use. Default: {DEFAULT_OPENROUTER_MODEL}"
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

    if not API_KEY:
        logger.error("OPENROUTER_API_KEY environment variable not set. Please set it to your API key.")
        return

    # Validate API key format
    if not API_KEY.startswith("sk-or-v1-"):
        logger.error("Invalid API key format. OpenRouter API keys should start with 'sk-or-v1-'")
        return

    is_modified = args.clinical_info_modified
    if args.patient_data_file:
        patient_file = args.patient_data_file
        if not is_modified:
            logger.info("Using a custom patient data file, so 'clinical_info_modified' is automatically set to True.")
            is_modified = True
    else:
        patient_file = Path(config.TUBO_EXCEL_FILE_PATH)

    try:
        logger.info(f"Initializing OpenRouter model: {args.llm_model}")
        llm = OpenRouterLLM(
            model_name=args.llm_model,
            temperature=LLM_TEMPERATURE,
            api_key=API_KEY
        )
        # Test prompt
        logger.info("Testing connection with simple prompt...")
        test_response = llm.invoke("Hi!")
        logger.info(f"Test prompt response: {test_response[:60]}...")
    except Exception as e:
        logger.error(f"Failed to connect to OpenRouter model '{args.llm_model}'.", exc_info=True)
        return

    run_processing_pipeline(
        llm=llm,
        llm_model_name=args.llm_model,
        patient_data_file=patient_file,
        output_file=args.output_file,
        is_clinical_info_modified=is_modified
    )


if __name__ == "__main__":
    main()