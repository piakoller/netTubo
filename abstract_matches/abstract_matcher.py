#!/usr/bin/env python3
"""
Abstract-Patient Matching System

This script reads patient data and scans through a directory of abstracts (in PDF format).
It uses an LLM to identify and extract abstracts that are clinically relevant to each patient
and saves the findings into individual reports.
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

from docling.document_converter import DocumentConverter  # Using docling for PDF processing
from data_loader import load_patient_data
from shared_logic import format_patient_data_for_prompt, PATIENT_FIELDS_FOR_PROMPT
from config import OPENROUTER_API_KEY, OPENROUTER_API_URL, OPENROUTER_MODEL
from netTubo.clinical_trials.clinical_trials_matcher import LLMStudyMatcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AbstractProcessor:
    """Handles PDF processing and abstract extraction."""

    def __init__(self):
        """Initialize the document converter."""
        self.converter = DocumentConverter()

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extracts all text from a given PDF file using docling."""
        try:
            # Use DocumentConverter to process the PDF
            result = self.converter.convert(str(pdf_path))
            return result.document.export_to_text()
        except Exception as e:
            logger.error(f"Could not read PDF {pdf_path} with docling: {e}")
            return ""

    def split_into_abstracts(self, full_text: str, pdf_name: str) -> List[Dict[str, str]]:
        """
        Splits the extracted text from a PDF into individual abstracts.
        This uses a simple heuristic: splitting by "abstract" keyword (case-insensitive).
        """
        # Normalize text by removing excessive newlines
        full_text = re.sub(r'\n\s*\n', '\n', full_text)
        
        # Split by the word "abstract", case-insensitive.
        # The (?i) flag makes it case-insensitive. The ( ) captures the delimiter.
        parts = re.split(r'((?i)abstract)', full_text)
        
        if len(parts) <= 1:
            # If no "abstract" keyword found, treat the whole document as one abstract
            return [{"source": pdf_name, "content": full_text.strip()}]

        abstracts = []
        # The split results in: ['before', 'Abstract', 'after', 'Abstract', 'after'...]
        # We want to combine the delimiter with the text that follows it.
        for i in range(1, len(parts), 2):
            content = parts[i] + parts[i+1]
            abstracts.append({"source": pdf_name, "content": content.strip()})
            
        return abstracts

class AbstractMatcher:
    """Uses an LLM to match patients with relevant abstracts."""

    def __init__(self, llm_matcher: LLMStudyMatcher):
        self.llm_matcher = llm_matcher

    def evaluate_abstract_relevance(self, patient_data: Dict, abstract_content: str) -> Tuple[float, str]:
        """
        Uses an LLM to evaluate if an abstract is clinically relevant to a patient.
        """
        patient_profile = format_patient_data_for_prompt(patient_data, PATIENT_FIELDS_FOR_PROMPT)

        prompt = f"""
You are a clinical expert in neuroendocrine tumors (NETs). Your task is to evaluate if the following research abstract is clinically relevant to the patient profile provided. Focus on direct relevance to the patient's diagnosis, tumor characteristics, and treatment history.

PATIENT PROFILE:
{patient_profile}

ABSTRACT TO EVALUATE:
{abstract_content}

EVALUATION CRITERIA:
1.  **Diagnosis Match:** Is the abstract about the patient's specific type of NET (e.g., pancreatic, small intestine)?
2.  **Treatment Relevance:** Does the abstract discuss a treatment (or diagnostic method) that is relevant to the patient's current situation or potential future options?
3.  **Clinical Applicability:** Is the information presented in the abstract of practical clinical value for this patient's case?

DECISION: [YES/NO] - Is this abstract clinically relevant enough to be brought to a tumor board for this patient?
REASONING: [Provide a brief, clear justification for your decision based on the criteria above.]
"""
        try:
            llm_response = self.llm_matcher.call_llm(prompt)
            
            if "ERROR:" in llm_response:
                logger.error(f"LLM evaluation of abstract failed: {llm_response}")
                return 0.0, "LLM evaluation failed"

            # Simple parsing for YES/NO
            decision_match = re.search(r'DECISION:\s*\[?(YES|NO)\]?', llm_response, re.IGNORECASE)
            score = 1.0 if decision_match and decision_match.group(1).upper() == 'YES' else 0.0
            
            reasoning = ""
            reasoning_match = re.search(r'REASONING:\s*(.*)', llm_response, re.IGNORECASE | re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()

            return score, reasoning

        except Exception as e:
            logger.error(f"An error occurred during abstract evaluation: {e}")
            return 0.0, f"Evaluation error: {e}"

def main():
    """Main function to run the abstract matching process."""
    # --- Configuration ---
    PATIENT_DATA_FILE = Path("C:/Users/pia/OneDrive - Universitaet Bern/Projects/NetTubo/netTubo/data/NET Tubo v2.xlsx")
    ABSTRACTS_DIR = Path("C:/Users/pia/OneDrive - Universitaet Bern/Projects/NetTubo/abstracts")
    OUTPUT_DIR = Path("C:/Users/pia/OneDrive - Universitaet Bern/Projects/NetTubo/netTubo/abstract_matches")
    
    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Initialization ---
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY is not set. Please configure it in your environment or config file.")
        return

    llm_matcher = LLMStudyMatcher(api_key=OPENROUTER_API_KEY, model=OPENROUTER_MODEL)
    abstract_processor = AbstractProcessor()
    abstract_matcher = AbstractMatcher(llm_matcher)

    # --- Load Data ---
    df_patients = load_patient_data(str(PATIENT_DATA_FILE))
    if df_patients is None or df_patients.empty:
        logger.error("Failed to load patient data. Exiting.")
        return
        
    pdf_files = list(ABSTRACTS_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {ABSTRACTS_DIR}. Exiting.")
        return

    logger.info(f"Found {len(df_patients)} patients and {len(pdf_files)} PDF files to process.")

    # --- Main Processing Loop ---
    for _, patient_row in df_patients.iterrows():
        patient_data = patient_row.to_dict()
        patient_id = patient_data.get("ID", "Unknown")
        logger.info(f"Processing patient ID: {patient_id}")

        relevant_abstracts = []

        for pdf_path in pdf_files:
            logger.debug(f"Scanning PDF: {pdf_path.name} for patient {patient_id}")
            full_text = abstract_processor.extract_text_from_pdf(pdf_path)
            if not full_text:
                continue

            potential_abstracts = abstract_processor.split_into_abstracts(full_text, pdf_path.name)
            
            for abstract in potential_abstracts:
                score, reason = abstract_matcher.evaluate_abstract_relevance(patient_data, abstract['content'])
                
                if score >= 1.0:
                    logger.info(f"Found RELEVANT abstract for patient {patient_id} in {abstract['source']}")
                    relevant_abstracts.append({
                        "source_file": abstract['source'],
                        "relevance_reason": reason,
                        "abstract_content": abstract['content']
                    })

        # --- Save Results for the Patient ---
        if relevant_abstracts:
            output_file = OUTPUT_DIR / f"patient_{patient_id}_relevant_abstracts.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Relevant Abstracts for Patient ID: {patient_id}\n")
                f.write(f"Patient Diagnosis: {patient_data.get('main_diagnosis_text', 'N/A')}\n")
                f.write("="*80 + "\n\n")
                
                for i, item in enumerate(relevant_abstracts, 1):
                    f.write(f"--- Abstract {i} ---\n")
                    f.write(f"Source PDF: {item['source_file']}\n")
                    f.write(f"Relevance Reasoning: {item['relevance_reason']}\n\n")
                    f.write(f"{item['abstract_content']}\n\n")
                    f.write("-" * 80 + "\n\n")
            
            logger.info(f"Saved {len(relevant_abstracts)} relevant abstracts for patient {patient_id} to {output_file}")
        else:
            logger.info(f"No relevant abstracts found for patient {patient_id}")

    logger.info("Abstract matching process completed.")

if __name__ == "__main__":
    # Add project root to path to allow imports
    sys.path.append(str(Path(__file__).parent.parent))
    main()
