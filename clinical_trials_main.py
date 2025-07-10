#!/usr/bin/env python3
"""
Main script for the Clinical Trials Matching System

This script orchestrates the matching of patients with relevant clinical trials
using the modular clinical_trials package.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from clinical_trials import (
    ClinicalTrialsAPI, 
    LLMStudyMatcher, 
    PublicationDownloader,
    PatientStudyMatcher,
    generate_study_report,
    generate_summary_report,
    test_api_connection
)
from data_loader import load_patient_data
from config import OPENROUTER_API_KEY, OPENROUTER_API_URL, OPENROUTER_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if OpenRouter API key is available
if not OPENROUTER_API_KEY:
    logger.warning("OPENROUTER_API_KEY not set. Please set it as an environment variable.")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")


def main():
    """Main function to run the clinical trials matching system."""
    
    # Check if OpenRouter API key is available
    if not OPENROUTER_API_KEY:
        print("❌ OPENROUTER_API_KEY not found in environment variables.")
        print("Please set your OpenRouter API key in the .env file or environment variables.")
        return
    
    # Test API connection first
    if not test_api_connection():
        print("Exiting due to API connection failure.")
        return
    
    # Configuration
    PATIENT_DATA_FILE = Path("C:/Users/pia/OneDrive - Universitaet Bern/Projects/NetTubo/netTubo/data/NET Tubo v2.xlsx")
    OUTPUT_DIR = Path("C:/Users/pia/OneDrive - Universitaet Bern/Projects/NetTubo/netTubo/clinical_trials_matches")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Initialize components
    api = ClinicalTrialsAPI(rate_limit_delay=1.0)
    llm_matcher = LLMStudyMatcher(OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_API_URL)
    pub_downloader = PublicationDownloader(OUTPUT_DIR)
    matcher = PatientStudyMatcher(api, llm_matcher, pub_downloader)
    
    # Load patient data
    try:
        df_patients = load_patient_data(str(PATIENT_DATA_FILE))
        if df_patients is None or df_patients.empty:
            logger.error("No patient data loaded")
            return
    except Exception as e:
        logger.error(f"Error loading patient data: {e}")
        return
    
    # Process each patient
    all_results = []
    patient_ids = [pid for pid in df_patients["ID"].unique() if pid and str(pid).strip()]
    
    logger.info(f"Processing {len(patient_ids)} patients with LLM-based matching")
    
    for i, patient_id in enumerate(patient_ids, 1):
        logger.info(f"Processing patient {i}/{len(patient_ids)} (ID: {patient_id})")
        
        try:
            # Get patient data
            patient_row = df_patients[df_patients["ID"] == patient_id].iloc[0]
            patient_data = patient_row.to_dict()
            
            # Find relevant studies using LLM evaluation
            matches = matcher.find_relevant_studies(
                patient_data, 
                min_relevance_score=0.3,  # Higher threshold since LLM provides better scoring
                max_studies=15  # Fewer studies but higher quality
            )
            
            # Generate report
            report = generate_study_report(patient_data, matches)
            
            # Save individual patient report
            patient_report_file = OUTPUT_DIR / f"patient_{patient_id}_clinical_trials_llm.txt"
            with open(patient_report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"Found {len(matches)} relevant studies for patient {patient_id}")
            
            # Store results for summary
            patient_result = {
                "patient_id": str(patient_id),
                "patient_data": patient_data,
                "matches_found": len(matches),
                "llm_evaluation_method": OPENROUTER_MODEL,
                "min_relevance_score": 0.3,
                "matches": [
                    {
                        "nct_id": match.nct_id,
                        "title": match.title,
                        "status": match.status,
                        "phase": match.phase,
                        "relevance_score": match.relevance_score,
                        "relevance_reason": match.relevance_reason,
                        "condition": match.condition,
                        "intervention": match.intervention,
                        "url": match.url,
                        "publications": match.publications if match.publications else None
                    } for match in matches
                ],
                "report_file": str(patient_report_file)
            }
            
            all_results.append(patient_result)
            
            # Add delay between patients to avoid overwhelming APIs
            if i < len(patient_ids):
                time.sleep(2.0)
            
        except Exception as e:
            logger.error(f"Error processing patient {patient_id}: {e}")
            continue
    
    # Save summary results
    summary_file = OUTPUT_DIR / "clinical_trials_summary_llm.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Generate and print overall summary
    summary_report = generate_summary_report(all_results, OUTPUT_DIR, summary_file, OPENROUTER_MODEL)
    
    print(summary_report)
    logger.info("LLM-based clinical trials matching completed successfully")


if __name__ == "__main__":
    main()
