#!/usr/bin/env python3
"""
ClinicalTrials.gov API Patient-Study Matching System

This script queries the ClinicalTrials.gov API to find relevant clinical trials
for patients based on their diagnosis and clinical characteristics.
It can find both active and completed studies.
"""

import json
import logging
import requests
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import sys

# Add the current directory to Python path to import shared_logic
sys.path.append(str(Path(__file__).parent))

from shared_logic import format_patient_data_for_prompt, PATIENT_FIELDS_FOR_PROMPT
from data_loader import load_patient_data
from config import OPENROUTER_API_KEY, OPENROUTER_API_URL, OPENROUTER_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if OpenRouter API key is available
if not OPENROUTER_API_KEY:
    logger.warning("OPENROUTER_API_KEY not set. Please set it as an environment variable.")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

MIN_RELEVANCE_SCORE = 0.6

class LLMStudyMatcher:
    """Uses an LLM to match patients with clinical studies."""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.model = model or OPENROUTER_MODEL
        self.base_url = OPENROUTER_API_URL
        
    def call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Call the LLM API with error handling and retries."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/piakoller/netTubo",
            "X-Title": "NET Clinical Trials Matcher",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.0
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    response_json = response.json()
                    if "choices" in response_json and len(response_json["choices"]) > 0:
                        return response_json["choices"][0]["message"]["content"]
                else:
                    logger.warning(f"API returned status {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"LLM API request failed (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return "ERROR: LLM API call failed after all retries"
    
    def evaluate_study_relevance(self, patient_data: Dict, study_info: Dict) -> Tuple[float, str]:
        """
        Use LLM to evaluate if a study is relevant for a patient.
        
        Returns:
            Tuple of (relevance_score, explanation)
        """
        # Format patient data using only the fields specified in PATIENT_FIELDS_FOR_PROMPT
        patient_data_formatted = format_patient_data_for_prompt(patient_data, PATIENT_FIELDS_FOR_PROMPT)
        
        # Extract study information
        protocol = study_info.get("protocolSection", {})
        identification = protocol.get("identificationModule", {})
        conditions = protocol.get("conditionsModule", {}).get("conditions", [])
        interventions = protocol.get("armsInterventionsModule", {}).get("interventions", [])
        eligibility = protocol.get("eligibilityModule", {})
        
        study_title = identification.get("briefTitle", "")
        # Get detailed description from descriptionModule for comprehensive evaluation
        description_module = protocol.get("descriptionModule", {})
        study_summary = description_module.get("briefSummary", "")
        detailed_description = description_module.get("detailedDescription", "")
        
        # For LLM evaluation, we use the detailed description for comprehensive comparison
        # Create prompt for LLM evaluation using detailed description
        prompt = f"""
Du bist Facharzt für Onkologie, Facharzt für Endokrinologie und Facharzt für Nuklearmedizin. Bewerte, ob die folgende klinische Studie für den Patienten relevant sein könnte.

{patient_data_formatted}

STUDIE INFORMATION:
- Titel: {study_title}
- Beschreibung: {detailed_description}

WICHTIGER HINWEIS:
Der Patient hat eine bestimmte Diagnose, aber die konkrete Behandlungsmethode ist noch NICHT festgelegt. Die Behandlung soll erst in einem zweiten Schritt basierend auf den gefundenen relevanten Studien entschieden werden.

AUFGABE:
Bewerte die Relevanz dieser Studie für den Patienten basierend auf der detaillierten Beschreibung auf einer Skala von 0.0 bis 1.0, wobei:
- 0.0 = Völlig irrelevant (andere Erkrankung, völlig unpassend)
- 0.3 = Teilweise relevant (ähnliche Erkrankung oder verwandte Indikation)
- 0.6 = Relevant (passende Erkrankung, Studie könnte für Behandlungsentscheidung hilfreich sein)
- 0.9 = Hochrelevant (sehr gut passende Erkrankung und Patientenprofil)
- 1.0 = Ideal geeignet (perfekte Übereinstimmung mit Diagnose und Patientencharakteristika)

Berücksichtige HAUPTSÄCHLICH:
1. Übereinstimmung der Diagnose/Erkrankung mit der Studienindikation aus der detaillierten Beschreibung
2. Relevanz der Studie für die Behandlungsentscheidung basierend auf der detaillierten Beschreibung
3. Passung der beschriebenen Patientenpopulation zum gegebenen Patienten
4. Spezifische Einschlusskriterien und Studiendesign aus der detaillierten Beschreibung

Antworte im folgenden Format:
RELEVANZ_SCORE: [0.0-1.0]
BEGRÜNDUNG: [Detaillierte Erklärung der Relevanz basierend auf Diagnose-Matching mit der detaillierten Beschreibung]
"""

        try:
            llm_response = self.call_llm(prompt)
            
            if "ERROR:" in llm_response:
                logger.error(f"LLM evaluation failed: {llm_response}")
                return 0.0, "LLM evaluation failed"
            
            # Parse LLM response
            relevance_score = 0.0
            explanation = "No explanation provided"
            
            lines = llm_response.split('\n')
            for line in lines:
                if line.startswith("RELEVANZ_SCORE:"):
                    try:
                        score_text = line.split(":", 1)[1].strip()
                        relevance_score = float(score_text)
                        relevance_score = max(0.0, min(1.0, relevance_score))  # Clamp to 0-1
                    except (ValueError, IndexError):
                        logger.warning(f"Could not parse relevance score from: {line}")
                        
                elif line.startswith("BEGRÜNDUNG:"):
                    explanation = line.split(":", 1)[1].strip()
            
            # If no structured response, try to extract score from text
            if relevance_score == 0.0 and explanation == "No explanation provided":
                import re
                score_match = re.search(r'(\d+\.?\d*)', llm_response)
                if score_match:
                    try:
                        relevance_score = float(score_match.group(1))
                        if relevance_score > 1.0:
                            relevance_score = relevance_score / 10.0  # Convert 8.5 to 0.85
                        relevance_score = max(0.0, min(1.0, relevance_score))
                    except ValueError:
                        pass
                explanation = llm_response[:200] + "..." if len(llm_response) > 200 else llm_response
            
            return relevance_score, explanation
            
        except Exception as e:
            logger.error(f"Error in LLM evaluation: {e}")
            return 0.0, f"Evaluation error: {str(e)}"

@dataclass
class ClinicalTrialMatch:
    """Represents a clinical trial match for a patient."""
    nct_id: str
    title: str
    status: str
    phase: str
    condition: str
    intervention: str
    brief_summary: str
    detailed_description: str
    eligibility_criteria: str
    start_date: str
    completion_date: str
    primary_outcome: str
    secondary_outcome: str
    sponsor: str
    relevance_score: float
    relevance_reason: str
    locations: List[str]
    url: str
    publications: Dict = None  # Will contain publication information

class ClinicalTrialsAPI:
    """Interface to ClinicalTrials.gov API v2."""
    
    BASE_URL = "https://clinicaltrials.gov/api/v2"
    
    def __init__(self, rate_limit_delay: float = 1.0):
        """Initialize with rate limiting."""
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Implement rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def search_studies(self, 
                      condition: str, 
                      intervention: Optional[str] = None,
                      status: Optional[List[str]] = None,
                      phase: Optional[List[str]] = None,
                      max_results: int = 100) -> List[Dict]:
        """
        Search for clinical trials based on condition and other criteria.
        
        Args:
            condition: Medical condition (e.g., "neuroendocrine tumor", "NET")
            intervention: Intervention/treatment (e.g., "PRRT", "everolimus")
            status: Study status (e.g., ["RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED"])
            phase: Study phases (e.g., ["PHASE1", "PHASE2", "PHASE3"])
            max_results: Maximum number of results to return
            
        Returns:
            List of study dictionaries
        """
        self._rate_limit()
        
        # Build query string manually for better control
        query_parts = [f"AREA[Condition]{condition}"]
        
        if intervention:
            query_parts.append(f"AREA[InterventionName]{intervention}")
            
        if status:
            # Convert status to proper format
            status_mapping = {
                "RECRUITING": "Recruiting",
                "ACTIVE_NOT_RECRUITING": "Active, not recruiting", 
                "COMPLETED": "Completed",
                "TERMINATED": "Terminated"
            }
            mapped_statuses = [status_mapping.get(s, s) for s in status]
            status_query = " OR ".join([f'AREA[OverallStatus]"{s}"' for s in mapped_statuses])
            query_parts.append(f"({status_query})")
        
        query_string = " AND ".join(query_parts)
        
        # Build parameters with simplified approach
        params = {
            "format": "json",
            "pageSize": min(max_results, 1000),
            "query.term": query_string
        }
        
        try:
            logger.info(f"Searching with query: {query_string}")
            response = requests.get(f"{self.BASE_URL}/studies", params=params, timeout=30)
            
            # Log the actual URL for debugging
            logger.debug(f"Request URL: {response.url}")
            
            if response.status_code == 400:
                logger.warning(f"Bad request for condition '{condition}'. Trying simpler query...")
                # Fallback to simpler query
                simple_params = {
                    "format": "json",
                    "pageSize": min(max_results, 1000),
                    "query.term": condition
                }
                response = requests.get(f"{self.BASE_URL}/studies", params=simple_params, timeout=30)
            
            response.raise_for_status()
            
            data = response.json()
            studies = data.get("studies", [])
            
            logger.info(f"Found {len(studies)} studies for condition: {condition}")
            return studies
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching studies: {e}")
            # Try one more fallback with basic search
            try:
                logger.info(f"Trying basic search for: {condition}")
                basic_params = {
                    "format": "json", 
                    "pageSize": min(max_results, 100),
                    "query.term": f'"{condition}"'
                }
                response = requests.get(f"{self.BASE_URL}/studies", params=basic_params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    studies = data.get("studies", [])
                    logger.info(f"Basic search found {len(studies)} studies")
                    return studies
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
            
            return []
    
    def get_study_details(self, nct_id: str) -> Optional[Dict]:
        """Get detailed information for a specific study."""
        self._rate_limit()
        
        try:
            response = requests.get(f"{self.BASE_URL}/studies/{nct_id}", 
                                  params={"format": "json"}, 
                                  timeout=30)
            response.raise_for_status()
            
            data = response.json()
            study = data.get("protocolSection", {})
            
            return {"protocolSection": study}
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting study details for {nct_id}: {e}")
            return None

class PatientStudyMatcher:
    """Matches patients with relevant clinical trials using LLM evaluation."""
    
    def __init__(self, api: ClinicalTrialsAPI, llm_matcher: LLMStudyMatcher, pub_downloader = None):
        self.api = api
        self.llm_matcher = llm_matcher
        self.pub_downloader = pub_downloader
        
    def find_relevant_studies(self, 
                            patient_data: Dict, 
                            min_relevance_score: float = MIN_RELEVANCE_SCORE,
                            max_studies: int = 50) -> List[ClinicalTrialMatch]:
        """
        Find relevant clinical trials for a patient using LLM evaluation.
        
        Args:
            patient_data: Patient information dictionary
            min_relevance_score: Minimum relevance score to include study
            max_studies: Maximum number of studies to return
            
        Returns:
            List of ClinicalTrialMatch objects sorted by relevance
        """
        matches = []
        
        # Extract search terms from patient data - simplified set for broader search
        search_terms = [
            "neuroendocrine tumor",
            "NET"
        ]
        
        # Search for studies with different terms
        all_studies = []
        for term in search_terms:
            logger.info(f"Searching for studies with term: {term}")
            studies = self.api.search_studies(
                condition=term,
                status=["RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED", "TERMINATED"],
                max_results=30  # Get more studies to evaluate with LLM
            )
            all_studies.extend(studies)
            # Add small delay between different search terms
            time.sleep(0.5)
        
        # Remove duplicates based on NCT ID
        unique_studies = {}
        for study in all_studies:
            nct_id = study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
            if nct_id and nct_id not in unique_studies:
                unique_studies[nct_id] = study
        
        logger.info(f"Found {len(unique_studies)} unique studies to evaluate")
        
        # Use LLM to evaluate each study
        evaluated_count = 0
        for nct_id, study in unique_studies.items():
            try:
                evaluated_count += 1
                logger.info(f"Evaluating study {evaluated_count}/{len(unique_studies)}: {nct_id}")
                
                # Use LLM to calculate relevance
                score, reason = self.llm_matcher.evaluate_study_relevance(patient_data, study)
                
                logger.info(f"Study {nct_id} - Relevance Score: {score:.3f}")
                
                if score >= min_relevance_score:
                    # Extract study details
                    protocol = study.get("protocolSection", {})
                    identification = protocol.get("identificationModule", {})
                    status_module = protocol.get("statusModule", {})
                    design_module = protocol.get("designModule", {})
                    conditions_module = protocol.get("conditionsModule", {})
                    interventions_module = protocol.get("armsInterventionsModule", {})
                    eligibility_module = protocol.get("eligibilityModule", {})
                    outcomes_module = protocol.get("outcomesModule", {})
                    sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
                    contacts_module = protocol.get("contactsLocationsModule", {})
                    description_module = protocol.get("descriptionModule", {})
                    
                    # Extract locations
                    locations = []
                    location_facilities = contacts_module.get("locations", [])
                    for loc in location_facilities:
                        facility = loc.get("facility", "")
                        city = loc.get("city", "")
                        country = loc.get("country", "")
                        if facility:
                            locations.append(f"{facility}, {city}, {country}")
                    
                    # Download publication information if downloader is available
                    publications_info = None
                    # if self.pub_downloader:
                    #     logger.info(f"Downloading publications for study {nct_id}")
                    #     publications_info = self.pub_downloader.download_publication_info(
                    #         nct_id, identification.get("briefTitle", "")
                    #     )
                    
                    match = ClinicalTrialMatch(
                        nct_id=nct_id,
                        title=identification.get("briefTitle", ""),
                        status=status_module.get("overallStatus", ""),
                        phase=design_module.get("phases", ["N/A"])[0] if design_module.get("phases") else "N/A",
                        condition="; ".join(conditions_module.get("conditions", [])),
                        intervention="; ".join([
                            interv.get("name", "") for interv in interventions_module.get("interventions", [])
                        ]),
                        brief_summary=description_module.get("briefSummary", ""),
                        detailed_description=description_module.get("detailedDescription", ""),
                        eligibility_criteria=eligibility_module.get("eligibilityCriteria", ""),
                        start_date=status_module.get("startDateStruct", {}).get("date", ""),
                        completion_date=status_module.get("completionDateStruct", {}).get("date", ""),
                        primary_outcome="; ".join([
                            outcome.get("measure", "") for outcome in outcomes_module.get("primaryOutcomes", [])
                        ]),
                        secondary_outcome="; ".join([
                            outcome.get("measure", "") for outcome in outcomes_module.get("secondaryOutcomes", [])
                        ]),
                        sponsor=sponsor_module.get("leadSponsor", {}).get("name", ""),
                        relevance_score=score,
                        relevance_reason=reason,
                        locations=locations,
                        url=f"https://clinicaltrials.gov/study/{nct_id}",
                        publications=publications_info
                    )
                    
                    matches.append(match)
                    logger.info(f"Added relevant study: {match.title} (Score: {score:.3f})")
                    
                # Add delay between LLM evaluations to avoid rate limiting
                time.sleep(1.0)
                    
            except Exception as e:
                logger.error(f"Error processing study {nct_id}: {e}")
                continue
        
        # Sort by relevance score (highest first)
        matches.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"Found {len(matches)} relevant studies (score >= {min_relevance_score})")
        
        return matches[:max_studies]

def generate_study_report(patient_data: Dict, matches: List[ClinicalTrialMatch]) -> str:
    """Generate a formatted report of relevant studies for a patient."""
    
    report = []
    report.append("="*80)
    report.append("CLINICAL TRIALS MATCHING REPORT")
    report.append("="*80)
    
    # Patient information
    report.append(f"\nPatient ID: {patient_data.get('ID', 'Unknown')}")
    report.append(f"Diagnosis: {patient_data.get('main_diagnosis_text', 'Not specified')}")
    report.append(f"Clinical Question: {patient_data.get('Fragestellung', 'Not specified')}")
    report.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    report.append(f"\nFound {len(matches)} relevant clinical trials")
    report.append("-"*80)
    
    for i, match in enumerate(matches, 1):
        report.append(f"\n{i}. {match.title}")
        report.append(f"   NCT ID: {match.nct_id}")
        report.append(f"   Status: {match.status}")
        report.append(f"   Phase: {match.phase}")
        report.append(f"   Relevance Score: {match.relevance_score:.3f}")
        report.append(f"   Relevance Reason: {match.relevance_reason}")
        report.append(f"   Condition: {match.condition}")
        report.append(f"   Intervention: {match.intervention}")
        report.append(f"   Sponsor: {match.sponsor}")
        report.append(f"   Start Date: {match.start_date}")
        report.append(f"   Completion Date: {match.completion_date}")
        if match.locations:
            report.append(f"   Locations: {'; '.join(match.locations[:3])}")  # Show first 3 locations
        report.append(f"   URL: {match.url}")
        
        # Add brief summary if available
        if match.brief_summary:
            summary = match.brief_summary[:300] + "..." if len(match.brief_summary) > 300 else match.brief_summary
            report.append(f"   Summary: {summary}")
        
        # Add publication information if available
        # if match.publications and match.publications.get('publications_found', 0) > 0:
        #     report.append(f"   Publications Found: {match.publications['publications_found']}")
        #     for i, pub in enumerate(match.publications['publications'][:2], 1):  # Show first 2 publications
        #         report.append(f"     {i}. {pub['title']} ({pub['year']})")
        #         report.append(f"        PMID: {pub['pmid']} | Journal: {pub['journal']}")
        # elif match.publications:
        #     report.append(f"   Publications: No publications found")
        
        report.append("-"*40)
    
    return "\n".join(report)

def test_api_connection():
    """Test if the ClinicalTrials.gov API is working."""
    api = ClinicalTrialsAPI(rate_limit_delay=1.0)
    
    print("Testing ClinicalTrials.gov API connection...")
    
    # Test with a simple, common condition
    test_studies = api.search_studies("cancer", max_results=5)
    
    if test_studies:
        print(f"✅ API connection successful! Found {len(test_studies)} studies for 'cancer'")
        
        # Show first study as example
        if test_studies:
            first_study = test_studies[0]
            protocol = first_study.get("protocolSection", {})
            identification = protocol.get("identificationModule", {})
            print(f"Example study: {identification.get('briefTitle', 'No title')}")
            print(f"NCT ID: {identification.get('nctId', 'No ID')}")
        return True
    else:
        print("❌ API connection failed!")
        return False

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
    
    # Initialize API, LLM matcher, publication downloader, and patient matcher
    api = ClinicalTrialsAPI(rate_limit_delay=1.0)
    llm_matcher = LLMStudyMatcher(OPENROUTER_API_KEY)
    # llm_matcher = LLMStudyMatcher(OPENROUTER_API_KEY, model="google/gemini-2.5-pro-exp-03-25")
    # pub_downloader = PublicationDownloader(OUTPUT_DIR)  # Commented out
    pub_downloader = None  # Disabled publication downloading
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
                min_relevance_score=MIN_RELEVANCE_SCORE,  # Higher threshold since LLM provides better scoring
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
                "min_relevance_score": MIN_RELEVANCE_SCORE,
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
    
    # Generate overall summary
    total_matches = sum(result["matches_found"] for result in all_results)
    avg_matches = total_matches / len(all_results) if all_results else 0
    
    # Calculate score distribution and publication statistics
    all_scores = []
    # total_publications = 0  # Commented out since publications are disabled
    # studies_with_publications = 0  # Commented out since publications are disabled
    
    for result in all_results:
        for match in result["matches"]:
            all_scores.append(match["relevance_score"])
            # if match.get("publications") and match["publications"].get("publications_found", 0) > 0:
            #     total_publications += match["publications"]["publications_found"]
            #     studies_with_publications += 1
    
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    high_relevance_count = len([s for s in all_scores if s >= 0.7])
    
    summary_report = [
        "="*80,
        "LLM-BASED CLINICAL TRIALS MATCHING SUMMARY",
        "="*80,
        f"Total patients processed: {len(all_results)}",
        f"Total studies found: {total_matches}",
        f"Average studies per patient: {avg_matches:.1f}",
        f"Average relevance score: {avg_score:.3f}",
        f"High relevance studies (≥0.7): {high_relevance_count}",
        # f"Studies with publications: {studies_with_publications}/{total_matches}",  # Commented out - publications disabled
        # f"Total publications downloaded: {total_publications}",  # Commented out - publications disabled
        # f"Publications directory: {OUTPUT_DIR}/publications",  # Commented out - publications disabled
        f"Evaluation method: LLM-based semantic matching",
        f"Results saved to: {OUTPUT_DIR}",
        f"Summary data: {summary_file}",
        "="*80,
        "Note: Studies were evaluated using AI to assess relevance",
        "based on patient diagnosis and clinical context.",
        "Publication downloading is currently disabled."
    ]
    
    print("\n".join(summary_report))
    logger.info("LLM-based clinical trials matching completed successfully")

class PublicationDownloader:
    """Downloads publications related to clinical trials."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.publications_dir = output_dir / "publications"
        self.publications_dir.mkdir(exist_ok=True)
        
    def search_pubmed_for_study(self, nct_id: str, study_title: str) -> List[Dict]:
        """Search PubMed for publications related to a clinical trial."""
        try:
            # PubMed E-utilities API
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            
            # Search terms: NCT ID and parts of the study title
            search_terms = [nct_id]
            
            # Add key terms from study title
            title_words = study_title.lower().split()
            # Filter out common words and keep meaningful terms
            meaningful_words = [word for word in title_words if len(word) > 3 and 
                              word not in ['study', 'trial', 'clinical', 'phase', 'randomized', 'controlled']]
            search_terms.extend(meaningful_words[:3])  # Take first 3 meaningful words
            
            search_query = " AND ".join(search_terms)
            
            params = {
                'db': 'pubmed',
                'term': search_query,
                'retmode': 'json',
                'retmax': 5,  # Limit to 5 most relevant results
                'sort': 'relevance'
            }
            
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            pmids = data.get('esearchresult', {}).get('idlist', [])
            
            if not pmids:
                logger.info(f"No PubMed articles found for study {nct_id}")
                return []
            
            # Get details for found articles
            return self._get_pubmed_details(pmids)
            
        except Exception as e:
            logger.error(f"Error searching PubMed for study {nct_id}: {e}")
            return []
    
    def _get_pubmed_details(self, pmids: List[str]) -> List[Dict]:
        """Get detailed information for PubMed articles."""
        try:
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            
            params = {
                'db': 'pubmed',
                'id': ','.join(pmids),
                'retmode': 'xml',
                'rettype': 'abstract'
            }
            
            response = requests.get(base_url, params=params, timeout=15)
            response.raise_for_status()
            
            # Parse XML response (basic parsing)
            articles = []
            xml_content = response.text
            
            # Extract basic information using regex (simplified approach)
            import re
            
            # Find all PubmedArticle blocks
            article_blocks = re.findall(r'<PubmedArticle>(.*?)</PubmedArticle>', xml_content, re.DOTALL)
            
            for block in article_blocks:
                try:
                    # Extract PMID
                    pmid_match = re.search(r'<PMID[^>]*>(\d+)</PMID>', block)
                    pmid = pmid_match.group(1) if pmid_match else "unknown"
                    
                    # Extract title
                    title_match = re.search(r'<ArticleTitle>(.*?)</ArticleTitle>', block, re.DOTALL)
                    title = title_match.group(1) if title_match else "No title"
                    title = re.sub(r'<[^>]+>', '', title).strip()  # Remove HTML tags
                    
                    # Extract authors
                    authors = []
                    author_blocks = re.findall(r'<Author[^>]*>(.*?)</Author>', block, re.DOTALL)
                    for author_block in author_blocks[:3]:  # First 3 authors
                        lastname_match = re.search(r'<LastName>(.*?)</LastName>', author_block)
                        firstname_match = re.search(r'<ForeName>(.*?)</ForeName>', author_block)
                        if lastname_match:
                            lastname = lastname_match.group(1)
                            firstname = firstname_match.group(1) if firstname_match else ""
                            authors.append(f"{firstname} {lastname}".strip())
                    
                    # Extract journal
                    journal_match = re.search(r'<Title>(.*?)</Title>', block)
                    journal = journal_match.group(1) if journal_match else "Unknown journal"
                    
                    # Extract publication year
                    year_match = re.search(r'<Year>(\d{4})</Year>', block)
                    year = year_match.group(1) if year_match else "Unknown year"
                    
                    # Extract DOI
                    doi_match = re.search(r'<ELocationID EIdType="doi"[^>]*>(.*?)</ELocationID>', block)
                    doi = doi_match.group(1) if doi_match else None
                    
                    # Extract abstract
                    abstract_match = re.search(r'<Abstract>(.*?)</Abstract>', block, re.DOTALL)
                    abstract = ""
                    if abstract_match:
                        abstract_text = abstract_match.group(1)
                        abstract = re.sub(r'<[^>]+>', '', abstract_text).strip()
                    
                    articles.append({
                        'pmid': pmid,
                        'title': title,
                        'authors': authors,
                        'journal': journal,
                        'year': year,
                        'doi': doi,
                        'abstract': abstract,
                        'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    })
                    
                except Exception as e:
                    logger.warning(f"Error parsing article block: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching PubMed details: {e}")
            return []
    
    def download_publication_info(self, nct_id: str, study_title: str) -> Dict:
        """Download publication information for a study."""
        try:
            # Search for publications
            publications = self.search_pubmed_for_study(nct_id, study_title)
            
            if not publications:
                return {
                    'nct_id': nct_id,
                    'study_title': study_title,
                    'publications_found': 0,
                    'publications': [],
                    'status': 'no_publications_found'
                }
            
            # Save publication information to file
            pub_info_file = self.publications_dir / f"{nct_id}_publications.json"
            pub_data = {
                'nct_id': nct_id,
                'study_title': study_title,
                'search_date': datetime.now().isoformat(),
                'publications_found': len(publications),
                'publications': publications
            }
            
            with open(pub_info_file, 'w', encoding='utf-8') as f:
                json.dump(pub_data, f, indent=2, ensure_ascii=False)
            
            # Also save a readable text summary
            self._save_publication_summary(nct_id, study_title, publications)
            
            logger.info(f"Found and saved {len(publications)} publications for study {nct_id}")
            
            return {
                'nct_id': nct_id,
                'study_title': study_title,
                'publications_found': len(publications),
                'publications': publications,
                'status': 'success',
                'info_file': str(pub_info_file)
            }
            
        except Exception as e:
            logger.error(f"Error downloading publication info for {nct_id}: {e}")
            return {
                'nct_id': nct_id,
                'study_title': study_title,
                'publications_found': 0,
                'publications': [],
                'status': 'error',
                'error': str(e)
            }
    
    def _save_publication_summary(self, nct_id: str, study_title: str, publications: List[Dict]):
        """Save a human-readable summary of publications."""
        summary_file = self.publications_dir / f"{nct_id}_publications_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"PUBLICATIONS FOR CLINICAL TRIAL: {nct_id}\n")
            f.write("="*80 + "\n")
            f.write(f"Study Title: {study_title}\n")
            f.write(f"Search Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Publications Found: {len(publications)}\n")
            f.write("\n" + "-"*80 + "\n")
            
            for i, pub in enumerate(publications, 1):
                f.write(f"\n{i}. {pub['title']}\n")
                f.write(f"   PMID: {pub['pmid']}\n")
                f.write(f"   Authors: {', '.join(pub['authors'][:3])}\n")
                f.write(f"   Journal: {pub['journal']}\n")
                f.write(f"   Year: {pub['year']}\n")
                if pub['doi']:
                    f.write(f"   DOI: {pub['doi']}\n")
                f.write(f"   URL: {pub['url']}\n")
                if pub['abstract']:
                    abstract = pub['abstract'][:300] + "..." if len(pub['abstract']) > 300 else pub['abstract']
                    f.write(f"   Abstract: {abstract}\n")
                f.write("\n" + "-"*40 + "\n")

if __name__ == "__main__":
    main()
