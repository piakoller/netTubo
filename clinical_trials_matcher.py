#!/usr/bin/env python3
"""
ClinicalTrials.gov API Patient-Study Matching System

This script queries the ClinicalTrials.gov API to find relevant clinical trials
for patients based on their diagnosis and characteristics.
It filters for studies that have both posted results AND publications, ensuring
high-quality evidence-based trial matches.
"""

import json
import logging
import re
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

MIN_RELEVANCE_SCORE = 1.0  # Include clinically relevant matches (YES decisions)

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
            "max_tokens": 1500,  # Increased for detailed structured analysis
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
You are a highly experienced specialist in neuroendocrine tumors (NETs), tasked with rigorously evaluating clinical trial eligibility for a specific patient. Your goal is to identify only those studies that are **clearly relevant and appropriate** for the patient's diagnosis, tumor biology, and treatment status. Avoid speculative matches and exclude marginally relevant studies.

PATIENT PROFILE:
{patient_data_formatted}

STUDY TO EVALUATE:
Title: {study_title}
Description: {detailed_description}

STRICT EVALUATION FRAMEWORK – Apply the following filters carefully and conservatively. A study must meet **all core criteria** to be considered suitable.

1. DIAGNOSTIC RELEVANCE (MANDATORY):
   ✓ The study must **explicitly target neuroendocrine tumors (NETs)**.
   ✓ Patient's **primary tumor site** must be included in the study population, or the study must allow **all NET sites**.
   ✗ Studies referring only to general "solid tumors" are **not acceptable** unless NETs are **specifically listed**.
   ✗ Do not include studies with vague or inferred inclusion of NETs.

2. TUMOR CHARACTERISTICS (STRICT MATCHING):
   ✓ Tumor **grade** must match exactly or fall within a clearly accepted adjacent category (e.g., G1–G2).
   ✓ Tumor **stage or extent** must be explicitly compatible with the study population.
   ✓ Functional status (e.g., hormone-secreting tumors) must be aligned if relevant.
   ✗ Exclude if patient's tumor characteristics deviate significantly from those described in the study.

3. TREATMENT CONTEXT (HISTORICAL COMPATIBILITY):
   ✓ The **line of therapy** must match the patient’s current treatment status (e.g., first-line, refractory).
   ✓ Patient must not have received **prior therapies that conflict** with study exclusion criteria.
   ✓ Study should be appropriate for either current disease state or well-defined future treatment planning.
   ✗ Exclude if prior treatments disqualify the patient.

4. CLINICAL ELIGIBILITY (BASIC REQUIREMENTS):
   ✓ Patient must fall within the **study’s age range**.
   ✓ Patient’s **performance status** must meet the study minimum (e.g., ECOG 0–2).
   ✓ Patient must likely meet standard **organ function and lab criteria**.
   ✗ Exclude if any major eligibility requirement is unlikely to be met.

5. THERAPEUTIC RATIONALE (CLINICAL VALUE):
   ✓ The **intervention** must be mechanistically or clinically relevant to this NET type.
   ✓ The study must address a question that is **meaningful and plausible** for the patient’s subtype.
   ✗ Exclude if the study investigates interventions irrelevant to the patient’s tumor biology or treatment needs.

DECISION RULE:
→ Only include studies where the patient is a **strong candidate** and where participation could be **reasonably justified based on clinical fit**. Do not include studies with speculative or borderline relevance.

STRUCTURED ANALYSIS REQUIRED:
DECISION: [YES/NO]

REASONING:
1. Diagnostic Relevance: [Does the study population clearly match the patient’s NET subtype and location?]
2. Tumor Characteristics: [Do grade, stage, and functional status align precisely or within acceptable bounds?]
3. Treatment Context: [Is the therapy line and prior treatment history fully compatible?]
4. Clinical Eligibility: [Would this patient realistically meet all major eligibility criteria?]
5. Therapeutic Rationale: [Would the intervention be clinically meaningful and potentially beneficial?]

CONCLUSION: [Summarize the strict clinical justification for inclusion or exclusion.]
"""
        try:
            llm_response = self.call_llm(prompt)
            
            if "ERROR:" in llm_response:
                logger.error(f"LLM evaluation failed: {llm_response}")
                return 0.0, "LLM evaluation failed"
            
            # Parse LLM response - handle both old and new structured formats
            relevance_score = 0.0
            explanation = "No explanation provided"
            
            lines = llm_response.split('\n')
            reasoning_started = False
            reasoning_parts = []
            
            for line in lines:
                if line.startswith("DECISION:"):
                    try:
                        decision_text = line.split(":", 1)[1].strip().upper()
                        # Remove markdown formatting (**, *, _, `, etc.) and extract YES/NO
                        clean_decision = re.sub(r'[*_`~#]', '', decision_text).strip()
                        
                        # Convert YES/NO to numerical score: YES = 1.0, NO = 0.0
                        if "YES" in clean_decision:
                            relevance_score = 1.0
                            logger.debug(f"Parsed decision as YES from: {line}")
                        elif "NO" in clean_decision:
                            relevance_score = 0.0
                            logger.debug(f"Parsed decision as NO from: {line}")
                        else:
                            logger.warning(f"Could not parse decision from: {line} (cleaned: {clean_decision})")
                    except (ValueError, IndexError):
                        logger.warning(f"Could not parse decision from: {line}")
                        
                elif line.startswith("REASONING:"):
                    reasoning_started = True
                    reasoning_text = line.split(":", 1)[1].strip()
                    if reasoning_text:
                        reasoning_parts.append(reasoning_text)
                elif reasoning_started and line.strip():
                    # Continue collecting reasoning text until we hit a new section or end
                    if not line.startswith(("DECISION:", "CONCLUSION:")):
                        reasoning_parts.append(line.strip())
                    else:
                        reasoning_started = False
                        
            # Combine all reasoning parts
            if reasoning_parts:
                explanation = " ".join(reasoning_parts)
                # Limit explanation length for readability
                if len(explanation) > 500:
                    explanation = explanation[:500] + "..."
            
            # If no structured response, try to extract YES/NO from text
            if relevance_score == 0.0 and explanation == "No explanation provided":
                # Look for YES or NO in the response
                if re.search(r'\bYES\b', llm_response.upper()):
                    relevance_score = 1.0
                elif re.search(r'\bNO\b', llm_response.upper()):
                    relevance_score = 0.0
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
                      phase: Optional[List[str]] = None,
                      with_publications: bool = True,
                      with_posted_results: bool = True,
                      max_results: int = 100) -> List[Dict]:
        """
        Search for clinical trials based on condition and other criteria.
        
        Args:
            condition: Medical condition (e.g., "neuroendocrine tumor", "NET")
            intervention: Intervention/treatment (e.g., "PRRT", "everolimus")
            phase: Study phases (e.g., ["PHASE1", "PHASE2", "PHASE3"])
            with_publications: If True, only return studies with publications
            with_posted_results: If True, only return studies with posted results
            max_results: Maximum number of results to return
            
        Returns:
            List of study dictionaries
        """
        self._rate_limit()
        
        # Build query string manually for better control
        query_parts = [f"AREA[Condition]{condition}"]
        
        if intervention:
            query_parts.append(f"AREA[InterventionName]{intervention}")
            
        # Filter by studies with publications (remove hasResults from query as it may not be supported)
        if with_publications:
            query_parts.append("AREA[ReferencesModule]NOT MISSING")
        
        # Note: We'll filter for posted results after retrieving studies since 
        # hasResults:true might not be supported in the API query syntax
        
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
            
            # Filter studies based on our criteria after retrieval
            filtered_studies = []
            for study in studies:
                # Check if study has publications (if required)
                has_publications = True
                if with_publications:
                    references_section = study.get("protocolSection", {}).get("referencesModule")
                    has_publications = references_section is not None and len(references_section) > 0
                
                # Check if study has posted results (if required)
                has_posted_results = True
                if with_posted_results:
                    has_posted_results = study.get("hasResults", False)
                
                # Include study if it meets our criteria
                if has_publications and has_posted_results:
                    filtered_studies.append(study)
            
            logger.info(f"Found {len(studies)} studies for condition: {condition}")
            logger.info(f"After filtering (publications: {with_publications}, posted results: {with_posted_results}): {len(filtered_studies)} studies")
            return filtered_studies
            
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
                    
                    # Apply the same filtering logic to fallback results
                    filtered_studies = []
                    for study in studies:
                        # Check if study has publications (if required)
                        has_publications = True
                        if with_publications:
                            references_section = study.get("protocolSection", {}).get("referencesModule")
                            has_publications = references_section is not None and len(references_section) > 0
                        
                        # Check if study has posted results (if required)
                        has_posted_results = True
                        if with_posted_results:
                            has_posted_results = study.get("hasResults", False)
                        
                        # Include study if it meets our criteria
                        if has_publications and has_posted_results:
                            filtered_studies.append(study)
                    
                    logger.info(f"Basic search found {len(studies)} studies")
                    logger.info(f"After filtering: {len(filtered_studies)} studies")
                    return filtered_studies
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
            min_relevance_score: Minimum score to include study (1.0 = perfect match only)
            max_studies: Maximum number of studies to return
            
        Returns:
            List of ClinicalTrialMatch objects sorted by relevance
        """
        matches = []
        
        # Extract search terms from patient data - comprehensive set for better coverage
        search_terms = [
            "neuroendocrine tumor",
            "NET",
            "neuroendocrine neoplasm",
            "carcinoid",
            "pancreatic neuroendocrine",
            "gastroenteropancreatic neuroendocrine",
            "GEP-NET"
        ]
        
        # Add patient-specific terms based on diagnosis
        diagnosis_text = patient_data.get('main_diagnosis_text', '').lower()
        if 'pankreas' in diagnosis_text or 'pancrea' in diagnosis_text:
            search_terms.extend(["pancreatic NET", "pancreatic neuroendocrine tumor", "pNET"])
        if 'dünndarm' in diagnosis_text or 'ileum' in diagnosis_text or 'small' in diagnosis_text:
            search_terms.extend(["small bowel NET", "ileal NET", "small intestine neuroendocrine"])
        if 'leber' in diagnosis_text or 'liver' in diagnosis_text:
            search_terms.extend(["neuroendocrine liver metastases"])
        if 'g1' in diagnosis_text:
            search_terms.extend(["grade 1 neuroendocrine", "well differentiated neuroendocrine"])
        if 'g2' in diagnosis_text:
            search_terms.extend(["grade 2 neuroendocrine", "moderately differentiated neuroendocrine"])
        if 'g3' in diagnosis_text:
            search_terms.extend(["grade 3 neuroendocrine", "poorly differentiated neuroendocrine"])
        
        # Remove duplicates while preserving order
        search_terms = list(dict.fromkeys(search_terms))
        
        # Search for studies with different terms - only studies with publications and posted results
        all_studies = []
        for term in search_terms:
            logger.info(f"Searching for studies with term: {term} (with publications and posted results only)")
            studies = self.api.search_studies(
                condition=term,
                with_publications=True,  # Only studies with publications
                with_posted_results=True,  # Only studies with posted results
                max_results=30  # Get studies to evaluate with ultra-strict LLM criteria
            )
            all_studies.extend(studies)
            # Add small delay between different search terms
            time.sleep(0.5)
        
        # Remove duplicates based on NCT ID and verify studies have both publications and posted results
        unique_studies = {}
        for study in all_studies:
            nct_id = study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
            
            # Check if study has publications
            references_section = study.get("protocolSection", {}).get("referencesModule")
            has_publications = references_section is not None and len(references_section) > 0
            
            # Check if study has posted results (hasResults field is at the root level)
            has_posted_results = study.get("hasResults", False)
            
            if nct_id and nct_id not in unique_studies and has_publications and has_posted_results:
                unique_studies[nct_id] = study
                logger.debug(f"Study {nct_id} has both publications and posted results - including in evaluation")
            elif nct_id and not (has_publications and has_posted_results):
                missing_criteria = []
                if not has_publications:
                    missing_criteria.append("publications")
                if not has_posted_results:
                    missing_criteria.append("posted results")
                logger.debug(f"Study {nct_id} missing {', '.join(missing_criteria)} - excluding from evaluation")
        
        logger.info(f"Found {len(unique_studies)} unique studies with both publications and posted results to evaluate")
        
        # Use LLM to evaluate each study
        evaluated_count = 0
        for nct_id, study in unique_studies.items():
            try:
                evaluated_count += 1
                logger.info(f"Evaluating study {evaluated_count}/{len(unique_studies)}: {nct_id}")
                
                # Use LLM to calculate relevance
                score, reason = self.llm_matcher.evaluate_study_relevance(patient_data, study)
                
                logger.info(f"Study {nct_id} - Decision: {'YES' if score >= 1.0 else 'NO'} (Score: {score:.1f})")
                
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
                    if self.pub_downloader:
                        logger.info(f"Downloading publications for study {nct_id}")
                        # First extract any references from the study data itself
                        references_module = protocol.get("referencesModule", {})
                        study_references = []
                        
                        # Extract references if available in the study data
                        if references_module:
                            references = references_module.get("references", [])
                            for ref in references:
                                if ref.get("pmid") or ref.get("citation"):
                                    study_ref = {
                                        "pmid": ref.get("pmid", ""),
                                        "citation": ref.get("citation", ""),
                                        "type": ref.get("type", ""),
                                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{ref.get('pmid')}/" if ref.get("pmid") else ""
                                    }
                                    study_references.append(study_ref)
                        
                        # Then search PubMed for additional publications
                        publications_info = self.pub_downloader.download_publication_info(
                            nct_id, identification.get("briefTitle", ""), study_references
                        )
                    
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
        
        logger.info(f"Found {len(matches)} clinically relevant studies with publications and posted results (score >= {min_relevance_score})")
        
        return matches[:max_studies]

def generate_study_report(patient_data: Dict, matches: List[ClinicalTrialMatch]) -> str:
    """Generate a formatted report of clinically relevant studies with publications and posted results for a patient."""
    
    report = []
    report.append("="*80)
    report.append("CLINICAL TRIALS MATCHES REPORT (STUDIES WITH PUBLICATIONS AND POSTED RESULTS ONLY)")
    report.append("="*80)
    
    # Patient information
    report.append(f"\nPatient ID: {patient_data.get('ID', 'Unknown')}")
    report.append(f"Diagnosis: {patient_data.get('main_diagnosis_text', 'Not specified')}")
    report.append(f"Clinical Question: {patient_data.get('Fragestellung', 'Not specified')}")
    report.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    report.append(f"\nFound {len(matches)} relevant clinical trials (with publications and posted results)")
    report.append("-"*80)
    
    for i, match in enumerate(matches, 1):
        report.append(f"\n{i}. {match.title}")
        report.append(f"   NCT ID: {match.nct_id}")
        report.append(f"   Status: {match.status}")
        report.append(f"   Phase: {match.phase}")
        report.append(f"   Clinically Relevant: {'YES' if match.relevance_score >= 1.0 else 'NO'} (Score: {match.relevance_score:.1f})")
        report.append(f"   Relevance Reason: {match.relevance_reason}")
        report.append(f"   Has Posted Results: YES (verified)")
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
        if match.publications and match.publications.get('publications_found', 0) > 0:
            report.append(f"   Publications Found: {match.publications['publications_found']}")
            for i, pub in enumerate(match.publications['publications'][:3], 1):  # Show first 3 publications
                report.append(f"     {i}. {pub['title']} ({pub['year']})")
                report.append(f"        PMID: {pub['pmid']} | Journal: {pub['journal']}")
                if pub.get('doi'):
                    report.append(f"        DOI: {pub['doi']}")
                report.append(f"        Link: {pub['url']}")
        elif match.publications:
            report.append(f"   Publications: No publications found")
        
        report.append("-"*40)
    
    return "\n".join(report)

def test_api_connection():
    """Test if the ClinicalTrials.gov API is working."""
    api = ClinicalTrialsAPI(rate_limit_delay=1.0)
    
    print("Testing ClinicalTrials.gov API connection...")
    
    # Test with a simple, common condition (no filtering for initial test)
    test_studies = api.search_studies("cancer", with_publications=False, with_posted_results=False, max_results=5)
    
    if test_studies:
        print(f"✅ API connection successful! Found {len(test_studies)} studies for 'cancer'")
        
        # Now test with filtering
        print("\nTesting with publications and posted results filtering...")
        filtered_studies = api.search_studies("cancer", with_publications=True, with_posted_results=True, max_results=5)
        print(f"✅ Filtering working! Found {len(filtered_studies)} studies with publications and posted results")
        
        # Show first study as example
        if test_studies:
            first_study = test_studies[0]
            protocol = first_study.get("protocolSection", {})
            identification = protocol.get("identificationModule", {})
            has_results = first_study.get("hasResults", False)
            print(f"Example study: {identification.get('briefTitle', 'No title')}")
            print(f"NCT ID: {identification.get('nctId', 'No ID')}")
            print(f"Has posted results: {has_results}")
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
    
    # Initialize API, LLM matcher, and patient matcher (no publication downloading)
    api = ClinicalTrialsAPI(rate_limit_delay=1.0)
    llm_matcher = LLMStudyMatcher(OPENROUTER_API_KEY)
    # llm_matcher = LLMStudyMatcher(OPENROUTER_API_KEY, model="google/gemini-2.5-pro-exp-03-25")
    pub_downloader = None  # Disable publication downloading
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
                min_relevance_score=MIN_RELEVANCE_SCORE,  # Include clinically relevant matches (YES decisions)
                max_studies=15  # Studies with good clinical relevance
            )
            
            # Generate report
            report = generate_study_report(patient_data, matches)
            
            # Save individual patient report
            patient_report_file = OUTPUT_DIR / f"patient_{patient_id}_clinical_trials_llm.txt"
            with open(patient_report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"Found {len(matches)} clinically relevant studies with publications and posted results for patient {patient_id}")
            
            # Store results for summary
            patient_result = {
                "patient_id": str(patient_id),
                "patient_data": patient_data,
                "matches_found": len(matches),
                "matches_with_publications": len(matches),  # All matches have publications (filtered)
                "matches_with_posted_results": len(matches),  # All matches have posted results (filtered)
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
                        "has_posted_results": True,  # All matches have posted results (filtered)
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
    
    # Calculate score distribution (no publication statistics since downloads are disabled)
    all_scores = []
    
    for result in all_results:
        for match in result["matches"]:
            all_scores.append(match["relevance_score"])
    
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    relevant_count = len([s for s in all_scores if s >= 1.0])  # Clinically relevant matches
    
    summary_report = [
        "="*80,
        "LLM-BASED CLINICAL TRIALS MATCHING SUMMARY (STUDIES WITH PUBLICATIONS AND POSTED RESULTS ONLY)",
        "="*80,
        f"Total patients processed: {len(all_results)}",
        f"Total studies found: {total_matches} (all with publications and posted results)",
        f"Average studies per patient: {avg_matches:.1f}",
        f"Clinically relevant matches: {relevant_count} (all matches are clinically relevant)",
        f"Studies with posted results: {total_matches}/{total_matches} (100% - filtering requirement)",
        f"Studies with publications: {total_matches}/{total_matches} (100% - filtering requirement)",
        f"Evaluation method: LLM-based semantic matching",
        f"Results saved to: {OUTPUT_DIR}",
        f"Summary data: {summary_file}",
        "="*80,
        "Note: Only studies with publications AND posted results were evaluated using AI",
        "to assess clinical relevance based on patient diagnosis and clinical context.",
        "Publication downloading was disabled for faster processing."
    ]
    
    print("\n".join(summary_report))
    logger.info("LLM-based clinical trials matching (studies with publications and posted results) completed successfully")

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
    
    def download_publication_info(self, nct_id: str, study_title: str, study_references: List[Dict] = None) -> Dict:
        """Download publication information for a study."""
        try:
            all_publications = []
            
            # Add study references if provided
            if study_references:
                for ref in study_references:
                    publication = {
                        'pmid': ref.get('pmid', 'N/A'),
                        'title': ref.get('citation', 'Study Reference')[:100] + '...' if len(ref.get('citation', '')) > 100 else ref.get('citation', 'Study Reference'),
                        'authors': [],
                        'journal': 'From Clinical Trial Data',
                        'year': 'N/A',
                        'doi': None,
                        'abstract': ref.get('citation', ''),
                        'url': ref.get('url', f"https://pubmed.ncbi.nlm.nih.gov/{ref.get('pmid')}/" if ref.get('pmid') else ""),
                        'source': 'study_data'
                    }
                    all_publications.append(publication)
            
            # Search for additional publications in PubMed
            pubmed_publications = self.search_pubmed_for_study(nct_id, study_title)
            
            # Add PubMed publications, avoiding duplicates by PMID
            existing_pmids = {pub.get('pmid') for pub in all_publications if pub.get('pmid') and pub.get('pmid') != 'N/A'}
            for pub in pubmed_publications:
                if pub.get('pmid') not in existing_pmids:
                    pub['source'] = 'pubmed_search'
                    all_publications.append(pub)
            
            if not all_publications:
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
                'publications_found': len(all_publications),
                'publications': all_publications
            }
            
            with open(pub_info_file, 'w', encoding='utf-8') as f:
                json.dump(pub_data, f, indent=2, ensure_ascii=False)
            
            # Also save a readable text summary
            self._save_publication_summary(nct_id, study_title, all_publications)
            
            logger.info(f"Found and saved {len(all_publications)} publications for study {nct_id}")
            
            return {
                'nct_id': nct_id,
                'study_title': study_title,
                'publications_found': len(all_publications),
                'publications': all_publications,
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
                if pub.get('authors'):
                    f.write(f"   Authors: {', '.join(pub['authors'][:3])}\n")
                f.write(f"   Journal: {pub['journal']}\n")
                f.write(f"   Year: {pub['year']}\n")
                if pub.get('doi'):
                    f.write(f"   DOI: {pub['doi']}\n")
                if pub.get('url'):
                    f.write(f"   URL: {pub['url']}\n")
                f.write(f"   Source: {pub.get('source', 'unknown')}\n")
                if pub.get('abstract'):
                    abstract = pub['abstract'][:300] + "..." if len(pub['abstract']) > 300 else pub['abstract']
                    f.write(f"   Abstract: {abstract}\n")
                f.write("\n" + "-"*40 + "\n")

if __name__ == "__main__":
    main()
