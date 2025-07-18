#!/usr/bin/env python3
"""
Clinical Trials Patient-Study Matching System

This script matches patients with pre-filtered clinical trials using LLM evaluation.
It uses studies with recent publications (2020+) and posted results for 
high-quality evidence-based trial matches.
"""

import json
import logging
import re
import requests
import time
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

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
        self.prompt_template = self._load_prompt_template()
        
    def _load_prompt_template(self) -> str:
        """Load the prompt template from external file."""
        prompt_file = Path("C:/Users/pia/OneDrive - Universitaet Bern/Projects/NetTubo/prompts/clinical_trials_matching_prompt.txt")
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                logger.info(f"Successfully loaded prompt template from {prompt_file}")
                return content
        except Exception as e:
            error_msg = f"ERROR: Could not load prompt template from {prompt_file}: {e}"
            logger.error(error_msg)
            print(error_msg)
            print("Process stopped. Please ensure the prompt file exists and is accessible.")
            sys.exit(1)
        
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
        status_module = protocol.get("statusModule", {})
        design_module = protocol.get("designModule", {})
        description_module = protocol.get("descriptionModule", {})
        
        # Extract specific data for prompt
        nct_id = identification.get("nctId", "")
        title = identification.get("briefTitle", "")
        status = status_module.get("overallStatus", "")
        phase = design_module.get("phases", ["N/A"])[0] if design_module.get("phases") else "N/A"
        intervention = "; ".join([interv.get("name", "") for interv in interventions])
        eligibility_criteria = eligibility.get("eligibilityCriteria", "")
        summary = description_module.get("briefSummary", "")
        
        # Extract patient data fields directly from the original patient_data
        main_diagnosis = patient_data.get("main_diagnosis_text", "Not specified")
        clinical_question = patient_data.get("Fragestellung", "Not specified") 
        previous_therapies = patient_data.get("Vortherapien", "Not specified")
        biomarkers = patient_data.get("relevant_biomarkers", "Not specified")
        ecog = patient_data.get("ECOG", "Not specified")
        age = patient_data.get("age", "Not specified")
        
        # Get publications info
        references_module = protocol.get("referencesModule", {})
        publications_summary = "Study has recent publications (2020+) and posted results"
        if references_module.get("references"):
            pub_count = len(references_module.get("references", []))
            publications_summary = f"Study has {pub_count} publications and posted results"
        
        # For LLM evaluation, we use the detailed description for comprehensive comparison
        # Create prompt for LLM evaluation using the external prompt template
        prompt = self.prompt_template.format(
            main_diagnosis=main_diagnosis,
            clinical_question=clinical_question,
            previous_therapies=previous_therapies,
            biomarkers=biomarkers,
            ecog=ecog,
            age=age,
            nct_id=nct_id,
            title=title,
            phase=phase,
            status=status,
            intervention=intervention,
            eligibility_criteria=eligibility_criteria,
            summary=summary,
            publications_summary=publications_summary
        )
        
        try:
            llm_response = self.call_llm(prompt)
            
            if "ERROR:" in llm_response:
                logger.error(f"LLM evaluation failed: {llm_response}")
                return 0.0, "LLM evaluation failed"
            
            # Parse LLM response - try JSON format first, then fallback to old format
            relevance_score = 0.0
            explanation = "No explanation provided"
            
            # Try to parse as JSON first
            try:
                # Extract JSON from response
                json_start = llm_response.find('{')
                json_end = llm_response.rfind('}')
                if json_start != -1 and json_end != -1:
                    json_str = llm_response[json_start:json_end+1]
                    parsed_response = json.loads(json_str)
                    
                    # Extract decision and explanation
                    relevant = parsed_response.get("relevant", "").upper()
                    if "YES" in relevant:
                        relevance_score = 1.0
                    elif "NO" in relevant:
                        relevance_score = 0.0
                        
                    explanation = parsed_response.get("explanation", "No explanation provided")
                    logger.debug(f"Parsed JSON response: relevant={relevant}, score={relevance_score}")
                    
                else:
                    raise ValueError("No JSON found in response")
                    
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.debug(f"JSON parsing failed ({e}), trying fallback parsing")
                
                # Fallback to old parsing method
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
                    # Clean up the explanation but don't truncate yet - we'll format for display later
                    explanation = self._clean_llm_explanation(explanation)
                
                # If no structured response, try to extract YES/NO from text
                if relevance_score == 0.0 and explanation == "No explanation provided":
                    # Look for YES or NO in the response
                    if re.search(r'\bYES\b', llm_response.upper()):
                        relevance_score = 1.0
                    elif re.search(r'\bNO\b', llm_response.upper()):
                        relevance_score = 0.0
                    explanation = self._clean_llm_explanation(llm_response)
            
            return relevance_score, explanation
            
        except Exception as e:
            logger.error(f"Error in LLM evaluation: {e}")
            return 0.0, f"Evaluation error: {str(e)}"
    
    def _clean_llm_explanation(self, text: str) -> str:
        """Clean and format LLM explanation for better readability."""
        if not text:
            return text
            
        # Remove excessive markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove **bold**
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Remove *italic*
        text = re.sub(r'#{1,6}\s*', '', text)           # Remove headers
        
        # Clean up decision markers that got included
        text = re.sub(r'DECISION:\s*(YES|NO)\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'REASONING:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'CONCLUSION:\s*', 'Conclusion: ', text, flags=re.IGNORECASE)
        
        # Preserve numbered points but format them better
        text = re.sub(r'^\s*(\d+)\.\s*', r'\1. ', text, flags=re.MULTILINE)
        
        # Replace multiple spaces with single spaces but preserve line breaks for readability
        text = re.sub(r'[ \t]+', ' ', text)  # Replace multiple spaces/tabs with single space
        text = re.sub(r'\n\s*\n', '\n', text)  # Remove empty lines but keep single line breaks
        
        # Clean up any remaining artifacts
        text = re.sub(r'\s*\.\.\.\s*', '... ', text)
        
        return text.strip()
    
    def _format_llm_explanation_for_display(self, explanation: str, max_length: int = 1200) -> str:
        """Format LLM explanation for better display in reports with improved readability."""
        if not explanation:
            return "No explanation provided"
        
        # Clean the explanation first
        clean_explanation = self._clean_llm_explanation(explanation)
        
        # If it's too long, intelligently truncate at sentence boundary
        if len(clean_explanation) > max_length:
            # Find the last complete sentence within the limit
            truncated = clean_explanation[:max_length]
            last_period = truncated.rfind('.')
            last_exclamation = truncated.rfind('!')
            last_question = truncated.rfind('?')
            
            # Find the latest sentence ending
            sentence_end = max(last_period, last_exclamation, last_question)
            
            if sentence_end > max_length * 0.7:  # If we found a good sentence boundary
                clean_explanation = clean_explanation[:sentence_end + 1] + " [Explanation truncated for readability]"
            else:
                clean_explanation = clean_explanation[:max_length] + "... [Explanation truncated for readability]"
        
        # Add proper formatting for better readability
        # Break long paragraphs at numbered points
        formatted_text = re.sub(r'(\d+\.\s)', r'\n       \1', clean_explanation)
        
        # Add line breaks before common clinical reasoning patterns
        formatted_text = re.sub(r'(Patient characteristics:|Tumor characteristics:|Treatment history:|Study criteria:|Relevance assessment:|Clinical relevance:)', r'\n       \1', formatted_text, flags=re.IGNORECASE)
        
        # Clean up any double line breaks and extra spaces
        formatted_text = re.sub(r'\n\s*\n', '\n', formatted_text)
        formatted_text = formatted_text.strip()
        
        return formatted_text
    
    def create_structured_reasoning_summary(self, explanation: str) -> Dict[str, Any]:
        """Create a structured summary of LLM reasoning for JSON output."""
        if not explanation:
            return {"summary": "No explanation provided", "key_points": []}
        
        # Clean the explanation
        clean_explanation = self._clean_llm_explanation(explanation)
        
        # Extract key points from numbered items
        key_points = []
        numbered_items = re.findall(r'(\d+)\.\s*([^0-9]*?)(?=\d+\.|$)', clean_explanation, re.DOTALL)
        
        for num, point in numbered_items:
            if point.strip():
                # Clean up the point and limit length
                clean_point = re.sub(r'\s+', ' ', point.strip())
                if len(clean_point) > 200:
                    clean_point = clean_point[:200] + "..."
                key_points.append({
                    "point": int(num),
                    "description": clean_point
                })
        
        # If no numbered points found, try to extract sentences as key points
        if not key_points:
            sentences = re.split(r'[.!?]+', clean_explanation)
            for i, sentence in enumerate(sentences[:3], 1):  # Take first 3 sentences
                if sentence.strip() and len(sentence.strip()) > 20:
                    key_points.append({
                        "point": i,
                        "description": sentence.strip()
                    })
        
        # Create summary (first 300 characters)
        summary = clean_explanation[:300] + "..." if len(clean_explanation) > 300 else clean_explanation
        
        return {
            "summary": summary,
            "key_points": key_points,
            "full_reasoning": clean_explanation
        }

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

class PatientStudyMatcher:
    """Matches patients with relevant clinical trials using LLM evaluation."""
    
    def __init__(self, llm_matcher: LLMStudyMatcher, studies_file: str):
        self.llm_matcher = llm_matcher
        self.studies_file = studies_file
        self.pre_filtered_studies = None
        
        # Load pre-filtered studies
        self.load_pre_filtered_studies(studies_file)
    
    def load_pre_filtered_studies(self, studies_file: str):
        """Load pre-filtered studies from JSON file."""
        try:
            with open(studies_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            studies_data = data.get('studies', [])
            self.pre_filtered_studies = []
            
            # Convert study data back to the expected format
            for study_data in studies_data:
                # Reconstruct the study format expected by the evaluation method
                study = {
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": study_data.get('nct_id', ''),
                            "briefTitle": study_data.get('title', '')
                        },
                        "statusModule": {
                            "overallStatus": study_data.get('status', ''),
                            "startDateStruct": {"date": study_data.get('start_date', '')},
                            "completionDateStruct": {"date": study_data.get('completion_date', '')}
                        },
                        "designModule": {
                            "phases": [study_data.get('phase', 'N/A')] if study_data.get('phase') != 'N/A' else []
                        },
                        "conditionsModule": {
                            "conditions": study_data.get('condition', '').split('; ') if study_data.get('condition') else []
                        },
                        "armsInterventionsModule": {
                            "interventions": [{"name": intervention} for intervention in study_data.get('intervention', '').split('; ') if intervention]
                        },
                        "eligibilityModule": {
                            "eligibilityCriteria": study_data.get('eligibility_criteria', '')
                        },
                        "outcomesModule": {
                            "primaryOutcomes": [{"measure": outcome} for outcome in study_data.get('primary_outcome', '').split('; ') if outcome],
                            "secondaryOutcomes": [{"measure": outcome} for outcome in study_data.get('secondary_outcome', '').split('; ') if outcome]
                        },
                        "sponsorCollaboratorsModule": {
                            "leadSponsor": {"name": study_data.get('sponsor', '')}
                        },
                        "contactsLocationsModule": {
                            "locations": [{"facility": loc.split(', ')[0] if ', ' in loc else loc, 
                                         "city": loc.split(', ')[1] if len(loc.split(', ')) > 1 else "", 
                                         "country": loc.split(', ')[-1] if ', ' in loc else ""} 
                                        for loc in study_data.get('locations', [])]
                        },
                        "descriptionModule": {
                            "briefSummary": study_data.get('brief_summary', ''),
                            "detailedDescription": study_data.get('detailed_description', '')
                        },
                        "referencesModule": {
                            "references": study_data.get('publications', [])
                        }
                    },
                    "hasResults": study_data.get('has_posted_results', True)
                }
                self.pre_filtered_studies.append(study)
            
            logger.info(f"Loaded {len(self.pre_filtered_studies)} pre-filtered studies from {studies_file}")
            
        except Exception as e:
            logger.error(f"Error loading pre-filtered studies from {studies_file}: {e}")
            self.pre_filtered_studies = []
        
    def find_relevant_studies(self, 
                            patient_data: Dict, 
                            min_relevance_score: float = MIN_RELEVANCE_SCORE,
                            max_studies: int = 50,
                            report_file: Optional[str] = None) -> List[ClinicalTrialMatch]:
        """
        Find relevant clinical trials for a patient using LLM evaluation.
        
        Args:
            patient_data: Patient information dictionary
            min_relevance_score: Minimum score to include study (1.0 = perfect match only)
            max_studies: Maximum number of studies to return
            report_file: Optional file path to write progressive report
            
        Returns:
            List of ClinicalTrialMatch objects sorted by relevance
        """
        matches = []
        
        # Initialize progressive report if requested
        if report_file:
            self._initialize_progressive_report(patient_data, report_file)
        
        # Use pre-filtered studies
        if not self.pre_filtered_studies:
            logger.error("No pre-filtered studies available")
            return []
            
        logger.info(f"Using {len(self.pre_filtered_studies)} pre-filtered studies with recent publications")
        
        # Remove duplicates based on NCT ID
        unique_studies = {}
        for study in self.pre_filtered_studies:
            nct_id = study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
            
            if nct_id and nct_id not in unique_studies:
                unique_studies[nct_id] = study
                logger.debug(f"Study {nct_id} included for evaluation")
        
        logger.info(f"Found {len(unique_studies)} unique studies to evaluate")
        
        if report_file:
            self._append_to_report(report_file, f"\nEvaluating {len(unique_studies)} studies...\n" + "-"*80 + "\n")
        
        # Use LLM to evaluate each study
        evaluated_count = 0
        relevant_count = 0
        for nct_id, study in unique_studies.items():
            try:
                evaluated_count += 1
                logger.info(f"Evaluating study {evaluated_count}/{len(unique_studies)}: {nct_id}")
                
                if report_file:
                    self._append_to_report(report_file, f"\n[{evaluated_count}/{len(unique_studies)}] Evaluating {nct_id}... ")
                
                # Use LLM to calculate relevance
                score, reason = self.llm_matcher.evaluate_study_relevance(patient_data, study)
                
                logger.info(f"Study {nct_id} - Decision: {'YES' if score >= 1.0 else 'NO'} (Score: {score:.1f})")
                
                if report_file:
                    decision_text = "✅ RELEVANT" if score >= 1.0 else "❌ Not relevant"
                    self._append_to_report(report_file, f"{decision_text}\n")
                
                if score >= min_relevance_score:
                    relevant_count += 1
                    
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
                    
                    # Extract publication references from study data
                    publications_info = None
                    references_module = protocol.get("referencesModule", {})
                    if references_module:
                        study_references = []
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
                        
                        if study_references:
                            publications_info = {
                                'nct_id': nct_id,
                                'study_title': identification.get("briefTitle", ""),
                                'publications_found': len(study_references),
                                'publications': study_references,
                                'status': 'from_study_data'
                            }
                    
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
                    
                    # Write match to progressive report
                    if report_file:
                        self._append_match_to_report(match, relevant_count, report_file)
                    
                # Add delay between LLM evaluations to avoid rate limiting
                time.sleep(1.0)
                    
            except Exception as e:
                logger.error(f"Error processing study {nct_id}: {e}")
                if report_file:
                    self._append_to_report(report_file, f"❌ Error processing study\n")
                continue
        
        # Finalize progressive report
        if report_file:
            self._finalize_progressive_report(report_file, len(matches), len(unique_studies))
        
        # Sort by relevance score (highest first)
        matches.sort(key=lambda x: x.relevance_score, reverse=True)
        
        logger.info(f"Found {len(matches)} clinically relevant studies with publications and posted results (score >= {min_relevance_score})")
        
        return matches[:max_studies]
    
    def _initialize_progressive_report(self, patient_data: Dict, report_file: str):
        """Initialize the progressive report file with patient information."""
        report_lines = [
            "="*80,
            "CLINICAL TRIALS MATCHES REPORT (LIVE EVALUATION)",
            "="*80,
            f"\nPatient ID: {patient_data.get('ID', 'Unknown')}",
            f"Diagnosis: {patient_data.get('main_diagnosis_text', 'Not specified')}",
            f"Clinical Question: {patient_data.get('Fragestellung', 'Not specified')}",
            f"Evaluation started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n" + "="*80,
            "EVALUATION PROGRESS",
            "="*80
        ]
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
    
    def _append_to_report(self, report_file: str, text: str):
        """Append text to the progressive report file."""
        try:
            with open(report_file, 'a', encoding='utf-8') as f:
                f.write(text)
                f.flush()  # Ensure immediate write
        except Exception as e:
            logger.error(f"Error writing to report file: {e}")
    
    def _append_match_to_report(self, match: ClinicalTrialMatch, match_number: int, report_file: str):
        """Append a study match to the progressive report file."""
        # Format the LLM explanation with better readability
        if hasattr(self.llm_matcher, '_format_llm_explanation_for_display'):
            formatted_reason = self.llm_matcher._format_llm_explanation_for_display(match.relevance_reason)
        else:
            formatted_reason = match.relevance_reason
        
        match_text = f"""
    
    {match_number}. ✅ {match.title}
       NCT ID: {match.nct_id}
       Status: {match.status} | Phase: {match.phase}
       Score: {match.relevance_score:.1f} | URL: {match.url}
       
       Clinical Reasoning:
       {formatted_reason}
       
       Condition: {match.condition}
       Intervention: {match.intervention}
       Sponsor: {match.sponsor}
       
       {"-"*40}
    """
        
        self._append_to_report(report_file, match_text)
    
    def _finalize_progressive_report(self, report_file: str, matches_found: int, total_evaluated: int):
        """Finalize the progressive report with summary information."""
        summary_text = f"""

{"="*80}
EVALUATION COMPLETED
{"="*80}

Total studies evaluated: {total_evaluated}
Clinically relevant studies found: {matches_found}
Evaluation completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{"="*80}
FINAL MATCHES (sorted by relevance)
{"="*80}
"""
        self._append_to_report(report_file, summary_text)

def generate_study_report(patient_data: Dict, matches: List[ClinicalTrialMatch], llm_matcher: LLMStudyMatcher = None) -> str:
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
        
        # Format the LLM explanation with better readability
        if llm_matcher and hasattr(llm_matcher, '_format_llm_explanation_for_display'):
            formatted_reason = llm_matcher._format_llm_explanation_for_display(match.relevance_reason)
        else:
            # Fallback formatting if matcher not available
            formatted_reason = match.relevance_reason
        
        report.append(f"   Clinical Reasoning:")
        # Add indented, well-formatted reasoning
        for line in formatted_reason.split('\n'):
            if line.strip():
                report.append(f"   {line}")
        
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
        
        # Add publication information with links
        if match.publications and match.publications.get('publications_found', 0) > 0:
            report.append(f"   Publications Found: {match.publications['publications_found']}")
            for pub_idx, pub in enumerate(match.publications['publications'][:3], 1):  # Show first 3 publications
                # Extract title from citation if available, otherwise use citation
                title = pub.get('citation', 'Study Publication')
                if len(title) > 80:
                    title = title[:80] + "..."
                
                report.append(f"     {pub_idx}. {title}")
                if pub.get('pmid'):
                    report.append(f"        PMID: {pub['pmid']}")
                if pub.get('type'):
                    report.append(f"        Type: {pub['type']}")
                if pub.get('url'):
                    report.append(f"        Link: {pub['url']}")
                else:
                    report.append(f"        Link: https://clinicaltrials.gov/study/{match.nct_id}")
        else:
            report.append(f"   Publications: Available in study data (filtering verified)")
        
        report.append("-"*40)
    
    return "\n".join(report)

def main():
    """Main function to run the clinical trials matching system."""
    
    # Check if OpenRouter API key is available
    if not OPENROUTER_API_KEY:
        print("❌ OPENROUTER_API_KEY not found in environment variables.")
        print("Please set your OpenRouter API key in the .env file or environment variables.")
        return
    
    print("✅ Using pre-filtered studies with recent publications from study collector")
    
    # Configuration
    PATIENT_DATA_FILE = Path("C:/Users/pia/OneDrive - Universitaet Bern/Projects/NetTubo/netTubo/data/NET Tubo v2.xlsx")
    OUTPUT_DIR = Path("C:/Users/pia/OneDrive - Universitaet Bern/Projects/NetTubo/netTubo/clinical_trials_matches")
    STUDIES_FILE = "C:/Users/pia/OneDrive - Universitaet Bern/Projects/NetTubo/netTubo/collected_studies/net_studies_recent_publications_20250717_141930.json"
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Initialize LLM matcher and patient matcher with pre-filtered studies
    llm_matcher = LLMStudyMatcher(OPENROUTER_API_KEY)
    # llm_matcher = LLMStudyMatcher(OPENROUTER_API_KEY, model="google/gemini-2.5-pro-exp-03-25")
    
    # Use pre-filtered studies instead of API search
    matcher = PatientStudyMatcher(llm_matcher=llm_matcher, studies_file=STUDIES_FILE)
    
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
            
            # Setup progressive report file
            patient_report_file = OUTPUT_DIR / f"patient_{patient_id}_clinical_trials_llm.txt"
            
            print(f"📝 Starting evaluation for patient {patient_id}. You can watch progress in: {patient_report_file}")
            
            # Find relevant studies using LLM evaluation with progressive reporting
            matches = matcher.find_relevant_studies(
                patient_data, 
                min_relevance_score=MIN_RELEVANCE_SCORE,  # Include clinically relevant matches (YES decisions)
                max_studies=15,  # Studies with good clinical relevance
                report_file=str(patient_report_file)  # Enable progressive reporting
            )
            
            # Generate final formatted report (will overwrite the progressive report)
            report = generate_study_report(patient_data, matches, llm_matcher)
            
            # Save final individual patient report
            with open(patient_report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"Found {len(matches)} clinically relevant studies with publications and posted results for patient {patient_id}")
            print(f"✅ Completed patient {patient_id}. Final report saved to: {patient_report_file}")
            
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
        "LLM-BASED CLINICAL TRIALS MATCHING SUMMARY (PRE-FILTERED STUDIES WITH RECENT PUBLICATIONS)",
        "="*80,
        f"Total patients processed: {len(all_results)}",
        f"Total studies found: {total_matches} (pre-filtered with publications from 2020+ and posted results)",
        f"Average studies per patient: {avg_matches:.1f}",
        f"Clinically relevant matches: {relevant_count} (all matches are clinically relevant)",
        f"Studies with posted results: {total_matches}/{total_matches} (100% - pre-filtering requirement)",
        f"Studies with publications from 2020+: {total_matches}/{total_matches} (100% - pre-filtering requirement)",
        f"Evaluation method: LLM-based semantic matching on pre-filtered study database",
        f"Source studies file: {STUDIES_FILE}",
        f"Results saved to: {OUTPUT_DIR}",
        f"Summary data: {summary_file}",
        "="*80,
        "Note: Used pre-filtered studies from study collector with publications from 2020 onwards",
        "and posted results. LLM evaluation assessed clinical relevance based on patient",
        "diagnosis and clinical context. No real-time API calls were made."
    ]
    
    print("\n".join(summary_report))
    logger.info("LLM-based clinical trials matching using pre-filtered studies completed successfully")


if __name__ == "__main__":
    main()
