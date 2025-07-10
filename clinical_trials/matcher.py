#!/usr/bin/env python3
"""
Patient-study matching logic.

This module handles the core logic of matching patients with clinical trials.
"""

import logging
import time
from typing import Dict, List

from .models import ClinicalTrialMatch
from .api import ClinicalTrialsAPI
from .llm_matcher import LLMStudyMatcher
from .publications import PublicationDownloader

logger = logging.getLogger(__name__)


class PatientStudyMatcher:
    """Matches patients with relevant clinical trials using LLM evaluation."""
    
    def __init__(self, api: ClinicalTrialsAPI, llm_matcher: LLMStudyMatcher, pub_downloader: PublicationDownloader = None):
        self.api = api
        self.llm_matcher = llm_matcher
        self.pub_downloader = pub_downloader
        
    def find_relevant_studies(self, 
                            patient_data: Dict, 
                            min_relevance_score: float = 0.3,
                            max_studies: int = 50) -> List[ClinicalTrialMatch]:
        """
        Find relevant clinical trials for a patient using LLM evaluation.
        
        Args:
            patient_data: Patient information dictionary
            min_relevance_score: Minimum relevance score to include study (increased from 0.1 to 0.3)
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
                    if self.pub_downloader:
                        logger.info(f"Downloading publications for study {nct_id}")
                        publications_info = self.pub_downloader.download_publication_info(
                            nct_id, identification.get("briefTitle", "")
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
        
        logger.info(f"Found {len(matches)} relevant studies (score >= {min_relevance_score})")
        
        return matches[:max_studies]
