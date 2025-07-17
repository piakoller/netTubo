#!/usr/bin/env python3
"""
Clinical Trials Study Collector

This script queries the ClinicalTrials.gov API to collect all studies that match
the filter criteria (studies with both posted results AND publications) without
LLM evaluation. It saves all matching studies for further analysis.
"""

import sys
import os
import json
import time
import logging
import requests
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from clinical_trials_matcher import ClinicalTrialsAPI

try:
    from geopy.geocoders import Nominatim
    from geopy.distance import geodesic
    GEOPY_AVAILABLE = True
except ImportError:
    print("Warning: geopy not available. Location-based scoring will be disabled.")
    GEOPY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocationScorer:
    """Handles geographic distance calculations for study location scoring."""
    
    def __init__(self):
        # Bern, Switzerland coordinates
        self.bern_coordinates = (46.9479, 7.4474)
        self.geocoder = Nominatim(user_agent="net_tubo_study_collector") if GEOPY_AVAILABLE else None
        self.location_cache = {}  # Cache for geocoded locations
        
    def get_coordinates(self, location_string: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a location string with caching."""
        if not GEOPY_AVAILABLE or not location_string:
            return None
            
        # Clean the location string
        clean_location = location_string.split(',')[0:2]  # Take facility and city
        clean_location = ', '.join(clean_location).strip()
        
        if clean_location in self.location_cache:
            return self.location_cache[clean_location]
        
        try:
            location = self.geocoder.geocode(clean_location, timeout=5)
            if location:
                coords = (location.latitude, location.longitude)
                self.location_cache[clean_location] = coords
                return coords
            else:
                self.location_cache[clean_location] = None
                return None
        except Exception as e:
            logger.debug(f"Geocoding failed for {clean_location}: {e}")
            self.location_cache[clean_location] = None
            return None
    
    def calculate_distance(self, location_string: str) -> Optional[float]:
        """Calculate distance from Bern to given location in kilometers."""
        coordinates = self.get_coordinates(location_string)
        if coordinates:
            try:
                distance = geodesic(self.bern_coordinates, coordinates).kilometers
                return distance
            except Exception as e:
                logger.debug(f"Distance calculation failed: {e}")
                return None
        return None
    
    def score_location_proximity(self, locations: List[str]) -> Tuple[int, str]:
        """
        Score location proximity to Bern.
        
        Args:
            locations: List of location strings
            
        Returns:
            Tuple of (score, description)
        """
        if not locations or not GEOPY_AVAILABLE:
            return 0, "No location data"
        
        distances = []
        for location in locations[:5]:  # Check first 5 locations to avoid API limits
            distance = self.calculate_distance(location)
            if distance is not None:
                distances.append(distance)
        
        if not distances:
            return 0, "Could not geocode locations"
        
        min_distance = min(distances)
        
        # Scoring based on distance ranges
        if min_distance <= 50:  # Within 50km (Switzerland/neighboring regions)
            return 20, f"Very close ({min_distance:.0f}km)"
        elif min_distance <= 200:  # Within 200km (Central Europe)
            return 15, f"Close ({min_distance:.0f}km)"
        elif min_distance <= 500:  # Within 500km (Europe)
            return 10, f"Moderate distance ({min_distance:.0f}km)"
        elif min_distance <= 1000:  # Within 1000km (Extended Europe)
            return 5, f"Far ({min_distance:.0f}km)"
        else:  # Very far
            return 2, f"Very far ({min_distance:.0f}km)"

@dataclass
class StudyRecord:
    """Represents a clinical trial study record."""
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
    locations: List[str]
    url: str
    has_posted_results: bool
    has_publications: bool
    publications_count: int
    search_terms_matched: List[str]

class StudyCollector:
    """Collects all studies matching filter criteria without LLM evaluation."""
    
    def __init__(self, api: ClinicalTrialsAPI, output_dir: Path):
        self.api = api
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # Default strict matching parameters
        self.min_relevance_score = 25
        self.max_matches_per_patient = 10
        
        # Initialize location scorer
        self.location_scorer = LocationScorer()
        
    def collect_net_studies_simple(self) -> List[StudyRecord]:
        """
        Simple API call to collect Neuroendocrine Tumour studies with posted results.
        Uses direct API query: AREA[ConditionSearch]("Neuroendocrine Tumour") AND (AREA[HasResults] true)
        
        Returns:
            List of StudyRecord objects
        """
        all_studies = []
        
        # More restrictive query - only Phase 2/3 completed studies with multiple publications
        query = '''AREA[ConditionSearch]("Neuroendocrine Tumour") AND 
                   (AREA[HasResults] true) AND 
                   (AREA[ReferencesModule]NOT MISSING) AND
                   (AREA[Phase] ("PHASE2" OR "PHASE3")) AND
                   (AREA[OverallStatus] ("COMPLETED" OR "ACTIVE_NOT_RECRUITING"))'''
        
        logger.info(f"Using simple API call with query: {query}")
        
        try:
            # Make direct API call with the specified query
            url = f"{self.api.BASE_URL}/studies"
            params = {
                "format": "json",
                "pageSize": 1000,  # Get all available results
                "query.term": query
            }
            
            logger.info(f"Making API call to: {url}")
            logger.info(f"Parameters: {params}")
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            logger.info(f"API Response Status: {response.status_code}")
            logger.info(f"Full URL: {response.url}")
            
            data = response.json()
            studies = data.get("studies", [])
            
            logger.info(f"Found {len(studies)} studies from API")
            
            # Process each study (convert to StudyRecord format)
            for study in studies:
                study_record = self._process_study(study, "simple_net_search")
                if study_record:
                    all_studies.append(study_record)
                    logger.debug(f"Added study: {study_record.nct_id} - {study_record.title[:50]}...")
                    
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Error in simple NET search: {e}")
            return []
        
        logger.info(f"Collected {len(all_studies)} studies using simple API call")
        return all_studies
    
    def _process_study(self, study: Dict, search_term: str) -> Optional[StudyRecord]:
        """Process a single study and convert to StudyRecord."""
        try:
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
            references_module = protocol.get("referencesModule", {})
            
            nct_id = identification.get("nctId", "")
            if not nct_id:
                return None
            
            # Check publications
            has_publications = False
            publications_count = 0
            if references_module:
                references = references_module.get("references", [])
                publications_count = len(references)
                has_publications = publications_count > 0
            
            # Check posted results
            has_posted_results = study.get("hasResults", False)
            
            # Extract locations
            locations = []
            location_facilities = contacts_module.get("locations", [])
            for loc in location_facilities:
                facility = loc.get("facility", "")
                city = loc.get("city", "")
                country = loc.get("country", "")
                if facility:
                    locations.append(f"{facility}, {city}, {country}")
            
            study_record = StudyRecord(
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
                locations=locations,
                url=f"https://clinicaltrials.gov/study/{nct_id}",
                has_posted_results=has_posted_results,
                has_publications=has_publications,
                publications_count=publications_count,
                search_terms_matched=[search_term]
            )
            
            return study_record
            
        except Exception as e:
            logger.error(f"Error processing study: {e}")
            return None
    
    def save_studies(self, studies: List[StudyRecord], filename_prefix: str = "collected_studies"):
        """Save collected studies to JSON and text files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_file = self.output_dir / f"{filename_prefix}_{timestamp}.json"
        studies_data = [asdict(study) for study in studies]
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                "collection_date": datetime.now().isoformat(),
                "total_studies": len(studies),
                "studies": studies_data
            }, f, indent=2, ensure_ascii=False)
        
        # Save as readable text report
        txt_file = self.output_dir / f"{filename_prefix}_{timestamp}.txt"
        self._generate_text_report(studies, txt_file)
        
        # Save summary statistics
        stats_file = self.output_dir / f"{filename_prefix}_stats_{timestamp}.json"
        
        logger.info(f"Saved {len(studies)} studies to:")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  Text: {txt_file}")
        logger.info(f"  Stats: {stats_file}")
    
    def _generate_text_report(self, studies: List[StudyRecord], output_file: Path):
        """Generate a human-readable text report of collected studies."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("COLLECTED CLINICAL TRIALS REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Collection Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Studies: {len(studies)}\n")
            f.write(f"Filter Criteria: Studies with publications AND posted results\n")
            f.write("-"*80 + "\n\n")
            
            for i, study in enumerate(studies, 1):
                f.write(f"{i}. {study.title}\n")
                f.write(f"   NCT ID: {study.nct_id}\n")
                f.write(f"   Status: {study.status}\n")
                f.write(f"   Phase: {study.phase}\n")
                f.write(f"   Condition: {study.condition}\n")
                f.write(f"   Intervention: {study.intervention}\n")
                f.write(f"   Sponsor: {study.sponsor}\n")
                f.write(f"   Start Date: {study.start_date}\n")
                f.write(f"   Completion Date: {study.completion_date}\n")
                f.write(f"   Has Posted Results: {study.has_posted_results}\n")
                f.write(f"   Publications Count: {study.publications_count}\n")
                f.write(f"   Search Terms Matched: {', '.join(study.search_terms_matched)}\n")
                f.write(f"   URL: {study.url}\n")
                
                # Add brief summary if available
                if study.brief_summary:
                    summary = study.brief_summary[:300] + "..." if len(study.brief_summary) > 300 else study.brief_summary
                    f.write(f"   Summary: {summary}\n")
                
                # Show first few locations
                if study.locations:
                    f.write(f"   Locations: {'; '.join(study.locations[:3])}\n")
                    if len(study.locations) > 3:
                        f.write(f"              ... and {len(study.locations) - 3} more locations\n")
                
                f.write("-"*40 + "\n\n")
    
    def quick_match_studies_for_patient(self, studies: List[StudyRecord], patient_data: Dict) -> List[StudyRecord]:
        """
        Quick filtering of studies based on patient characteristics.
        Uses simple keyword matching without LLM evaluation for speed.
        
        Args:
            studies: List of all collected studies
            patient_data: Patient information dictionary
            
        Returns:
            List of potentially relevant studies sorted by relevance score
        """
        matched_studies = []
        
        # Extract patient characteristics for matching
        diagnosis = patient_data.get('main_diagnosis_text', '').lower()
        tumor_location = patient_data.get('tumor_location', '').lower()
        patient_age = patient_data.get('age', 0)
        
        logger.info(f"Quick matching for patient with diagnosis: {diagnosis}")
        
        for study in studies:
            relevance_score = 0
            match_reasons = []
            
            # Check condition matching
            study_conditions = study.condition.lower()
            study_title = study.title.lower()
            study_summary = study.brief_summary.lower()
            
            # STRICTER MATCHING CRITERIA
            
            # Primary diagnosis matching (REQUIRED - higher threshold)
            has_net_match = any(term in study_conditions or term in study_title for term in [
                'neuroendocrine', 'carcinoid', 'net', 'pancreatic islet'
            ])
            if not has_net_match:
                continue  # Skip studies without clear NET indication
            
            relevance_score += 15
            match_reasons.append("NET condition match")
            
            # REQUIRED: Specific anatomical site matching (stricter)
            anatomical_match = False
            
            if 'pankreas' in diagnosis or 'pancrea' in diagnosis:
                if any(term in study_conditions or term in study_title or term in study_summary for term in [
                    'pancreatic', 'pancreas', 'islet', 'pnet'
                ]):
                    relevance_score += 15
                    match_reasons.append("Pancreatic NET match")
                    anatomical_match = True
            
            if 'dünndarm' in diagnosis or 'ileum' in diagnosis or 'small intestine' in diagnosis:
                if any(term in study_conditions or term in study_title or term in study_summary for term in [
                    'small intestine', 'midgut', 'intestinal', 'ileum', 'jejunum', 'duodenum'
                ]):
                    relevance_score += 15
                    match_reasons.append("Small intestine NET match")
                    anatomical_match = True
            
            # Require anatomical match for high relevance
            if not anatomical_match:
                relevance_score -= 5  # Penalize non-specific studies
            
            # Metastases pattern matching (more specific)
            if 'leber' in diagnosis or 'liver' in diagnosis:
                if any(term in study_conditions or term in study_title or term in study_summary for term in [
                    'liver', 'hepatic', 'metastases', 'metastatic'
                ]):
                    relevance_score += 8
                    match_reasons.append("Liver metastases match")
            
            # Grade-specific matching (G1/G2 vs G3 different treatments)
            if 'g1' in diagnosis or 'g2' in diagnosis:
                if any(term in study_conditions or term in study_title or term in study_summary for term in [
                    'low grade', 'well differentiated', 'g1', 'g2'
                ]):
                    relevance_score += 10
                    match_reasons.append("Low-grade NET match")
            elif 'g3' in diagnosis:
                if any(term in study_conditions or term in study_title or term in study_summary for term in [
                    'high grade', 'poorly differentiated', 'g3', 'neuroendocrine carcinoma'
                ]):
                    relevance_score += 10
                    match_reasons.append("High-grade NET match")
            
            # STRONG preference for completed Phase 2/3 studies
            if study.status == 'COMPLETED':
                if study.phase in ['PHASE2', 'PHASE3']:
                    relevance_score += 12
                    match_reasons.append(f"Completed {study.phase} study")
                else:
                    relevance_score += 5
                    match_reasons.append("Completed study")
            elif study.status == 'ACTIVE_NOT_RECRUITING':
                relevance_score += 3
                match_reasons.append("Active study")
            else:
                relevance_score -= 3  # Penalize terminated/withdrawn studies
            
            # Strong bonus for multiple publications (evidence quality)
            if study.publications_count >= 5:
                relevance_score += 8
                match_reasons.append(f"{study.publications_count} publications")
            elif study.publications_count >= 3:
                relevance_score += 5
                match_reasons.append(f"{study.publications_count} publications")
            elif study.publications_count >= 1:
                relevance_score += 2
                match_reasons.append(f"{study.publications_count} publications")
            
            # LOCATION PROXIMITY SCORING (NEW FEATURE)
            location_score, location_desc = self.location_scorer.score_location_proximity(study.locations)
            if location_score > 0:
                relevance_score += location_score
                match_reasons.append(f"Location: {location_desc}")
            
            # MUCH HIGHER threshold for inclusion (only high-quality matches)
            if relevance_score >= self.min_relevance_score:
                # Add match info to study record copy
                study_copy = StudyRecord(
                    nct_id=study.nct_id,
                    title=study.title,
                    status=study.status,
                    phase=study.phase,
                    condition=study.condition,
                    intervention=study.intervention,
                    brief_summary=study.brief_summary,
                    detailed_description=study.detailed_description,
                    eligibility_criteria=study.eligibility_criteria,
                    start_date=study.start_date,
                    completion_date=study.completion_date,
                    primary_outcome=study.primary_outcome,
                    secondary_outcome=study.secondary_outcome,
                    sponsor=study.sponsor,
                    locations=study.locations,
                    url=study.url,
                    has_posted_results=study.has_posted_results,
                    has_publications=study.has_publications,
                    publications_count=study.publications_count,
                    search_terms_matched=[f"Relevance: {relevance_score} - {'; '.join(match_reasons)}"]
                )
                matched_studies.append(study_copy)
        
        # Sort by relevance score (highest first) and limit to fewer, higher-quality matches
        matched_studies.sort(key=lambda x: int(x.search_terms_matched[0].split(':')[1].split(' -')[0]), reverse=True)
        
        logger.info(f"Found {len(matched_studies)} high-relevance studies (strict criteria)")
        return matched_studies[:self.max_matches_per_patient]  # Return top matches based on strictness setting
    
    def batch_match_patients(self, studies: List[StudyRecord], patient_data_file: str) -> Dict:
        """
        Quickly match all patients against the study database.
        
        Args:
            studies: List of all collected studies
            patient_data_file: Path to patient data Excel file
            
        Returns:
            Dictionary with patient matches
        """
        from data_loader import load_patient_data
        
        # Load patient data
        df_patients = load_patient_data(patient_data_file)
        if df_patients is None or df_patients.empty:
            logger.error("No patient data loaded")
            return {}
        
        all_matches = {}
        patient_ids = [pid for pid in df_patients["ID"].unique() if pid and str(pid).strip()]
        
        logger.info(f"Quick matching {len(patient_ids)} patients against {len(studies)} studies")
        
        for patient_id in patient_ids:
            try:
                # Get patient data
                patient_row = df_patients[df_patients["ID"] == patient_id].iloc[0]
                patient_data = patient_row.to_dict()
                
                # Quick match studies
                matches = self.quick_match_studies_for_patient(studies, patient_data)
                
                all_matches[str(patient_id)] = {
                    'patient_data': patient_data,
                    'matches': matches,
                    'match_count': len(matches)
                }
                
                logger.info(f"Patient {patient_id}: {len(matches)} relevant studies found")
                
            except Exception as e:
                logger.error(f"Error matching patient {patient_id}: {e}")
                continue
        
        return all_matches

    def save_patient_matches(self, all_matches: Dict, filename_prefix: str = "patient_matches"):
        """Save patient study matches to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary JSON
        summary_file = self.output_dir / f"{filename_prefix}_summary_{timestamp}.json"
        
        # Prepare summary data (without full study objects for JSON serialization)
        summary_data = {}
        for patient_id, data in all_matches.items():
            summary_data[patient_id] = {
                'patient_diagnosis': data['patient_data'].get('main_diagnosis_text', ''),
                'match_count': data['match_count'],
                'top_matches': [
                    {
                        'nct_id': match.nct_id,
                        'title': match.title,
                        'phase': match.phase,
                        'status': match.status,
                        'relevance': match.search_terms_matched[0] if match.search_terms_matched else "No relevance info"
                    }
                    for match in data['matches'][:5]  # Top 5 matches only
                ]
            }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        # Save detailed text report
        report_file = self.output_dir / f"{filename_prefix}_detailed_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("QUICK PATIENT-STUDY MATCHING REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total patients: {len(all_matches)}\n\n")
            
            for patient_id, data in all_matches.items():
                f.write(f"PATIENT {patient_id}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Diagnosis: {data['patient_data'].get('main_diagnosis_text', 'Not specified')}\n")
                f.write(f"Relevant studies found: {data['match_count']}\n\n")
                
                for i, match in enumerate(data['matches'][:10], 1):  # Top 10 matches
                    f.write(f"  {i}. {match.title}\n")
                    f.write(f"     NCT ID: {match.nct_id}\n")
                    f.write(f"     Phase: {match.phase} | Status: {match.status}\n")
                    f.write(f"     {match.search_terms_matched[0] if match.search_terms_matched else 'No relevance info'}\n")
                    f.write(f"     URL: {match.url}\n\n")
                
                f.write("\n" + "="*80 + "\n\n")
        
        logger.info(f"Patient matches saved to:")
        logger.info(f"  Summary: {summary_file}")
        logger.info(f"  Detailed: {report_file}")
    
    def collect_net_studies_with_strictness(self, strictness_level: str = "medium") -> List[StudyRecord]:
        """
        Collect NET studies with different levels of strictness.
        
        Args:
            strictness_level: "loose", "medium", "strict", or "ultra_strict"
            
        Returns:
            List of StudyRecord objects
        """
        
        if strictness_level == "loose":
            # Original query - all studies with results and publications
            query = 'AREA[ConditionSearch]("Neuroendocrine Tumour") AND (AREA[HasResults] true) AND (AREA[ReferencesModule]NOT MISSING)'
            logger.info("Using LOOSE criteria: All NET studies with results and publications")
            
        elif strictness_level == "medium":
            # Medium strictness - Phase 2/3 only, any status
            query = '''AREA[ConditionSearch]("Neuroendocrine Tumour") AND 
                       (AREA[HasResults] true) AND 
                       (AREA[ReferencesModule]NOT MISSING) AND
                       (AREA[Phase] ("PHASE2" OR "PHASE3"))'''
            logger.info("Using MEDIUM criteria: Phase 2/3 NET studies with results and publications")
            
        elif strictness_level == "strict":
            # Current strict query - Phase 2/3, completed/active only
            query = '''AREA[ConditionSearch]("Neuroendocrine Tumour") AND 
                       (AREA[HasResults] true) AND 
                       (AREA[ReferencesModule]NOT MISSING) AND
                       (AREA[Phase] ("PHASE2" OR "PHASE3")) AND
                       (AREA[OverallStatus] ("COMPLETED" OR "ACTIVE_NOT_RECRUITING"))'''
            logger.info("Using STRICT criteria: Completed/Active Phase 2/3 NET studies with results and publications")
            
        elif strictness_level == "ultra_strict":
            # Ultra strict - Phase 3 completed only, multiple publications
            query = '''AREA[ConditionSearch]("Neuroendocrine Tumour") AND 
                       (AREA[HasResults] true) AND 
                       (AREA[ReferencesModule]NOT MISSING) AND
                       (AREA[Phase] "PHASE3") AND
                       (AREA[OverallStatus] "COMPLETED")'''
            logger.info("Using ULTRA-STRICT criteria: Completed Phase 3 NET studies only")
            
        else:
            raise ValueError(f"Unknown strictness level: {strictness_level}")
        
        logger.info(f"Query: {query}")
        
        try:
            # Make API call
            url = f"{self.api.BASE_URL}/studies"
            params = {
                "format": "json",
                "pageSize": 1000,
                "query.term": query
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            studies = data.get("studies", [])
            
            logger.info(f"Found {len(studies)} studies with {strictness_level} criteria")
            
            # Process studies
            all_studies = []
            for study in studies:
                study_record = self._process_study(study, f"{strictness_level}_search")
                if study_record:
                    all_studies.append(study_record)
                    
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Error in {strictness_level} search: {e}")
            return []
        
        logger.info(f"Collected {len(all_studies)} studies using {strictness_level} criteria")
        return all_studies
    
    def set_matching_strictness(self, strictness_level: str):
        """Set the strictness level for patient matching."""
        if strictness_level == "loose":
            self.min_relevance_score = 5
            self.max_matches_per_patient = 20
        elif strictness_level == "medium":
            self.min_relevance_score = 15
            self.max_matches_per_patient = 15
        elif strictness_level == "strict":
            self.min_relevance_score = 25
            self.max_matches_per_patient = 10
        elif strictness_level == "ultra_strict":
            self.min_relevance_score = 35
            self.max_matches_per_patient = 5
        else:
            # Default to strict
            self.min_relevance_score = 25
            self.max_matches_per_patient = 10

    def analyze_study_locations(self, studies: List[StudyRecord]) -> Dict:
        """
        Analyze the geographic distribution of study locations.
        
        Args:
            studies: List of studies to analyze
            
        Returns:
            Dictionary with location analysis
        """
        location_analysis = {
            'total_studies': len(studies),
            'studies_with_locations': 0,
            'distance_distribution': {},
            'closest_studies': [],
            'countries': {},
            'cities': {}
        }
        
        if not GEOPY_AVAILABLE:
            location_analysis['error'] = "Geopy not available for location analysis"
            return location_analysis
        
        logger.info("Analyzing study locations relative to Bern, Switzerland...")
        
        for study in studies:
            if study.locations:
                location_analysis['studies_with_locations'] += 1
                
                # Analyze each location
                for location_str in study.locations[:3]:  # Limit to first 3 locations
                    # Extract country and city
                    parts = location_str.split(', ')
                    if len(parts) >= 3:
                        city = parts[1].strip()
                        country = parts[2].strip()
                        
                        # Count countries and cities
                        location_analysis['countries'][country] = location_analysis['countries'].get(country, 0) + 1
                        location_analysis['cities'][city] = location_analysis['cities'].get(city, 0) + 1
                
                # Calculate distance to closest location
                location_score, location_desc = self.location_scorer.score_location_proximity(study.locations)
                
                # Extract distance from description
                distance_km = None
                if "km)" in location_desc:
                    try:
                        distance_km = float(location_desc.split('(')[1].split('km')[0])
                    except:
                        pass
                
                if distance_km is not None:
                    # Categorize by distance
                    if distance_km <= 50:
                        category = "Very close (≤50km)"
                    elif distance_km <= 200:
                        category = "Close (≤200km)"
                    elif distance_km <= 500:
                        category = "Moderate (≤500km)"
                    elif distance_km <= 1000:
                        category = "Far (≤1000km)"
                    else:
                        category = "Very far (>1000km)"
                    
                    location_analysis['distance_distribution'][category] = \
                        location_analysis['distance_distribution'].get(category, 0) + 1
                    
                    # Track closest studies
                    location_analysis['closest_studies'].append({
                        'nct_id': study.nct_id,
                        'title': study.title[:60] + "...",
                        'distance_km': distance_km,
                        'location_desc': location_desc,
                        'phase': study.phase,
                        'status': study.status
                    })
        
        # Sort closest studies by distance
        location_analysis['closest_studies'].sort(key=lambda x: x['distance_km'])
        location_analysis['closest_studies'] = location_analysis['closest_studies'][:10]  # Top 10 closest
        
        # Sort countries and cities by frequency
        location_analysis['countries'] = dict(sorted(location_analysis['countries'].items(), 
                                                   key=lambda x: x[1], reverse=True)[:10])
        location_analysis['cities'] = dict(sorted(location_analysis['cities'].items(), 
                                                key=lambda x: x[1], reverse=True)[:10])
        
        return location_analysis
    
    def print_location_analysis(self, location_analysis: Dict):
        """Print a formatted location analysis report."""
        print("\n" + "="*80)
        print("STUDY LOCATION ANALYSIS (Relative to Bern, Switzerland)")
        print("="*80)
        
        if 'error' in location_analysis:
            print(f"❌ {location_analysis['error']}")
            return
        
        total = location_analysis['total_studies']
        with_locations = location_analysis['studies_with_locations']
        
        print(f"📍 Studies with location data: {with_locations}/{total} ({100*with_locations/total:.1f}%)")
        
        # Distance distribution
        print(f"\n🌍 Distance Distribution:")
        for category, count in location_analysis['distance_distribution'].items():
            percentage = 100 * count / with_locations if with_locations > 0 else 0
            print(f"   {category:<20}: {count:>3} studies ({percentage:>5.1f}%)")
        
        # Top countries
        print(f"\n🌎 Top Countries:")
        for country, count in list(location_analysis['countries'].items())[:5]:
            print(f"   {country:<20}: {count:>3} studies")
        
        # Top cities
        print(f"\n🏙️ Top Cities:")
        for city, count in list(location_analysis['cities'].items())[:5]:
            print(f"   {city:<20}: {count:>3} studies")
        
        # Closest studies
        print(f"\n⭐ Closest Studies to Bern:")
        for i, study in enumerate(location_analysis['closest_studies'][:5], 1):
            print(f"   {i}. {study['title']}")
            print(f"      NCT: {study['nct_id']} | {study['phase']} | {study['status']}")
            print(f"      Distance: {study['distance_km']:.0f}km")
            print()
        
        print("="*80)
        
def main():
    """Main function to collect NET studies using simple API call."""
    
    # Configuration
    OUTPUT_DIR = Path("C:/Users/pia/OneDrive - Universitaet Bern/Projects/NetTubo/netTubo/collected_studies")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Initialize API and collector
    api = ClinicalTrialsAPI(rate_limit_delay=1.0)
    collector = StudyCollector(api, OUTPUT_DIR)
    
    logger.info("Starting STRICT NET study collection (Phase 2/3, completed studies only)")
    logger.info('Query: AREA[ConditionSearch]("Neuroendocrine Tumour") AND (AREA[HasResults] true) AND (AREA[ReferencesModule]NOT MISSING) AND (AREA[Phase] ("PHASE2" OR "PHASE3")) AND (AREA[OverallStatus] ("COMPLETED" OR "ACTIVE_NOT_RECRUITING"))')
    
    # Collect studies using simple API call
    try:
        studies = collector.collect_net_studies_simple()
        
        if studies:
            # Save all collected studies
            collector.save_studies(studies, "net_studies_simple")
            
            # LOCATION ANALYSIS
            logger.info("Analyzing study locations...")
            location_analysis = collector.analyze_study_locations(studies)
            collector.print_location_analysis(location_analysis)
            
            # QUICK PATIENT MATCHING
            logger.info("Starting quick patient matching against collected studies...")
            
            # Load patient data and do quick matching
            patient_data_file = "C:/Users/pia/OneDrive - Universitaet Bern/Projects/NetTubo/netTubo/data/NET Tubo v2.xlsx"
            
            try:
                patient_matches = collector.batch_match_patients(studies, patient_data_file)
                
                if patient_matches:
                    # Save patient matches
                    collector.save_patient_matches(patient_matches)
                    
                    # Show quick summary
                    print("\n" + "-"*60)
                    print("QUICK PATIENT MATCHING RESULTS")
                    print("-"*60)
                    
                    total_patients = len(patient_matches)
                    patients_with_matches = sum(1 for data in patient_matches.values() if data['match_count'] > 0)
                    total_matches = sum(data['match_count'] for data in patient_matches.values())
                    avg_matches = total_matches / total_patients if total_patients > 0 else 0
                    
                    print(f"Patients processed: {total_patients}")
                    print(f"Patients with relevant studies: {patients_with_matches}")
                    print(f"Total study matches: {total_matches}")
                    print(f"Average matches per patient: {avg_matches:.1f}")
                    
                    # Show examples
                    print(f"\nExample matches (first 3 patients):")
                    for i, (patient_id, data) in enumerate(list(patient_matches.items())[:3]):
                        diagnosis = data['patient_data'].get('main_diagnosis_text', 'Not specified')[:50]
                        print(f"  Patient {patient_id}: {data['match_count']} studies - {diagnosis}")
                    
                    print("-"*60)
                    
                else:
                    print("No patient matches found.")
                    
            except Exception as e:
                logger.error(f"Error in patient matching: {e}")
                print("Quick patient matching failed - see logs for details")
            
            # Generate summary report
            print("\n" + "="*80)
            print("STRICT NET STUDY COLLECTION SUMMARY")
            print("="*80)
            print(f"Total studies collected: {len(studies)}")
            print('Query used: AREA[ConditionSearch]("Neuroendocrine Tumour") AND (AREA[HasResults] true) AND (AREA[ReferencesModule]NOT MISSING) AND (AREA[Phase] ("PHASE2" OR "PHASE3")) AND (AREA[OverallStatus] ("COMPLETED" OR "ACTIVE_NOT_RECRUITING"))')
            print(f"Results saved to: {OUTPUT_DIR}")
            
            # Show some statistics
            status_counts = {}
            phase_counts = {}
            results_counts = {"has_results": 0, "no_results": 0}
            
            for study in studies:
                status_counts[study.status] = status_counts.get(study.status, 0) + 1
                phase_counts[study.phase] = phase_counts.get(study.phase, 0) + 1
                if study.has_posted_results:
                    results_counts["has_results"] += 1
                else:
                    results_counts["no_results"] += 1
            
            print(f"\nStatus distribution:")
            for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {status}: {count}")
            
            print(f"\nPhase distribution:")
            for phase, count in sorted(phase_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {phase}: {count}")
                
            print(f"\nResults availability:")
            print(f"  Studies with posted results: {results_counts['has_results']}")
            print(f"  Studies without posted results: {results_counts['no_results']}")
            
            print("="*80)
            
        else:
            print("No studies found using simple API call.")
            
    except Exception as e:
        logger.error(f"Error during study collection: {e}")
        return

if __name__ == "__main__":
    main()
