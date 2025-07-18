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
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from netTubo.clinical_trials.clinical_trials_matcher import ClinicalTrialsAPI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    publications: List[Dict]  # Store full publication details
    recent_publications_count: int  # Publications from April 2020 onwards
    search_terms_matched: List[str]

class StudyCollector:
    """Collects all studies matching filter criteria without LLM evaluation."""
    
    def __init__(self, api: ClinicalTrialsAPI, output_dir: Path):
        self.api = api
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
               
    def collect_net_studies_simple(self, filter_recent_publications: bool = False) -> List[StudyRecord]:
        """
        Simple API call to collect Neuroendocrine Tumor studies with posted results.
        Uses direct API query: AREA[ConditionSearch](Neuroendocrine Tumors) AND (AREA[HasResults] true)
        
        Args:
            filter_recent_publications: If True, only return studies with publications from 2020 onwards
        
        Returns:
            List of StudyRecord objects
        """
        all_studies = []
        
        # Query for NET studies with results and publications
        query = 'AREA[ConditionSearch](Neuroendocrine Tumors) AND (AREA[HasResults] true)'
        
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
        
        # Filter by recent publications if requested
        if filter_recent_publications:
            all_studies = self.filter_studies_with_recent_publications(all_studies)
        
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
            
            # Check publications and filter by date
            has_publications = False
            publications_count = 0
            recent_publications_count = 0
            publications = []
            cutoff_date = date(2020, 1, 1)  # January 2020
            
            if references_module:
                references = references_module.get("references", [])
                publications_count = len(references)
                has_publications = publications_count > 0
                
                # Analyze each publication for date
                for ref in references:
                    pub_info = {
                        'citation': ref.get('citation', ''),
                        'pmid': ref.get('pmid', ''),
                        'type': ref.get('type', ''),
                        'date': None,
                        'is_recent': False
                    }
                    
                    # Try to extract publication date from citation
                    citation = ref.get('citation', '')
                    pub_date = self._extract_publication_date(citation)
                    
                    if pub_date:
                        pub_info['date'] = pub_date.isoformat()  # Convert to string for JSON serialization
                        if pub_date >= cutoff_date:
                            pub_info['is_recent'] = True
                            recent_publications_count += 1
                    
                    publications.append(pub_info)
            
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
                publications=publications,
                recent_publications_count=recent_publications_count,
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
            
    def _extract_publication_date(self, citation: str) -> Optional[date]:
        """
        Extract publication date from citation string.
        
        Args:
            citation: Citation string from the API
            
        Returns:
            Date object if found, None otherwise
        """
        if not citation:
            return None
        
        # Common date patterns in citations
        date_patterns = [
            r'(\d{4})\s+(\w{3})\s+(\d{1,2})',  # 2021 Jan 15
            r'(\d{4})\s+(\w{3,9})\s+(\d{1,2})',  # 2021 January 15
            r'(\d{4})\s+(\w{3})',  # 2021 Jan
            r'(\d{4})\s+(\w{3,9})',  # 2021 January
            r'\b(\d{4})\b',  # Just year: 2021
        ]
        
        month_map = {
            'jan': 1, 'january': 1,
            'feb': 2, 'february': 2,
            'mar': 3, 'march': 3,
            'apr': 4, 'april': 4,
            'may': 5,
            'jun': 6, 'june': 6,
            'jul': 7, 'july': 7,
            'aug': 8, 'august': 8,
            'sep': 9, 'september': 9,
            'oct': 10, 'october': 10,
            'nov': 11, 'november': 11,
            'dec': 12, 'december': 12
        }
        
        for pattern in date_patterns:
            match = re.search(pattern, citation, re.IGNORECASE)
            if match:
                try:
                    year = int(match.group(1))
                    
                    if len(match.groups()) >= 2:
                        month_str = match.group(2).lower()
                        month = month_map.get(month_str, 1)
                    else:
                        month = 1
                    
                    if len(match.groups()) >= 3:
                        day = int(match.group(3))
                    else:
                        day = 1
                    
                    return date(year, month, day)
                    
                except (ValueError, IndexError):
                    continue
        
        return None

    def filter_studies_with_recent_publications(self, studies: List[StudyRecord]) -> List[StudyRecord]:
        """
        Filter studies to only include those with publications from 2020 onwards.
        
        Args:
            studies: List of all studies
            
        Returns:
            List of studies with recent publications
        """
        recent_studies = []
        
        for study in studies:
            if study.recent_publications_count > 0:
                recent_studies.append(study)
                logger.debug(f"Study {study.nct_id} has {study.recent_publications_count} recent publications")
        
        logger.info(f"Filtered to {len(recent_studies)} studies with publications from 2020 onwards (from {len(studies)} total)")
        return recent_studies

    
def main():
    """Main function to collect NET studies using simple API call."""
    
    # Configuration
    OUTPUT_DIR = Path("C:/Users/pia/OneDrive - Universitaet Bern/Projects/NetTubo/netTubo/collected_studies")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Initialize API and collector
    api = ClinicalTrialsAPI(rate_limit_delay=1.0)
    collector = StudyCollector(api, OUTPUT_DIR)
    
    logger.info("Starting NET study collection with publication date filtering")
    logger.info('Query: AREA[ConditionSearch](Neuroendocrine Tumors) AND (AREA[HasResults] true)')
    
    # Collect studies using simple API call
    try:
        # First, collect all studies with the specified query
        all_studies = collector.collect_net_studies_simple(filter_recent_publications=False)
        
        if all_studies:
            logger.info(f"Collected {len(all_studies)} total studies")
            
            # Filter for studies with publications newer than 2020
            recent_studies = collector.filter_studies_with_recent_publications(all_studies)
            
            if recent_studies:
                # Save only the filtered studies with recent publications
                collector.save_studies(recent_studies, "net_studies_recent_publications")
                
                print(f"\n" + "="*80)
                print("STUDY COLLECTION AND FILTERING RESULTS")
                print("="*80)
                print(f"Total studies found: {len(all_studies)}")
                print(f"Studies with publications from 2020 onwards: {len(recent_studies)}")
                
                # Show some examples
                print(f"\nExamples of studies with recent publications:")
                for i, study in enumerate(recent_studies[:5]):
                    recent_pubs = [pub for pub in study.publications if pub.get('is_recent', False)]
                    recent_dates = [pub.get('date') for pub in recent_pubs if pub.get('date')]
                    print(f"  {i+1}. {study.nct_id}: {study.recent_publications_count} recent publications")
                    if recent_dates:
                        print(f"     Recent dates: {', '.join(str(d) for d in recent_dates[:3])}")
                
                # Use filtered studies for patient matching
                studies = recent_studies
            else:
                print("No studies found with publications from 2020 onwards")
                return
                
            # QUICK PATIENT MATCHING
            logger.info("Starting quick patient matching against collected studies...")
            
            # Load patient data and do quick matching
            patient_data_file = "C:/Users/pia/OneDrive - Universitaet Bern/Projects/NetTubo/netTubo/data/NET Tubo v2.xlsx"
            
            try:
                patient_matches = collector.batch_match_patients(studies, patient_data_file)
                
                if patient_matches:

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
            print("NET STUDY COLLECTION SUMMARY")
            print("="*80)
            print(f"Total studies collected: {len(studies)}")
            print('Query used: AREA[ConditionSearch](Neuroendocrine Tumors) AND (AREA[HasResults] true)')
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
            print("No studies found matching the criteria.")
            
    except Exception as e:
        logger.error(f"Error during study collection: {e}")
        return

if __name__ == "__main__":
    main()
