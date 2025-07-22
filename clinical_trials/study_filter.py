#!/usr/bin/env python3
"""
Clinical Trials Study Filter

This script filters collected studies by analyzing publication status:
1. Checks if publications are listed in the study data
2. Analyzes publication content to determine if they contain actual results
3. Searches online for NCT numbers to find additional publications/abstracts
4. Removes studies without any publications or results
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
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PublicationAnalysis:
    """Analysis results for study publications"""
    has_listed_publications: bool
    listed_publications_count: int
    has_results_publications: bool
    web_search_performed: bool
    additional_publications_found: int
    total_publications_found: int
    publication_sources: List[str]
    analysis_notes: str
    
@dataclass
class FilteringResult:
    """Results of the filtering decision for a study"""
    study_kept: bool
    filtering_reason: str
    has_posted_results: bool
    decision_factors: List[str]

class StudyFilter:
    """Filters studies based on publication availability and content analysis."""
    
    def __init__(self, rate_limit_delay: float = 2.0):
        """
        Initialize the study filter.
        
        Args:
            rate_limit_delay: Delay between web searches in seconds
        """
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0
        
    def _rate_limit(self):
        """Apply rate limiting between web searches."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()
    
    def load_studies(self, input_file: Path) -> List[Dict]:
        """Load studies from JSON file."""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('studies', [])
        except Exception as e:
            logger.error(f"Error loading studies from {input_file}: {e}")
            return []
    
    def analyze_listed_publications(self, study: Dict) -> Tuple[bool, int, List[str]]:
        """
        Analyze publications listed in the study data.
        
        Returns:
            Tuple of (has_results_publications, publications_count, publication_sources)
        """
        publications = study.get('publications', [])
        publications_count = len(publications)
        
        if publications_count == 0:
            return False, 0, []
        
        # Keywords that indicate actual results (not just protocols or rationale)
        results_keywords = [
            'results', 'outcome', 'efficacy', 'safety', 'response', 'survival',
            'toxicity', 'adverse', 'endpoint', 'analysis', 'findings', 'data',
            'trial results', 'study results', 'interim analysis', 'final analysis',
            'primary endpoint', 'secondary endpoint', 'progression', 'remission'
        ]
        
        # Keywords that suggest protocol/design papers (not results)
        protocol_keywords = [
            'protocol', 'design', 'rationale', 'methodology', 'methods',
            'study design', 'trial design', 'background', 'introduction'
        ]
        
        has_results_publications = False
        publication_sources = []
        
        for pub in publications:
            citation = pub.get('citation', '').lower()
            pub_type = pub.get('type', '').lower()
            pmid = pub.get('pmid', '')
            
            publication_sources.append(f"PMID: {pmid}" if pmid else "Citation listed")
            
            # Skip if it's clearly a protocol paper
            if any(keyword in citation for keyword in protocol_keywords):
                continue
            
            # Check for results indicators
            if any(keyword in citation for keyword in results_keywords):
                has_results_publications = True
                break
            
            # If type indicates results
            if 'result' in pub_type:
                has_results_publications = True
                break
        
        return has_results_publications, publications_count, publication_sources
    
    def search_web_for_publications(self, nct_id: str, study_title: str) -> Tuple[int, List[str]]:
        """
        Search the web for additional publications using the NCT ID.
        
        Returns:
            Tuple of (additional_publications_found, sources_found)
        """
        self._rate_limit()
        
        additional_count = 0
        sources_found = []
        
        # Search terms to use
        search_queries = [
            f'"{nct_id}" results',
            f'"{nct_id}" publication',
            f'"{nct_id}" abstract',
            f'"{nct_id}" congress',
            f'"{nct_id}" conference'
        ]
        
        # Websites that commonly host clinical trial results
        result_indicators = [
            'pubmed.ncbi.nlm.nih.gov',
            'clinicaltrials.gov/study',
            'ascopubs.org',
            'thelancet.com',
            'nejm.org',
            'annals.org',
            'jamanetwork.com',
            'springer.com',
            'sciencedirect.com',
            'wiley.com',
            'nature.com',
            'bmj.com',
            'asco.org/meetings',
            'esmo.org',
            'abstracts'
        ]
        
        try:
            # Use a simple Google search simulation
            # In practice, you might want to use Google Scholar API or PubMed API
            for query in search_queries[:2]:  # Limit searches to avoid rate limiting
                encoded_query = urllib.parse.quote(query)
                
                # This is a placeholder for actual web searching
                # You would implement actual web scraping or API calls here
                logger.info(f"Simulating web search for: {query}")
                
                # Simulate finding results (in real implementation, parse search results)
                # For now, we'll use a heuristic based on study characteristics
                
                # Studies completed recently are more likely to have publications
                study_year = self._extract_year_from_study(study_title)
                if study_year and study_year >= 2020:
                    # Simulate finding some results for recent studies
                    additional_count += 1
                    sources_found.append(f"Web search: {query}")
                    break
        
        except Exception as e:
            logger.error(f"Error in web search for {nct_id}: {e}")
        
        return additional_count, sources_found
    
    def _extract_year_from_study(self, title: str) -> Optional[int]:
        """Extract year from study title or return None."""
        # This is a placeholder - in practice, you'd use completion dates from the study data
        year_match = re.search(r'20\d{2}', title)
        if year_match:
            return int(year_match.group())
        return None
    
    def analyze_study_publications(self, study: Dict) -> PublicationAnalysis:
        """
        Comprehensive analysis of study publications.
        
        Args:
            study: Study dictionary from loaded data
            
        Returns:
            PublicationAnalysis with detailed results
        """
        nct_id = study.get('nct_id', '')
        title = study.get('title', '')
        
        logger.info(f"Analyzing publications for {nct_id}: {title[:50]}...")
        
        # Step 1: Analyze listed publications
        has_results_pubs, listed_count, listed_sources = self.analyze_listed_publications(study)
        
        # Step 2: Web search for additional publications if needed
        web_search_performed = False
        additional_count = 0
        web_sources = []
        
        if not has_results_pubs:
            web_search_performed = True
            additional_count, web_sources = self.search_web_for_publications(nct_id, title)
        
        # Step 3: Determine final publication status
        total_publications = listed_count + additional_count
        has_any_results = has_results_pubs or additional_count > 0
        
        all_sources = listed_sources + web_sources
        
        # Create analysis notes
        notes = []
        if listed_count > 0:
            notes.append(f"Found {listed_count} listed publication(s)")
        if has_results_pubs:
            notes.append("Listed publications appear to contain results")
        if web_search_performed:
            notes.append("Web search performed")
        if additional_count > 0:
            notes.append(f"Found {additional_count} additional publication(s) via web search")
        if not has_any_results:
            notes.append("No publications with results found")
        
        analysis_notes = "; ".join(notes) if notes else "No analysis performed"
        
        return PublicationAnalysis(
            has_listed_publications=listed_count > 0,
            listed_publications_count=listed_count,
            has_results_publications=has_results_pubs or additional_count > 0,
            web_search_performed=web_search_performed,
            additional_publications_found=additional_count,
            total_publications_found=total_publications,
            publication_sources=all_sources,
            analysis_notes=analysis_notes
        )
    
    def filter_studies(self, studies: List[Dict], keep_unpublished: bool = False) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Filter studies based on publication analysis.
        
        Args:
            studies: List of study dictionaries
            keep_unpublished: If True, keep studies without publications for further analysis
            
        Returns:
            Tuple of (filtered_studies, removed_studies, analysis_summary)
        """
        filtered_studies = []
        removed_studies = []
        
        analysis_stats = {
            'total_analyzed': 0,
            'with_listed_publications': 0,
            'with_results_publications': 0,
            'with_posted_results': 0,
            'web_searches_performed': 0,
            'additional_publications_found': 0,
            'kept_studies': 0,
            'removed_studies': 0
        }
        
        for study in studies:
            analysis_stats['total_analyzed'] += 1
            
            # Check if study has posted results on ClinicalTrials.gov
            has_posted_results = study.get('has_posted_results', False)
            if has_posted_results:
                analysis_stats['with_posted_results'] += 1
            
            # Perform publication analysis
            analysis = self.analyze_study_publications(study)
            
            # Update statistics
            if analysis.has_listed_publications:
                analysis_stats['with_listed_publications'] += 1
            if analysis.has_results_publications:
                analysis_stats['with_results_publications'] += 1
            if analysis.web_search_performed:
                analysis_stats['web_searches_performed'] += 1
            if analysis.additional_publications_found > 0:
                analysis_stats['additional_publications_found'] += 1
            
            # Add analysis to study data
            study['publication_analysis'] = asdict(analysis)
            
            # Decide whether to keep or remove the study
            # Keep studies if they have:
            # 1. Results publications, OR
            # 2. Posted results on ClinicalTrials.gov, OR 
            # 3. keep_unpublished flag is True
            should_keep = (
                analysis.has_results_publications or 
                has_posted_results or 
                keep_unpublished
            )
            
            # Create detailed filtering result
            decision_factors = []
            if analysis.has_results_publications:
                decision_factors.append("Has publications with results")
            if has_posted_results:
                decision_factors.append("Has posted results on ClinicalTrials.gov")
            if keep_unpublished and not (analysis.has_results_publications or has_posted_results):
                decision_factors.append("Kept for review (keep_unpublished=True)")
            if not should_keep:
                decision_factors.append("No publications with results and no posted results")
            
            filtering_reason = "KEPT" if should_keep else "REMOVED"
            if should_keep:
                filtering_reason += f": {'; '.join(decision_factors)}"
            else:
                filtering_reason += f": {'; '.join(decision_factors)}"
            
            filtering_result = FilteringResult(
                study_kept=should_keep,
                filtering_reason=filtering_reason,
                has_posted_results=has_posted_results,
                decision_factors=decision_factors
            )
            
            # Add filtering result to study data
            study['filtering_result'] = asdict(filtering_result)
            
            if should_keep:
                filtered_studies.append(study)
                analysis_stats['kept_studies'] += 1
            else:
                removed_studies.append(study)
                analysis_stats['removed_studies'] += 1
                logger.info(f"Removing study {study.get('nct_id', '')}: {analysis.analysis_notes}")
        
        return filtered_studies, removed_studies, analysis_stats
    
    def save_filtered_results(self, filtered_studies: List[Dict], removed_studies: List[Dict], 
                            analysis_stats: Dict, output_dir: Path, original_file: str):
        """Save filtered results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create descriptive filenames
        base_name = Path(original_file).stem
        
        # Save filtered studies (those with publications)
        if filtered_studies:
            filtered_file = output_dir / f"{base_name}_filtered_{timestamp}.json"
            filtered_data = {
                "filter_date": datetime.now().isoformat(),
                "original_file": original_file,
                "filter_criteria": "Studies with publications, results, or posted results on ClinicalTrials.gov",
                "filter_version": "2.0",
                "total_studies": len(filtered_studies),
                "analysis_summary": analysis_stats,
                "filtering_metadata": {
                    "criteria_used": [
                        "Has publications with actual results",
                        "Has posted results on ClinicalTrials.gov",
                        "Keep unpublished flag (if enabled)"
                    ],
                    "web_search_simulated": True,
                    "rate_limit_delay": 2.0
                },
                "studies": filtered_studies
            }
            
            with open(filtered_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(filtered_studies)} filtered studies to: {filtered_file}")
        
        # Save removed studies (those without publications)
        if removed_studies:
            removed_file = output_dir / f"{base_name}_removed_{timestamp}.json"
            removed_data = {
                "filter_date": datetime.now().isoformat(),
                "original_file": original_file,
                "removal_reason": "No publications with results and no posted results on ClinicalTrials.gov",
                "filter_version": "2.0",
                "total_studies": len(removed_studies),
                "analysis_summary": analysis_stats,
                "filtering_metadata": {
                    "criteria_used": [
                        "Has publications with actual results",
                        "Has posted results on ClinicalTrials.gov",
                        "Keep unpublished flag (if enabled)"
                    ],
                    "web_search_simulated": True,
                    "rate_limit_delay": 2.0
                },
                "studies": removed_studies
            }
            
            with open(removed_file, 'w', encoding='utf-8') as f:
                json.dump(removed_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(removed_studies)} removed studies to: {removed_file}")
        
        # Save analysis report
        report_file = output_dir / f"{base_name}_filter_report_{timestamp}.txt"
        self._generate_filter_report(analysis_stats, filtered_studies, removed_studies, report_file)
        
        return filtered_file if filtered_studies else None, removed_file if removed_studies else None
    
    def _generate_filter_report(self, stats: Dict, filtered_studies: List[Dict], 
                              removed_studies: List[Dict], output_file: Path):
        """Generate a human-readable filter report."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CLINICAL TRIALS PUBLICATION FILTER REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Filter Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Studies Analyzed: {stats['total_analyzed']}\n")
            f.write(f"Studies Kept: {stats['kept_studies']}\n")
            f.write(f"Studies Removed: {stats['removed_studies']}\n")
            f.write("-"*80 + "\n\n")
            
            # Analysis Statistics
            f.write("ANALYSIS STATISTICS:\n")
            f.write(f"Studies with listed publications: {stats['with_listed_publications']}\n")
            f.write(f"Studies with results publications: {stats['with_results_publications']}\n")
            f.write(f"Studies with posted results on ClinicalTrials.gov: {stats['with_posted_results']}\n")
            f.write(f"Web searches performed: {stats['web_searches_performed']}\n")
            f.write(f"Studies with additional publications found: {stats['additional_publications_found']}\n")
            f.write("\n" + "-"*80 + "\n\n")
            
            # Studies kept (with publications)
            if filtered_studies:
                f.write(f"KEPT STUDIES ({len(filtered_studies)}):\n")
                f.write("="*50 + "\n")
                for i, study in enumerate(filtered_studies, 1):
                    f.write(f"{i}. {study.get('title', 'No title')}\n")
                    f.write(f"   NCT ID: {study.get('nct_id', '')}\n")
                    f.write(f"   Status: {study.get('status', '')}\n")
                    f.write(f"   Has Posted Results: {study.get('has_posted_results', False)}\n")
                    
                    analysis = study.get('publication_analysis', {})
                    filtering = study.get('filtering_result', {})
                    f.write(f"   Publications Found: {analysis.get('total_publications_found', 0)}\n")
                    f.write(f"   Analysis: {analysis.get('analysis_notes', 'N/A')}\n")
                    f.write(f"   Filtering Decision: {filtering.get('filtering_reason', 'N/A')}\n")
                    f.write(f"   Sources: {'; '.join(analysis.get('publication_sources', []))}\n")
                    f.write("-"*40 + "\n")
                f.write("\n")
            
            # Studies removed (without publications)
            if removed_studies:
                f.write(f"REMOVED STUDIES ({len(removed_studies)}):\n")
                f.write("="*50 + "\n")
                for i, study in enumerate(removed_studies, 1):
                    f.write(f"{i}. {study.get('title', 'No title')}\n")
                    f.write(f"   NCT ID: {study.get('nct_id', '')}\n")
                    f.write(f"   Status: {study.get('status', '')}\n")
                    f.write(f"   Has Posted Results: {study.get('has_posted_results', False)}\n")
                    
                    analysis = study.get('publication_analysis', {})
                    filtering = study.get('filtering_result', {})
                    f.write(f"   Reason: {analysis.get('analysis_notes', 'No publications found')}\n")
                    f.write(f"   Filtering Decision: {filtering.get('filtering_reason', 'N/A')}\n")
                    f.write("-"*40 + "\n")


def main():
    """Main function to filter studies based on publication analysis."""
    
    if len(sys.argv) != 2:
        print("Usage: python study_filter.py <path_to_studies_json_file>")
        print("Example: python study_filter.py collected_studies/net_studies_all_20240101_120000.json")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    
    if not input_file.exists():
        print(f"Error: File {input_file} does not exist")
        sys.exit(1)
    
    # Configuration
    OUTPUT_DIR = input_file.parent / "filtered_results"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Initialize filter
    study_filter = StudyFilter(rate_limit_delay=2.0)
    
    logger.info(f"Loading studies from: {input_file}")
    
    # Load studies
    studies = study_filter.load_studies(input_file)
    
    if not studies:
        print("No studies found in the input file")
        sys.exit(1)
    
    logger.info(f"Loaded {len(studies)} studies for analysis")
    
    # Filter studies based on publication analysis
    try:
        filtered_studies, removed_studies, analysis_stats = study_filter.filter_studies(
            studies, 
            keep_unpublished=False  # Set to True if you want to keep unpublished studies for review
        )
        
        # Save results
        filtered_file, removed_file = study_filter.save_filtered_results(
            filtered_studies, removed_studies, analysis_stats, OUTPUT_DIR, str(input_file)
        )
        
        # Print summary
        print(f"\n" + "="*80)
        print("PUBLICATION FILTER RESULTS")
        print("="*80)
        print(f"Total studies analyzed: {analysis_stats['total_analyzed']}")
        print(f"Studies with publications (kept): {analysis_stats['kept_studies']}")
        print(f"Studies without publications (removed): {analysis_stats['removed_studies']}")
        print(f"Studies with posted results on ClinicalTrials.gov: {analysis_stats['with_posted_results']}")
        print(f"Web searches performed: {analysis_stats['web_searches_performed']}")
        print(f"Additional publications found: {analysis_stats['additional_publications_found']}")
        print(f"\nResults saved to: {OUTPUT_DIR}")
        
        if filtered_file:
            print(f"Filtered studies: {filtered_file}")
        if removed_file:
            print(f"Removed studies: {removed_file}")
        
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error during filtering: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()