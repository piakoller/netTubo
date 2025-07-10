#!/usr/bin/env python3
"""
Report generation for clinical trials matching.

This module handles the generation of formatted reports and summaries.
"""

from datetime import datetime
from typing import Dict, List

from .models import ClinicalTrialMatch


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
        if match.publications and match.publications.get('publications_found', 0) > 0:
            report.append(f"   Publications Found: {match.publications['publications_found']}")
            for i, pub in enumerate(match.publications['publications'][:2], 1):  # Show first 2 publications
                report.append(f"     {i}. {pub['title']} ({pub['year']})")
                report.append(f"        PMID: {pub['pmid']} | Journal: {pub['journal']}")
        elif match.publications:
            report.append(f"   Publications: No publications found")
        
        report.append("-"*40)
    
    return "\n".join(report)


def generate_summary_report(all_results: List[Dict], output_dir, summary_file, openrouter_model: str) -> str:
    """Generate an overall summary report."""
    total_matches = sum(result["matches_found"] for result in all_results)
    avg_matches = total_matches / len(all_results) if all_results else 0
    
    # Calculate score distribution and publication statistics
    all_scores = []
    total_publications = 0
    studies_with_publications = 0
    
    for result in all_results:
        for match in result["matches"]:
            all_scores.append(match["relevance_score"])
            if match.get("publications") and match["publications"].get("publications_found", 0) > 0:
                total_publications += match["publications"]["publications_found"]
                studies_with_publications += 1
    
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
        f"Studies with publications: {studies_with_publications}/{total_matches}",
        f"Total publications downloaded: {total_publications}",
        f"Publications directory: {output_dir}/publications",
        f"Evaluation method: LLM-based semantic matching ({openrouter_model})",
        f"Results saved to: {output_dir}",
        f"Summary data: {summary_file}",
        "="*80,
        "Note: Studies were evaluated using AI to assess relevance",
        "based on patient diagnosis and clinical context."
    ]
    
    return "\n".join(summary_report)
