#!/usr/bin/env python3
"""
Publication Summarizer for Therapy Recommendations

This script extracts all publications from clinical trials matches,
retrieves full-text content, and creates LLM-generated summaries
focused on therapy recommendations for each patient.
"""

import json
import logging
import os
import requests
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import OPENROUTER_API_KEY, OPENROUTER_API_URL, OPENROUTER_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PublicationSummary:
    """Represents a summarized publication for therapy recommendations."""
    pmid: str
    title: str
    citation: str
    study_nct_id: str
    study_title: str
    therapy_relevance_summary: str
    key_findings: str
    therapy_recommendations: str
    patient_population: str
    contraindications: str
    evidence_quality: str
    publication_date: str
    summary_date: str

class PublicationSummarizer:
    """Summarizes publications for therapy recommendations using LLM."""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.model = model or OPENROUTER_MODEL
        self.base_url = OPENROUTER_API_URL
        
        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY not set. Please set it as an environment variable.")
    
    def call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Call the LLM API with error handling and retries."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/piakoller/netTubo",
            "X-Title": "NET Publication Summarizer",
            "Content-Type": "application/json"
        }
        
        # Use a model that supports system messages, or fallback to user-only format
        system_content = "You are a medical expert specializing in neuroendocrine tumors (NET) and analyzing scientific publications for therapy recommendations. Your task is to analyze publications and create precise, therapy-relevant summaries."
        
        # Try with system message first (for models that support it)
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system", 
                    "content": system_content
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.0,
            "max_tokens": 2000
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        return result['choices'][0]['message']['content'].strip()
                    else:
                        logger.warning(f"Unexpected API response structure: {result}")
                        return "ERROR: Unexpected API response structure"
                else:
                    logger.warning(f"API returned status {response.status_code}: {response.text}")
                    
                    # If we get a 400 error about system messages, try without system message
                    if response.status_code == 400 and "Developer instruction" in response.text:
                        logger.info("Model doesn't support system messages, retrying with combined prompt...")
                        # Combine system message with user prompt
                        combined_prompt = f"{system_content}\n\n{prompt}"
                        data_fallback = {
                            "model": self.model,
                            "messages": [
                                {
                                    "role": "user", 
                                    "content": combined_prompt
                                }
                            ],
                            "temperature": 0.0,
                            "max_tokens": 2000
                        }
                        
                        response = requests.post(
                            self.base_url,
                            headers=headers,
                            json=data_fallback,
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            if 'choices' in result and len(result['choices']) > 0:
                                return result['choices'][0]['message']['content'].strip()
                        else:
                            logger.warning(f"Fallback also failed with status {response.status_code}: {response.text}")
                    
                    if response.status_code == 429:  # Rate limit
                        wait_time = 30 * (attempt + 1)
                        logger.info(f"Rate limit hit, waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                
            except Exception as e:
                logger.error(f"LLM API request failed (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
        
        return "ERROR: LLM API call failed after all retries"
    
    def fetch_pubmed_abstract(self, pmid: str) -> Optional[Dict]:
        """Fetch abstract and metadata from PubMed."""
        try:
            # Use PubMed eUtils API to fetch abstract
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            params = {
                "db": "pubmed",
                "id": pmid,
                "retmode": "xml",
                "rettype": "abstract"
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML to extract abstract and metadata
            xml_content = response.text
            
            # Simple regex extraction (for more robust parsing, use xml.etree.ElementTree)
            title_match = re.search(r'<ArticleTitle>(.*?)</ArticleTitle>', xml_content, re.DOTALL)
            abstract_match = re.search(r'<AbstractText[^>]*>(.*?)</AbstractText>', xml_content, re.DOTALL)
            
            title = title_match.group(1).strip() if title_match else "Title not available"
            abstract = abstract_match.group(1).strip() if abstract_match else "Abstract not available"
            
            # Clean up HTML entities
            title = re.sub(r'<[^>]+>', '', title)
            abstract = re.sub(r'<[^>]+>', '', abstract)
            
            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "full_content": f"Title: {title}\n\nAbstract: {abstract}"
            }
            
        except Exception as e:
            logger.error(f"Error fetching PubMed abstract for PMID {pmid}: {e}")
            return None
    
    def create_therapy_summary_prompt(self, publication_data: Dict, patient_data: Dict, study_context: Dict) -> str:
        """Create a prompt for therapy-focused publication summary."""
        
        patient_diagnosis = patient_data.get('main_diagnosis_text', 'NET diagnosis not specified')
        patient_question = patient_data.get('Fragestellung', 'Therapy guidance needed')
        
        prompt = f"""
Analyze the following scientific publication in the context of therapy recommendations for a NET patient.

PATIENT CONTEXT:
Diagnosis: {patient_diagnosis}
Clinical Question: {patient_question}

STUDY CONTEXT:
NCT ID: {study_context.get('nct_id', 'N/A')}
Study Title: {study_context.get('title', 'N/A')}
Phase: {study_context.get('phase', 'N/A')}
Intervention: {study_context.get('intervention', 'N/A')}

PUBLICATION:
PMID: {publication_data.get('pmid', 'N/A')}
Title: {publication_data.get('title', 'N/A')}
Full Content/Abstract: {publication_data.get('full_content', 'N/A')}

TASK:
Create a structured summary focused on therapy recommendations. Analyze:

1. THERAPY RELEVANCE: How relevant is this publication for treating NET patients?
2. KEY FINDINGS: What are the most important therapeutic insights?
3. THERAPY RECOMMENDATIONS: What specific treatment approaches are recommended or studied?
4. PATIENT POPULATION: For which patients are these findings relevant?
5. CONTRAINDICATIONS: Which patients should NOT receive this therapy?
6. EVIDENCE QUALITY: How strong is the evidence (Phase I/II/III, patient numbers, etc.)?

Respond in the following structured format:

THERAPY_RELEVANCE_SUMMARY: [2-3 sentences about relevance for NET therapy]

KEY_FINDINGS: [Most important therapeutic insights, 3-4 bullet points]

THERAPY_RECOMMENDATIONS: [Specific treatment recommendations based on results]

PATIENT_POPULATION: [Description of suitable patients]

CONTRAINDICATIONS: [Description of unsuitable patients or precautions]

EVIDENCE_QUALITY: [Assessment of study quality and evidence strength]

Stick strictly to this format and write in English.
"""
        return prompt
    
    def summarize_publication(self, publication: Dict, patient_data: Dict, study_context: Dict) -> Optional[PublicationSummary]:
        """Summarize a single publication for therapy recommendations."""
        
        pmid = publication.get('pmid', '')
        if not pmid:
            logger.warning("No PMID found for publication")
            return None
        
        # Fetch full abstract from PubMed
        pubmed_data = self.fetch_pubmed_abstract(pmid)
        if not pubmed_data:
            logger.warning(f"Could not fetch PubMed data for PMID {pmid}")
            # Fallback to citation
            pubmed_data = {
                "pmid": pmid,
                "title": "Title not available",
                "abstract": "Abstract not available",
                "full_content": publication.get('citation', 'No content available')
            }
        
        # Create therapy-focused summary prompt
        prompt = self.create_therapy_summary_prompt(pubmed_data, patient_data, study_context)
        
        # Get LLM summary
        logger.info(f"Generating therapy summary for PMID {pmid}...")
        llm_response = self.call_llm(prompt)
        
        if llm_response.startswith("ERROR:"):
            logger.error(f"LLM summarization failed for PMID {pmid}: {llm_response}")
            return None
        
        # Parse structured response
        parsed_summary = self._parse_llm_response(llm_response)
        
        # Extract publication date from citation
        pub_date = self._extract_publication_date(publication.get('citation', ''))
        
        return PublicationSummary(
            pmid=pmid,
            title=pubmed_data.get('title', 'Title not available'),
            citation=publication.get('citation', ''),
            study_nct_id=study_context.get('nct_id', ''),
            study_title=study_context.get('title', ''),
            therapy_relevance_summary=parsed_summary.get('therapierelevanz_zusammenfassung', ''),
            key_findings=parsed_summary.get('kernbefunde', ''),
            therapy_recommendations=parsed_summary.get('therapieempfehlungen', ''),
            patient_population=parsed_summary.get('patientenpopulation', ''),
            contraindications=parsed_summary.get('kontraindikationen', ''),
            evidence_quality=parsed_summary.get('evidenzqualität', ''),
            publication_date=pub_date,
            summary_date=datetime.now().isoformat()
        )
    
    def _parse_llm_response(self, response: str) -> Dict[str, str]:
        """Parse structured LLM response into components."""
        parsed = {}
        
        sections = [
            ('therapierelevanz_zusammenfassung', 'THERAPY_RELEVANCE_SUMMARY:'),
            ('kernbefunde', 'KEY_FINDINGS:'),
            ('therapieempfehlungen', 'THERAPY_RECOMMENDATIONS:'),
            ('patientenpopulation', 'PATIENT_POPULATION:'),
            ('kontraindikationen', 'CONTRAINDICATIONS:'),
            ('evidenzqualität', 'EVIDENCE_QUALITY:')
        ]
        
        for key, marker in sections:
            pattern = f"{re.escape(marker)}\\s*(.*?)(?=\\n[A-Z_]+:|$)"
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                parsed[key] = match.group(1).strip()
            else:
                parsed[key] = "Information not available"
        
        return parsed
    
    def _extract_publication_date(self, citation: str) -> str:
        """Extract publication date from citation."""
        if not citation:
            return "Date not available"
        
        # Look for year patterns
        year_match = re.search(r'\b(19|20)\d{2}\b', citation)
        if year_match:
            return year_match.group(0)
        
        return "Date not available"
    
    def process_patient_publications(self, patient_data: Dict, matches: List[Dict]) -> List[PublicationSummary]:
        """Process all publications for a single patient."""
        summaries = []
        total_publications = 0
        
        patient_id = patient_data.get('patient_id', 'Unknown')
        logger.info(f"Processing publications for Patient {patient_id}...")
        
        for match in matches:
            study_context = {
                'nct_id': match.get('nct_id', ''),
                'title': match.get('title', ''),
                'phase': match.get('phase', ''),
                'intervention': match.get('intervention', '')
            }
            
            publications_info = match.get('publications', {})
            if isinstance(publications_info, dict):
                publications = publications_info.get('publications', [])
                total_publications += len(publications)
                
                for pub in publications:
                    try:
                        summary = self.summarize_publication(pub, patient_data, study_context)
                        if summary:
                            summaries.append(summary)
                            logger.info(f"  ✓ Summarized PMID {summary.pmid}")
                        else:
                            logger.warning(f"  ✗ Failed to summarize PMID {pub.get('pmid', 'Unknown')}")
                        
                        # Rate limiting
                        time.sleep(2)
                        
                    except Exception as e:
                        logger.error(f"Error processing publication {pub.get('pmid', 'Unknown')}: {e}")
                        continue
        
        logger.info(f"Patient {patient_id}: {len(summaries)}/{total_publications} publications successfully summarized")
        return summaries
    
    def save_summaries(self, all_summaries: Dict[str, List[PublicationSummary]], output_dir: Path):
        """Save publication summaries to files."""
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_file = output_dir / f"publication_summaries_{timestamp}.json"
        
        # Convert to serializable format
        summaries_data = {}
        for patient_id, summaries in all_summaries.items():
            summaries_data[patient_id] = [asdict(summary) for summary in summaries]
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                "generation_date": datetime.now().isoformat(),
                "total_patients": len(all_summaries),
                "total_summaries": sum(len(summaries) for summaries in all_summaries.values()),
                "patient_summaries": summaries_data
            }, f, indent=2, ensure_ascii=False)
        
        # Save as readable text report
        txt_file = output_dir / f"publication_summaries_report_{timestamp}.txt"
        self._generate_text_report(all_summaries, txt_file)
        
        # Save therapy-focused summaries for prompt integration
        prompt_file = output_dir / f"therapy_evidence_for_prompt_{timestamp}.txt"
        self._generate_prompt_ready_summaries(all_summaries, prompt_file)
        
        logger.info(f"Saved publication summaries to:")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  Report: {txt_file}")
        logger.info(f"  Prompt Ready: {prompt_file}")
    
    def _generate_text_report(self, all_summaries: Dict[str, List[PublicationSummary]], output_file: Path):
        """Generate human-readable text report."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("PUBLICATION THERAPY SUMMARIES REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Patients: {len(all_summaries)}\n")
            f.write(f"Total Summaries: {sum(len(summaries) for summaries in all_summaries.values())}\n")
            f.write("-"*80 + "\n\n")
            
            for patient_id in sorted(all_summaries.keys(), key=lambda x: int(x) if x.isdigit() else x):
                summaries = all_summaries[patient_id]
                f.write(f"PATIENT {patient_id}\n")
                f.write("="*40 + "\n")
                f.write(f"Publications Summarized: {len(summaries)}\n\n")
                
                for i, summary in enumerate(summaries, 1):
                    f.write(f"{i}. {summary.title}\n")
                    f.write(f"   PMID: {summary.pmid}\n")
                    f.write(f"   Study: {summary.study_title} ({summary.study_nct_id})\n")
                    f.write(f"   Publication Date: {summary.publication_date}\n")
                    f.write(f"   \n")
                    f.write(f"   THERAPIERELEVANZ:\n")
                    f.write(f"   {summary.therapy_relevance_summary}\n\n")
                    f.write(f"   KERNBEFUNDE:\n")
                    f.write(f"   {summary.key_findings}\n\n")
                    f.write(f"   THERAPIEEMPFEHLUNGEN:\n")
                    f.write(f"   {summary.therapy_recommendations}\n\n")
                    f.write(f"   PATIENTENPOPULATION:\n")
                    f.write(f"   {summary.patient_population}\n\n")
                    f.write(f"   EVIDENZQUALITÄT:\n")
                    f.write(f"   {summary.evidence_quality}\n")
                    f.write("-"*40 + "\n\n")
                
                f.write("\n")
    
    def _generate_prompt_ready_summaries(self, all_summaries: Dict[str, List[PublicationSummary]], output_file: Path):
        """Generate therapy evidence summaries ready for use in therapy prompts."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# THERAPY EVIDENCE SUMMARIES FOR PROMPT INTEGRATION\n\n")
            f.write("# Diese Zusammenfassungen können direkt in Therapieempfehlungs-Prompts integriert werden\n\n")
            
            for patient_id in sorted(all_summaries.keys(), key=lambda x: int(x) if x.isdigit() else x):
                summaries = all_summaries[patient_id]
                f.write(f"## PATIENT {patient_id} - EVIDENZBASIS\n\n")
                
                f.write("### VERFÜGBARE THERAPIE-EVIDENZ:\n\n")
                
                for i, summary in enumerate(summaries, 1):
                    f.write(f"**Evidenz {i}: {summary.study_title}**\n")
                    f.write(f"- Publikation: {summary.title} (PMID: {summary.pmid})\n")
                    f.write(f"- Therapierelevanz: {summary.therapy_relevance_summary}\n")
                    f.write(f"- Kernbefunde: {summary.key_findings}\n")
                    f.write(f"- Empfehlungen: {summary.therapy_recommendations}\n")
                    f.write(f"- Geeignete Patienten: {summary.patient_population}\n")
                    f.write(f"- Evidenzqualität: {summary.evidence_quality}\n\n")
                
                f.write("---\n\n")

def main():
    """Main function to process all patient publications."""
    
    # Configuration
    INPUT_FILE = Path("C:/Users/pia/OneDrive - Universitaet Bern/Projects/NetTubo/netTubo/clinical_trials/clinical_trials_matches/clinical_trials_summary_llm.json")
    OUTPUT_DIR = Path("C:/Users/pia/OneDrive - Universitaet Bern/Projects/NetTubo/netTubo/clinical_trials/publication_summaries")
    
    if not INPUT_FILE.exists():
        logger.error(f"Input file not found: {INPUT_FILE}")
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load clinical trials data
    logger.info(f"Loading clinical trials data from {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        clinical_trials_data = json.load(f)
    
    # Initialize summarizer
    summarizer = PublicationSummarizer()
    
    # Process each patient
    all_summaries = {}
    total_patients = len(clinical_trials_data)
    
    logger.info(f"Starting publication summarization for {total_patients} patients...")
    
    for i, patient_entry in enumerate(clinical_trials_data):
        patient_id = patient_entry.get('patient_id', str(i+1))
        patient_data = patient_entry.get('patient_data', {})
        matches = patient_entry.get('matches', [])
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Patient {patient_id} ({i+1}/{total_patients})")
        logger.info(f"{'='*60}")
        
        try:
            patient_summaries = summarizer.process_patient_publications(
                {**patient_data, 'patient_id': patient_id}, 
                matches
            )
            all_summaries[patient_id] = patient_summaries
            
        except Exception as e:
            logger.error(f"Error processing Patient {patient_id}: {e}")
            all_summaries[patient_id] = []
    
    # Save all summaries
    logger.info(f"\nSaving publication summaries...")
    summarizer.save_summaries(all_summaries, OUTPUT_DIR)
    
    # Print summary statistics
    total_summaries = sum(len(summaries) for summaries in all_summaries.values())
    successful_patients = sum(1 for summaries in all_summaries.values() if len(summaries) > 0)
    
    print(f"\n" + "="*80)
    print("PUBLICATION SUMMARIZATION RESULTS")
    print("="*80)
    print(f"Total patients processed: {total_patients}")
    print(f"Patients with successful summaries: {successful_patients}")
    print(f"Total publication summaries created: {total_summaries}")
    print(f"Average summaries per patient: {total_summaries/total_patients:.1f}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
