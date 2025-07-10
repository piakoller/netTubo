#!/usr/bin/env python3
"""
Publication downloader for clinical trials.

This module handles downloading and managing publication information
related to clinical trials from PubMed.
"""

import json
import logging
import requests
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


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
