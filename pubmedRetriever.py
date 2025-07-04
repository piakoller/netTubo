import logging
from datetime import datetime
import time
from typing import Dict, List, Optional
import requests
from pathlib import Path
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PubMedRetriever:
    """Class to retrieve relevant papers from PubMed API"""
    
    def __init__(self, 
                 email: str,
                 tool: str = "NetTuboPubMedRetriever",
                 base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"):
        self.email = email
        self.tool = tool
        self.base_url = base_url
        self.cache_dir = Path("data/pubmed_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _build_search_query(self, diagnosis: str, start_date: str) -> str:
        """Build PubMed search query with date restriction"""
        # Convert diagnosis to search terms
        search_terms = [
            diagnosis,
            "neuroendocrine tumor",
            "NET",
            "(treatment OR therapy)"
        ]
        
        # Add date restriction
        date_restriction = f"'{start_date}'[Date - Publication] : '3000'[Date - Publication]"
        
        # Combine terms
        query = f"({' AND '.join(search_terms)}) AND {date_restriction}"
        return query

    def _get_cache_path(self, diagnosis: str, start_date: str) -> Path:
        """Get cache file path for given parameters"""
        safe_diagnosis = "".join(c if c.isalnum() else "_" for c in diagnosis)
        return self.cache_dir / f"{safe_diagnosis}_{start_date}.json"

    def _load_from_cache(self, cache_path: Path) -> Optional[Dict]:
        """Load results from cache if available and not expired"""
        if not cache_path.exists():
            return None
            
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            
            # Check if cache is older than 7 days
            cache_date = datetime.fromisoformat(cached['cache_date'])
            if (datetime.now() - cache_date).days > 7:
                return None
                
            return cached['results']
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def _save_to_cache(self, cache_path: Path, results: Dict):
        """Save results to cache with timestamp"""
        try:
            cache_data = {
                'cache_date': datetime.now().isoformat(),
                'results': results
            }
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

    def search_publications(self, 
                          diagnosis: str, 
                          start_date: str,
                          max_results: int = 100) -> List[Dict]:
        """
        Search for relevant publications on PubMed
        
        Args:
            diagnosis: Patient diagnosis to search for
            start_date: Start date in format 'YYYY/MM/DD'
            max_results: Maximum number of results to return
            
        Returns:
            List of publication dictionaries with metadata
        """
        cache_path = self._get_cache_path(diagnosis, start_date)
        cached_results = self._load_from_cache(cache_path)
        if cached_results:
            logger.info(f"Using cached results for {diagnosis} from {start_date}")
            return cached_results

        try:
            # First get IDs
            search_query = self._build_search_query(diagnosis, start_date)
            search_params = {
                'db': 'pubmed',
                'term': search_query,
                'retmax': max_results,
                'tool': self.tool,
                'email': self.email,
                'retmode': 'json',
                'sort': 'date'
            }
            
            search_url = f"{self.base_url}/esearch.fcgi"
            response = requests.get(search_url, params=search_params)
            response.raise_for_status()
            
            pmids = response.json()['esearchresult']['idlist']
            if not pmids:
                logger.info(f"No results found for {diagnosis} from {start_date}")
                return []

            # Then fetch details for these IDs
            details_params = {
                'db': 'pubmed',
                'id': ','.join(pmids),
                'tool': self.tool,
                'email': self.email,
                'retmode': 'json'
            }
            
            details_url = f"{self.base_url}/esummary.fcgi"
            response = requests.get(details_url, params=details_params)
            response.raise_for_status()
            
            results = []
            papers = response.json()['result']
            for pmid in pmids:
                paper = papers[pmid]
                results.append({
                    'pmid': pmid,
                    'title': paper.get('title', ''),
                    'authors': [author.get('name', '') for author in paper.get('authors', [])],
                    'journal': paper.get('fulljournalname', ''),
                    'publication_date': paper.get('pubdate', ''),
                    'doi': paper.get('elocationid', ''),
                    'abstract_url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                })

            # Cache results
            self._save_to_cache(cache_path, results)
            
            return results

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching from PubMed: {e}")
            return []

def main():
    """Example usage of PubMedRetriever"""
    # Initialize retriever with your email
    retriever = PubMedRetriever(email="pia.koller@unibe.ch")
    ENET ="2023/04/27"
    ESMO = "2020/04/06"
    diagnosis = "neuroendocrine tumor"
    start_date = ENET

    
    papers = retriever.search_publications(
        diagnosis=diagnosis,
        start_date=start_date,
        max_results=100
    )
    
    # Print results
    print(f"\nFound {len(papers)} papers for {diagnosis} since {start_date}:")
    for paper in papers:
        print(f"\nTitle: {paper['title']}")
        print(f"Authors: {', '.join(paper['authors'])}")
        print(f"Journal: {paper['journal']}")
        print(f"Date: {paper['publication_date']}")
        print(f"URL: {paper['abstract_url']}")
        print("-" * 80)

if __name__ == "__main__":
    main()