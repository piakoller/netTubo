#!/usr/bin/env python3
"""
Online Result Checker for Clinical Trials

This script provides utilities to search external websites (PubMed, Onclive, Google)
for publications and results related to clinical trials using NCT IDs and titles.
"""

import sys
import time
import logging
import re
import requests
from bs4 import BeautifulSoup
import urllib.parse
from typing import List, Dict, Tuple, Optional
import random # CHANGED: Import random for jitter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OnlineResultChecker:
    """
    Performs external web searches for clinical trial results and publications.
    """
    
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    
    REAL_BROWSER_HEADERS = {
        'User-Agent': USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    
    def __init__(self, rate_limit_delay: float = 2.0):
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0
        
        self.results_keywords = [
            'results', 'outcome', 'efficacy', 'safety', 'response', 'survival',
            'toxicity', 'adverse', 'endpoint', 'analysis', 'findings', 'data',
            'trial results', 'study results', 'interim analysis', 'final analysis',
            'primary endpoint', 'secondary endpoint', 'progression', 'remission',
            'objective response rate', 'overall survival', 'progression-free survival'
        ]
        
    def _rate_limit(self):
        """Simple rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def _make_request_with_session(self, url: str, timeout: int = 15) -> Optional[requests.Response]:
        """Make a simplified request with session and basic error handling."""
        self._rate_limit()
        try:
            session = requests.Session()
            session.headers.update(self.REAL_BROWSER_HEADERS)
            response = session.get(url, timeout=timeout)
            
            if response.status_code == 200:
                return response
            elif response.status_code == 403:
                logger.warning(f"[403] Access blocked to {url}")
            elif response.status_code == 429:
                logger.warning(f"[429] Rate limited by {url}")
            else:
                logger.warning(f"[{response.status_code}] Error for {url}")
                
        except requests.RequestException as e:
            logger.warning(f"Request failed for {url}: {e}")
        
        return None

    def _has_results_content(self, text: str) -> bool:
        """Check if text contains relevant results keywords."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.results_keywords)



    def _contains_results_keywords(self, text: str) -> bool:
        """Simplified keyword check focusing on results content."""
        return self._has_results_content(text)
    
    def _calculate_relevance_score(self, text: str, nct_id: str) -> float:
        """Calculate relevance score for search results."""
        score = 0.0
        text_lower = text.lower()
        
        # High score for NCT ID match
        if nct_id.lower() in text_lower:
            score += 10.0
        
        # Score for results keywords
        results_count = sum(1 for keyword in self.results_keywords if keyword in text_lower)
        score += results_count * 2.0
        
        return score

    def _search_google(self, nct_id: str, site_domain: str, source_name: str) -> List[Dict]:
        """Simplified Google search."""
        articles = []
        search_query = f"site:{site_domain} \"{nct_id}\""
        google_url = f"https://www.google.com/search?q={urllib.parse.quote(search_query)}"
        
        logger.info(f"Searching {source_name} via Google: {search_query}")
        response = self._make_request_with_session(google_url)
        
        if not response:
            return articles
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.select('div.yuRUbf')
        
        if not results:
            logger.info(f"No Google results for {source_name}")
            return articles

        for result_div in results[:5]:  # Limit to first 5 results
            try:
                link_tag = result_div.find('a', href=True)
                if not link_tag:
                    continue
                
                url = link_tag.get('href', '')
                if url.startswith('/url?q='):
                    url = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get('q', [''])[0]

                if not url or site_domain not in url:
                    continue

                title_tag = link_tag.find('h3')
                title = title_tag.get_text(strip=True) if title_tag else f"{source_name} article"
                
                # Get snippet from parent container
                parent_container = result_div.find_parent('div.g')
                snippet = ""
                if parent_container:
                    snippet_tag = parent_container.find('div', {'data-sncf': '1'})
                    if snippet_tag:
                        snippet = snippet_tag.get_text(strip=True)

                full_text = f"{title} {snippet}"
                if self._has_results_content(full_text):
                    articles.append({
                        'title': title,
                        'url': url,
                        'source': source_name,
                        'abstract_text': snippet,
                        'relevance_score': self._calculate_relevance_score(full_text, nct_id)
                    })
            except Exception as e:
                logger.debug(f"Error processing Google result: {e}")
                
        logger.info(f"Found {len(articles)} {source_name} articles via Google")
        return articles

    def search_congress_abstracts(self, nct_id: str, study_title: str) -> Tuple[bool, Dict]:
        """Search for congress abstracts from key sources."""
        logger.info(f"Searching congress abstracts for {nct_id}")
        
        abstracts = []
        abstracts.extend(self._search_google(nct_id, 'ascopubs.org', 'ASCO'))
        abstracts.extend(self._search_google(nct_id, 'annalsofoncology.org', 'Annals of Oncology'))
        
        # Only search ENETS for relevant studies
        if any(term in study_title.lower() for term in ['neuroendocrine', 'net', 'carcinoid', 'islet']):
            abstracts.extend(self._search_enets_direct(nct_id))
        
        result_data = {
            'abstracts_found': len(abstracts),
            'abstracts': sorted(abstracts, key=lambda x: x.get('relevance_score', 0), reverse=True),
            'search_successful': True,
            'sources_searched': ['ASCO', 'Annals of Oncology', 'ENETS']
        }
        
        logger.info(f"Found {len(abstracts)} congress abstracts for {nct_id}")
        return len(abstracts) > 0, result_data

    def _search_enets_direct(self, nct_id: str) -> List[Dict]:
        """Direct search of ENETS website."""
        abstracts = []
        try:
            search_url = f"https://www.enets.org/abstract-library/search/{urllib.parse.quote(nct_id)}.html"
            response = self._make_request_with_session(search_url)
            
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                for div in soup.select('.abstract-item, .search-result, article')[:3]:
                    title_tag = div.select_one('h1, h2, h3, h4, .title, .abstract-title')
                    link_tag = div.select_one('a[href*="abstract"], a[href*="congress"], a')
                    
                    if title_tag and link_tag:
                        title = title_tag.get_text(strip=True)
                        url = urllib.parse.urljoin("https://www.enets.org/", link_tag.get('href', ''))
                        snippet_tag = div.select_one('.abstract-text, .summary, p')
                        snippet = snippet_tag.get_text(strip=True)[:200] if snippet_tag else ""
                        
                        full_text = f"{title} {snippet}"
                        if self._has_results_content(full_text):
                            abstracts.append({
                                'title': title,
                                'url': url,
                                'source': 'ENETS',
                                'abstract_text': snippet,
                                'relevance_score': self._calculate_relevance_score(full_text, nct_id)
                            })
        except Exception as e:
            logger.warning(f"ENETS search failed: {e}")
        
        return abstracts

    def search_onclive_enhanced(self, nct_id: str, study_title: str) -> Tuple[bool, Dict]:
        """Search Onclive for relevant articles."""
        logger.info(f"Searching Onclive for {nct_id}")
        
        articles = self._search_google(nct_id, 'onclive.com', 'Onclive')
        
        result_data = {
            'articles_found': len(articles),
            'articles': sorted(articles, key=lambda x: x.get('relevance_score', 0), reverse=True),
            'search_successful': True
        }
        
        logger.info(f"Found {len(articles)} Onclive articles for {nct_id}")
        return len(articles) > 0, result_data

    def search_pubmed_enhanced(self, nct_id: str, study_title: str) -> Tuple[bool, Dict]:
        """Search PubMed for publications."""
        logger.info(f"Searching PubMed for {nct_id}")
        pubmed_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        
        # Search strategies in order of preference
        search_strategies = [
            f'("{nct_id}"[ClinicalTrialIdentifier])',
            f'("{nct_id}"[Text Word])',
            f'{nct_id}'
        ]
        
        all_pmids = set()
        for search_term in search_strategies:
            esearch_params = {"db": "pubmed", "term": search_term, "retmax": 10, "sort": "relevance"}
            response = self._make_request_with_session(f"{pubmed_base}esearch.fcgi?{urllib.parse.urlencode(esearch_params)}")
            
            if response:
                soup = BeautifulSoup(response.text, 'xml')
                all_pmids.update(id_tag.text for id_tag in soup.find_all('Id'))
                
            if len(all_pmids) >= 10:
                break
        
        if not all_pmids:
            return False, {'publications_found': 0, 'publications': [], 'search_successful': False}
        
        # Fetch article details
        pmid_list = list(all_pmids)[:10]
        efetch_params = {"db": "pubmed", "id": ",".join(pmid_list), "retmode": "xml"}
        response = self._make_request_with_session(f"{pubmed_base}efetch.fcgi?{urllib.parse.urlencode(efetch_params)}")
        
        if not response:
            return False, {'publications_found': 0, 'publications': [], 'search_successful': False}
        
        soup = BeautifulSoup(response.text, 'xml')
        publications = []
        
        for article in soup.find_all('PubmedArticle'):
            try:
                abstract_tag = article.find('AbstractText')
                title_tag = article.find('ArticleTitle')
                pmid_tag = article.find('PMID')
                
                abstract_text = abstract_tag.get_text(separator=" ", strip=True) if abstract_tag else ""
                title = title_tag.get_text(strip=True) if title_tag else ""
                pmid = pmid_tag.text if pmid_tag else ""
                
                full_text = f"{title} {abstract_text}"
                publications.append({
                    'pmid': pmid,
                    'title': title,
                    'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    'has_results_keywords': self._has_results_content(full_text),
                    'relevance_score': self._calculate_relevance_score(full_text, nct_id),
                    'abstract': abstract_text[:400] + "..." if len(abstract_text) > 400 else abstract_text
                })
            except Exception as e:
                logger.warning(f"Error processing PubMed article: {e}")
        
        publications.sort(key=lambda x: x['relevance_score'], reverse=True)
        result_data = {
            'publications_found': len(publications),
            'publications': publications,
            'search_successful': True
        }
        
        logger.info(f"Found {len(publications)} PubMed publications for {nct_id}")
        return len(publications) > 0, result_data

    def search_google_scholar(self, nct_id: str, study_title: str) -> Tuple[bool, Dict]:
        """Search Google Scholar for publications."""
        logger.info(f"Searching Google Scholar for {nct_id}")
        
        search_query = f'"{nct_id}" results OR efficacy OR outcome'
        scholar_url = f"https://scholar.google.com/scholar?q={urllib.parse.quote(search_query)}"
        
        response = self._make_request_with_session(scholar_url)
        if not response:
            return False, {'publications_found': 0, 'publications': [], 'search_successful': False}
        
        soup = BeautifulSoup(response.text, 'html.parser')
        publications = []
        
        for div in soup.select('div.gs_r.gs_or.gs_scl')[:5]:
            try:
                title_tag = div.select_one('h3.gs_rt a')
                if not title_tag:
                    continue
                    
                title = title_tag.get_text(strip=True)
                url = title_tag.get('href', '')
                
                snippet_tag = div.select_one('.gs_rs')
                snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                
                full_text = f"{title} {snippet}"
                if url and self._has_results_content(full_text):
                    publications.append({
                        'title': title,
                        'url': url,
                        'snippet': snippet,
                        'source': 'Google Scholar',
                        'relevance_score': self._calculate_relevance_score(full_text, nct_id)
                    })
            except Exception as e:
                logger.debug(f"Error processing Google Scholar result: {e}")
        
        publications.sort(key=lambda x: x['relevance_score'], reverse=True)
        result_data = {
            'publications_found': len(publications),
            'publications': publications,
            'search_successful': True
        }
        
        logger.info(f"Found {len(publications)} Google Scholar publications for {nct_id}")
        return len(publications) > 0, result_data

    def search_for_study_results(self, nct_id: str, study_title: str) -> Dict:
        logger.info(f"Comprehensive search for study results: {nct_id}")
        results = {
            'nct_id': nct_id, 'study_title': study_title, 'search_timestamp': time.time(),
            'pubmed': {}, 'google_scholar': {}, 'congress_abstracts': {}, 'onclive': {}
        }
        try:
            _, results['pubmed'] = self.search_pubmed_enhanced(nct_id, study_title)
        except Exception as e:
            logger.error(f"PubMed search failed for {nct_id}: {e}")
        try:
            _, results['congress_abstracts'] = self.search_congress_abstracts(nct_id, study_title)
        except Exception as e:
            logger.error(f"Congress search failed for {nct_id}: {e}")
        try:
            _, results['google_scholar'] = self.search_google_scholar(nct_id, study_title)
        except Exception as e:
            logger.error(f"Google Scholar search failed for {nct_id}: {e}")
        try:
            _, results['onclive'] = self.search_onclive_enhanced(nct_id, study_title)
        except Exception as e:
            logger.error(f"Onclive search failed for {nct_id}: {e}")
        return results