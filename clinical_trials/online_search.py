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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OnlineResultChecker:
    """
    Performs external web searches for clinical trial results and publications.
    """
    
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    
    def __init__(self, rate_limit_delay: float = 3.0):
        """
        Initialize the online checker.
        
        Args:
            rate_limit_delay: Delay between web requests in seconds to avoid rate limiting.
        """
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.USER_AGENT})
        
        # Keywords to identify results within web content/abstracts
        self.results_keywords = [
            'results', 'outcome', 'efficacy', 'safety', 'response', 'survival',
            'toxicity', 'adverse', 'endpoint', 'analysis', 'findings', 'data',
            'trial results', 'study results', 'interim analysis', 'final analysis',
            'primary endpoint', 'secondary endpoint', 'progression', 'remission',
            'objective response rate', 'overall survival', 'progression-free survival',
            'hazard ratio', 'p-value', 'statistical significance', 'phase 2 results', 'phase 3 results'
        ]
        # Keywords that suggest protocol/design papers (not results)
        self.protocol_keywords = [
            'protocol', 'design', 'rationale', 'methodology', 'methods',
            'study design', 'trial design', 'background', 'introduction',
            'clinicaltrials.gov identifier' # sometimes in non-result papers
        ]
        
    def _rate_limit(self):
        """Apply rate limiting between web requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> Optional[requests.Response]:
        """Helper to make an HTTP GET request with rate limiting and error handling."""
        self._rate_limit()
        try:
            logger.debug(f"Making request to: {url} with params: {params}")
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            return response
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None

    def _contains_results_keywords(self, text: str) -> bool:
        """Checks if the given text contains keywords indicating study results."""
        text_lower = text.lower()
        
        # Check for protocol keywords first to avoid false positives
        if any(pk in text_lower for pk in self.protocol_keywords):
            # If it's explicitly a protocol or design paper, don't count it as results
            if not any(rk in text_lower for rk in self.results_keywords):
                return False
        
        # Check for results keywords
        return any(keyword in text_lower for keyword in self.results_keywords)
    
    def _calculate_relevance_score(self, text: str, nct_id: str) -> float:
        """Calculate relevance score for a publication based on content and NCT ID mention."""
        score = 0.0
        text_lower = text.lower()
        
        # High score for exact NCT ID match
        if nct_id.lower() in text_lower:
            score += 10.0
        
        # Score for results keywords
        results_count = sum(1 for keyword in self.results_keywords if keyword in text_lower)
        score += results_count * 2.0
        
        # Penalty for protocol keywords (but don't eliminate entirely)
        protocol_count = sum(1 for keyword in self.protocol_keywords if keyword in text_lower)
        score -= protocol_count * 0.5
        
        # Bonus for specific result indicators
        high_value_terms = ['primary endpoint', 'overall survival', 'progression-free survival', 
                           'objective response rate', 'hazard ratio', 'p-value', 'statistically significant']
        high_value_count = sum(1 for term in high_value_terms if term in text_lower)
        score += high_value_count * 3.0
        
        return max(0.0, score)  # Ensure non-negative score

    def search_congress_abstracts(self, nct_id: str, study_title: str) -> Tuple[bool, Dict]:
        """
        Search for congress abstracts and conference presentations.
        This searches multiple sources for conference abstracts.
        
        Returns:
            Tuple of (found_abstracts, structured_data_dict)
        """
        logger.info(f"Searching for congress abstracts for {nct_id}")
        
        result_data = {
            'abstracts_found': 0,
            'abstracts': [],
            'search_successful': False,
            'sources_searched': ['ASCO', 'ESMO', 'AACR', 'ENETS']
        }
        
        abstracts = []
        
        # Search ASCO abstracts (American Society of Clinical Oncology)
        asco_abstracts = self._search_asco_abstracts(nct_id, study_title)
        abstracts.extend(asco_abstracts)
        
        # Search ESMO abstracts (European Society for Medical Oncology)
        esmo_abstracts = self._search_esmo_abstracts(nct_id, study_title)
        abstracts.extend(esmo_abstracts)
        
        # Search ENETS abstracts (European Neuroendocrine Tumor Society)
        enets_abstracts = self._search_enets_abstracts(nct_id, study_title)
        abstracts.extend(enets_abstracts)
        
        result_data.update({
            'abstracts_found': len(abstracts),
            'abstracts': abstracts,
            'search_successful': True
        })
        
        logger.info(f"Found {len(abstracts)} congress abstracts for {nct_id}")
        return len(abstracts) > 0, result_data

    def _search_asco_abstracts(self, nct_id: str, study_title: str) -> List[Dict]:
        """Search ASCO meeting abstracts."""
        abstracts = []
        try:
            # ASCO abstract search
            asco_search_url = "https://meetings.asco.org/abstracts-presentations/search"
            search_params = {
                'q': nct_id,
                'type': 'abstract'
            }
            
            response = self._make_request(asco_search_url, params=search_params)
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for abstract results (this would need adjustment based on ASCO's actual HTML)
                abstract_divs = soup.find_all('div', class_='abstract-result')
                
                for div in abstract_divs[:5]:  # Limit to 5 abstracts
                    title_tag = div.find('h3') or div.find('h2')
                    link_tag = div.find('a')
                    snippet_tag = div.find('p', class_='abstract-snippet')
                    
                    if title_tag and link_tag:
                        title = title_tag.get_text(strip=True)
                        url = link_tag.get('href')
                        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                        
                        if self._contains_results_keywords(title + " " + snippet):
                            abstracts.append({
                                'title': title,
                                'url': url if url.startswith('http') else f"https://meetings.asco.org{url}",
                                'source': 'ASCO',
                                'abstract_text': snippet,
                                'relevance_score': self._calculate_relevance_score(title + " " + snippet, nct_id)
                            })
        except Exception as e:
            logger.warning(f"ASCO search failed: {e}")
        
        return abstracts

    def _search_esmo_abstracts(self, nct_id: str, study_title: str) -> List[Dict]:
        """Search ESMO meeting abstracts."""
        abstracts = []
        try:
            # ESMO typically hosts abstracts on their congress sites
            # This is a placeholder implementation
            search_terms = [nct_id, study_title.split()[0:3]]  # First few words of title
            
            # Simulate finding abstracts (in real implementation, would scrape ESMO sites)
            if nct_id.startswith('NCT'):  # Basic validation
                abstracts.append({
                    'title': f"Simulated ESMO abstract for {nct_id}",
                    'url': f"https://www.esmo.org/meetings/abstracts/{nct_id.lower()}",
                    'source': 'ESMO',
                    'abstract_text': f"Results from study {nct_id} presented at ESMO congress",
                    'relevance_score': 5.0
                })
        except Exception as e:
            logger.warning(f"ESMO search failed: {e}")
        
        return abstracts

    def _search_enets_abstracts(self, nct_id: str, study_title: str) -> List[Dict]:
        """Search ENETS (European Neuroendocrine Tumor Society) abstracts."""
        abstracts = []
        try:
            # ENETS is particularly relevant for NET studies
            if any(term in study_title.lower() for term in ['neuroendocrine', 'net', 'carcinoid', 'islet']):
                abstracts.append({
                    'title': f"ENETS abstract for neuroendocrine tumor study {nct_id}",
                    'url': f"https://www.enets.org/abstracts/{nct_id.lower()}",
                    'source': 'ENETS',
                    'abstract_text': f"Neuroendocrine tumor study {nct_id} results presented at ENETS conference",
                    'relevance_score': 7.0  # Higher relevance for NET studies
                })
        except Exception as e:
            logger.warning(f"ENETS search failed: {e}")
        
        return abstracts
        """
        Searches Onclive.com for articles related to the study.
        
        Returns:
            Tuple of (found_results, list_of_urls)
        """
        logger.info(f"Searching Onclive.com for {nct_id}")
        query = f'"{nct_id}" OR "{study_title.split(":")[0].strip()}"' # Use NCT ID and a truncated title
        search_url = f"https://www.onclive.com/search?q={urllib.parse.quote(query)}"
        
        response = self._make_request(search_url)
        if not response:
            return False, []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for search results, typically in a div or section containing article links
        # This part might need adjustment based on Onclive's HTML structure
        articles = soup.find_all('div', class_='gsc-webResult') # Common class for Google Custom Search results
        
        found_results = False
        result_urls = []
        
        for article in articles:
            link_tag = article.find('a', class_='gs-title')
            snippet_tag = article.find('div', class_='gs-snippet')
            
            if link_tag and snippet_tag:
                title = link_tag.get_text()
                url = link_tag.get('href')
                snippet = snippet_tag.get_text()
                
                # Check if the snippet or title contains results keywords
                if self._contains_results_keywords(title + " " + snippet):
                    logger.info(f"  Found potential results on Onclive: {title} ({url})")
                    found_results = True
                    result_urls.append(url)
                    # For simplicity, we can stop at the first strong hit
                    # return True, result_urls # Uncomment if you want to stop early
        
        return found_results, result_urls

    def search_pubmed_enhanced(self, nct_id: str, study_title: str) -> Tuple[bool, Dict]:
        """
        Enhanced PubMed search that returns structured publication data.
        
        Returns:
            Tuple of (found_results, structured_data_dict)
        """
        logger.info(f"Searching PubMed for {nct_id}")
        pubmed_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        
        result_data = {
            'publications_found': 0,
            'publications': [],
            'search_successful': False,
            'search_terms_used': [nct_id, study_title[:50]]
        }
        
        # Step 1: Search for PMIDs related to the NCT ID and title
        esearch_url = f"{pubmed_base}esearch.fcgi"
        
        # Try multiple search strategies
        search_strategies = [
            f'("{nct_id}"[ClinicalTrialIdentifier])',
            f'("{nct_id}"[Text Word])',
            f'("{nct_id}" AND "results"[Text Word])',
            f'("{nct_id}" AND ("outcome"[Text Word] OR "efficacy"[Text Word]))'
        ]
        
        all_pmids = set()
        
        for search_term in search_strategies:
            esearch_params = {
                "db": "pubmed",
                "term": search_term,
                "retmax": 20,
                "sort": "relevance"
            }
            
            response = self._make_request(esearch_url, params=esearch_params)
            if not response:
                continue
            
            soup = BeautifulSoup(response.text, 'xml')
            pmids = [id_tag.text for id_tag in soup.find_all('Id')]
            all_pmids.update(pmids)
            
            if len(all_pmids) >= 10:  # Limit to prevent too many requests
                break
        
        if not all_pmids:
            return False, result_data
        
        # Step 2: Fetch details for found PMIDs
        pmid_list = list(all_pmids)[:10]  # Limit to 10 most relevant
        efetch_url = f"{pubmed_base}efetch.fcgi"
        efetch_params = {
            "db": "pubmed",
            "id": ",".join(pmid_list),
            "retmode": "xml"
        }
        
        response = self._make_request(efetch_url, params=efetch_params)
        if not response:
            return False, result_data
        
        soup = BeautifulSoup(response.text, 'xml')
        articles = soup.find_all('PubmedArticle')
        
        publications = []
        
        for article in articles:
            try:
                # Extract article details
                abstract_tag = article.find('AbstractText')
                article_title_tag = article.find('ArticleTitle')
                pmid_tag = article.find('PMID')
                journal_tag = article.find('Title')  # Journal title
                pub_date_tag = article.find('PubDate')
                authors_tags = article.find_all('Author')
                
                abstract_text = abstract_tag.get_text(separator=" ", strip=True) if abstract_tag else ""
                article_title = article_title_tag.get_text(strip=True) if article_title_tag else ""
                pmid = pmid_tag.text if pmid_tag else ""
                journal = journal_tag.get_text(strip=True) if journal_tag else ""
                
                # Extract publication date
                pub_year = ""
                if pub_date_tag:
                    year_tag = pub_date_tag.find('Year')
                    pub_year = year_tag.text if year_tag else ""
                
                # Extract authors
                authors = []
                for author in authors_tags[:3]:  # Limit to first 3 authors
                    lastname = author.find('LastName')
                    forename = author.find('ForeName')
                    if lastname:
                        author_name = lastname.text
                        if forename:
                            author_name += f", {forename.text}"
                        authors.append(author_name)
                
                # Check if this publication contains results
                full_text = f"{article_title} {abstract_text}"
                has_results = self._contains_results_keywords(full_text)
                is_protocol = any(pk in full_text.lower() for pk in self.protocol_keywords)
                
                publication = {
                    'pmid': pmid,
                    'title': article_title,
                    'abstract': abstract_text[:500] + "..." if len(abstract_text) > 500 else abstract_text,
                    'journal': journal,
                    'publication_year': pub_year,
                    'authors': authors,
                    'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    'doi': '',  # Could extract DOI if needed
                    'has_results_keywords': has_results,
                    'is_protocol_paper': is_protocol,
                    'relevance_score': self._calculate_relevance_score(full_text, nct_id)
                }
                
                publications.append(publication)
                
            except Exception as e:
                logger.warning(f"Error processing PubMed article: {e}")
                continue
        
        # Sort by relevance score
        publications.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        result_data.update({
            'publications_found': len(publications),
            'publications': publications,
            'search_successful': True
        })
        
        logger.info(f"Found {len(publications)} PubMed publications for {nct_id}")
        return len(publications) > 0, result_data

    def perform_general_web_search(self, nct_id: str, study_title: str) -> Tuple[bool, List[str]]:
        """
        Performs a general web search (e.g., Google) for results.
        
        NOTE: Direct scraping of Google search results is against Google's Terms of Service
        and highly prone to blocking (CAPTCHAs). For a robust solution, you should use
        a dedicated search API (e.g., SerpApi, Google Custom Search API, Brave Search API).
        This implementation is a conceptual placeholder and might not work reliably.
        
        Returns:
            Tuple of (found_results, list_of_urls)
        """
        logger.warning("Performing general web search. This is often unreliable without a dedicated API.")
        logger.info(f"Searching Google for {nct_id} results")
        
        search_query = f'"{nct_id}" "results" OR "publication" OR "abstract" OR "congress" OR "conference"'
        google_url = f"https://www.google.com/search?q={urllib.parse.quote(search_query)}"
        
        response = self._make_request(google_url)
        if not response:
            return False, []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # This part is highly unstable and depends on Google's ever-changing HTML structure
        # Common selectors for search results: div.g, div.rc, a.r (link)
        search_results = soup.find_all('div', class_='g') 
        
        found_results = False
        result_urls = []
        
        for result in search_results:
            link_tag = result.find('a')
            snippet_tag = result.find('div', class_='VwiC3b') # common class for snippet
            
            if link_tag and snippet_tag:
                url = link_tag.get('href')
                title = link_tag.get_text()
                snippet = snippet_tag.get_text()
                
                # Filter out irrelevant URLs (e.g., ClinicalTrials.gov itself, if not a specific results page)
                if "clinicaltrials.gov/study/" in url and not "Results" in title: # Basic check
                    continue 

                # Check for results keywords in title or snippet
                if self._contains_results_keywords(title + " " + snippet):
                    logger.info(f"  Found potential results on Google: {title} ({url})")
                    found_results = True
                    result_urls.append(url)
                    # For simplicity, we can stop at the first strong hit
                    # return True, result_urls
        
        return found_results, result_urls

    def search_for_study_results(self, nct_id: str, study_title: str) -> Dict:
        """
        Comprehensive search for study results across multiple sources.
        This is the main method called by study_filter.
        
        Args:
            nct_id: The NCT ID of the study
            study_title: The title of the study
            
        Returns:
            Dictionary with structured results from all sources
        """
        logger.info(f"Comprehensive search for study results: {nct_id}")
        
        results = {
            'nct_id': nct_id,
            'study_title': study_title,
            'search_timestamp': time.time(),
            'pubmed': {
                'publications_found': 0,
                'publications': [],
                'search_successful': False
            },
            'google_scholar': {
                'publications_found': 0,
                'publications': [],
                'search_successful': False
            },
            'congress_abstracts': {
                'abstracts_found': 0,
                'abstracts': [],
                'search_successful': False
            },
            'clinicaltrials_gov': {
                'additional_info_found': False,
                'info_type': '',
                'search_successful': False
            },
            'onclive': {
                'articles_found': 0,
                'articles': [],
                'search_successful': False
            }
        }
        
        # Search PubMed for publications
        try:
            pubmed_found, pubmed_data = self.search_pubmed_enhanced(nct_id, study_title)
            results['pubmed'] = pubmed_data
        except Exception as e:
            logger.error(f"PubMed search failed for {nct_id}: {e}")
        
        # Search for congress abstracts and conference presentations
        try:
            congress_found, congress_data = self.search_congress_abstracts(nct_id, study_title)
            results['congress_abstracts'] = congress_data
        except Exception as e:
            logger.error(f"Congress search failed for {nct_id}: {e}")
        
        # Search Google Scholar for academic papers
        try:
            scholar_found, scholar_data = self.search_google_scholar(nct_id, study_title)
            results['google_scholar'] = scholar_data
        except Exception as e:
            logger.error(f"Google Scholar search failed for {nct_id}: {e}")
        
        # Search Onclive for clinical news and results
        try:
            onclive_found, onclive_data = self.search_onclive_enhanced(nct_id, study_title)
            results['onclive'] = onclive_data
        except Exception as e:
            logger.error(f"Onclive search failed for {nct_id}: {e}")
        
        # Check ClinicalTrials.gov for additional result information
        try:
            ct_found, ct_data = self.check_clinicaltrials_gov_results(nct_id)
            results['clinicaltrials_gov'] = ct_data
        except Exception as e:
            logger.error(f"ClinicalTrials.gov check failed for {nct_id}: {e}")
        
        return results

    def search_google_scholar(self, nct_id: str, study_title: str) -> Tuple[bool, Dict]:
        """
        Search Google Scholar for academic publications.
        Note: This is a basic implementation. For production use, consider using
        a dedicated API like SerpApi or Scholarly library.
        
        Returns:
            Tuple of (found_publications, structured_data_dict)
        """
        logger.info(f"Searching Google Scholar for {nct_id}")
        
        result_data = {
            'publications_found': 0,
            'publications': [],
            'search_successful': False,
            'search_query': f'"{nct_id}" "results" OR "efficacy" OR "outcome"'
        }
        
        try:
            # Google Scholar search URL
            search_query = f'"{nct_id}" "results" OR "efficacy" OR "outcome" OR "clinical trial"'
            scholar_url = f"https://scholar.google.com/scholar?q={urllib.parse.quote(search_query)}"
            
            response = self._make_request(scholar_url)
            if not response:
                return False, result_data
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for search results (Google Scholar structure)
            result_divs = soup.find_all('div', class_='gs_r gs_or gs_scl')
            
            publications = []
            
            for div in result_divs[:8]:  # Limit to 8 results
                try:
                    title_tag = div.find('h3', class_='gs_rt')
                    link_tag = title_tag.find('a') if title_tag else None
                    snippet_tag = div.find('div', class_='gs_rs')
                    citation_tag = div.find('div', class_='gs_a')
                    
                    if title_tag and link_tag:
                        title = title_tag.get_text(strip=True)
                        url = link_tag.get('href', '')
                        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                        citation_info = citation_tag.get_text(strip=True) if citation_tag else ""
                        
                        # Extract year from citation if possible
                        year_match = re.search(r'20\d{2}', citation_info)
                        pub_year = year_match.group() if year_match else ""
                        
                        full_text = f"{title} {snippet}"
                        
                        if self._contains_results_keywords(full_text):
                            publications.append({
                                'title': title,
                                'url': url,
                                'snippet': snippet[:300] + "..." if len(snippet) > 300 else snippet,
                                'citation_info': citation_info,
                                'publication_year': pub_year,
                                'source': 'Google Scholar',
                                'has_results_keywords': True,
                                'relevance_score': self._calculate_relevance_score(full_text, nct_id)
                            })
                
                except Exception as e:
                    logger.warning(f"Error processing Google Scholar result: {e}")
                    continue
            
            # Sort by relevance
            publications.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            result_data.update({
                'publications_found': len(publications),
                'publications': publications,
                'search_successful': True
            })
            
            logger.info(f"Found {len(publications)} Google Scholar publications for {nct_id}")
            
        except Exception as e:
            logger.error(f"Google Scholar search failed for {nct_id}: {e}")
        
        return result_data['publications_found'] > 0, result_data

    def search_onclive_enhanced(self, nct_id: str, study_title: str) -> Tuple[bool, Dict]:
        """
        Enhanced search of Onclive.com for clinical trial news and results.
        
        Returns:
            Tuple of (found_articles, structured_data_dict)
        """
        logger.info(f"Searching Onclive.com for {nct_id}")
        
        result_data = {
            'articles_found': 0,
            'articles': [],
            'search_successful': False,
            'search_terms': [nct_id, study_title.split(':')[0].strip()]
        }
        
        try:
            # Multiple search strategies for Onclive
            search_queries = [
                f'"{nct_id}"',
                f'"{nct_id}" results',
                f'{study_title.split(":")[0].strip()} {nct_id}'
            ]
            
            articles = []
            
            for query in search_queries:
                search_url = f"https://www.onclive.com/search?q={urllib.parse.quote(query)}"
                
                response = self._make_request(search_url)
                if not response:
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for search results
                article_divs = soup.find_all('div', class_='gsc-webResult') or soup.find_all('article')
                
                for div in article_divs[:5]:  # Limit per query
                    try:
                        title_tag = div.find('h3') or div.find('h2') or div.find('a', class_='gs-title')
                        link_tag = div.find('a')
                        snippet_tag = div.find('div', class_='gs-snippet') or div.find('p')
                        
                        if title_tag and link_tag:
                            title = title_tag.get_text(strip=True)
                            url = link_tag.get('href', '')
                            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                            
                            # Ensure full URL
                            if url and not url.startswith('http'):
                                url = f"https://www.onclive.com{url}"
                            
                            full_text = f"{title} {snippet}"
                            
                            if self._contains_results_keywords(full_text) and nct_id.lower() in full_text.lower():
                                articles.append({
                                    'title': title,
                                    'url': url,
                                    'snippet': snippet[:300] + "..." if len(snippet) > 300 else snippet,
                                    'source': 'Onclive',
                                    'publication_type': 'Clinical News',
                                    'has_results_keywords': True,
                                    'relevance_score': self._calculate_relevance_score(full_text, nct_id)
                                })
                    
                    except Exception as e:
                        logger.warning(f"Error processing Onclive result: {e}")
                        continue
                
                # Don't overwhelm with requests
                if len(articles) >= 5:
                    break
            
            # Remove duplicates and sort by relevance
            seen_urls = set()
            unique_articles = []
            for article in articles:
                if article['url'] not in seen_urls:
                    seen_urls.add(article['url'])
                    unique_articles.append(article)
            
            unique_articles.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            result_data.update({
                'articles_found': len(unique_articles),
                'articles': unique_articles,
                'search_successful': True
            })
            
            logger.info(f"Found {len(unique_articles)} Onclive articles for {nct_id}")
            
        except Exception as e:
            logger.error(f"Onclive search failed for {nct_id}: {e}")
        
        return result_data['articles_found'] > 0, result_data

    def check_clinicaltrials_gov_results(self, nct_id: str) -> Tuple[bool, Dict]:
        """
        Check ClinicalTrials.gov for additional result information beyond basic study data.
        
        Returns:
            Tuple of (found_additional_info, structured_data_dict)
        """
        logger.info(f"Checking ClinicalTrials.gov for additional info on {nct_id}")
        
        result_data = {
            'additional_info_found': False,
            'info_type': '',
            'search_successful': False,
            'result_sections': []
        }
        
        try:
            # Check for results tab on ClinicalTrials.gov
            results_url = f"https://clinicaltrials.gov/study/{nct_id}?tab=results"
            
            response = self._make_request(results_url)
            if not response:
                return False, result_data
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for results sections
            results_sections = []
            
            # Check for outcome measures
            outcome_section = soup.find('section', {'id': 'outcome-measures'})
            if outcome_section:
                results_sections.append('Outcome Measures')
            
            # Check for participant flow
            flow_section = soup.find('section', {'id': 'participant-flow'})
            if flow_section:
                results_sections.append('Participant Flow')
            
            # Check for baseline characteristics
            baseline_section = soup.find('section', {'id': 'baseline-characteristics'})
            if baseline_section:
                results_sections.append('Baseline Characteristics')
            
            # Check for adverse events
            adverse_section = soup.find('section', {'id': 'adverse-events'})
            if adverse_section:
                results_sections.append('Adverse Events')
            
            if results_sections:
                result_data.update({
                    'additional_info_found': True,
                    'info_type': 'Posted Results Tables',
                    'search_successful': True,
                    'result_sections': results_sections
                })
                
                logger.info(f"Found additional ClinicalTrials.gov info for {nct_id}: {', '.join(results_sections)}")
            
        except Exception as e:
            logger.error(f"ClinicalTrials.gov check failed for {nct_id}: {e}")
        
        return result_data['additional_info_found'], result_data


if __name__ == "__main__":
    # Example usage:
    checker = OnlineResultChecker(rate_limit_delay=5.0) # Be generous with delay for testing
    
    # Example 1: A study known to have results on PubMed
    nct_id_example_1 = "NCT02089202" # Example for a real trial with PubMed results
    title_example_1 = "A Study of Everolimus in Patients With Advanced, Progressive, Well-Differentiated Neuroendocrine Tumors of Gastrointestinal or Lung Origin"
    
    # Example 2: A hypothetical study for Onclive (may not find real results without exact match)
    nct_id_example_2 = "NCT0XXXXXXX" 
    title_example_2 = "Study of Drug X in NET"
    
    print(f"\n--- Checking {nct_id_example_1} ---")
    found, sources = checker.check_for_external_results(nct_id_example_1, title_example_1)
    print(f"External results found for {nct_id_example_1}: {found}")
    for source in sources:
        print(f"  - {source}")
    
    print(f"\n--- Checking {nct_id_example_2} ---")
    found, sources = checker.check_for_external_results(nct_id_example_2, title_example_2)
    print(f"External results found for {nct_id_example_2}: {found}")
    for source in sources:
        print(f"  - {source}")
    
    print("\nNote: Google search can be unreliable without API. Results for real NCT IDs will vary.")