#!/usr/bin/env python3
"""
Online Result Checker for Clinical Trials

This script provides utilities to search external websites using HTTP requests.
Simplified version using requests and BeautifulSoup instead of Selenium.
"""

import sys
import time
import logging
import re
import urllib.parse
from typing import List, Dict, Tuple, Optional
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OnlineResultChecker:
    """
    Performs external web searches for clinical trial results using HTTP requests.
    """
    
    def __init__(self, rate_limit_delay: float = 2.0, scrape_content: bool = True, max_content_length: int = 2000):
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0
        self.scrape_content = scrape_content
        self.max_content_length = max_content_length
        
        # Setup HTTP session with headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        self.results_keywords = [
            'results', 'outcome', 'efficacy', 'safety', 'response', 'survival',
            'toxicity', 'adverse', 'endpoint', 'analysis', 'findings', 'data',
            'trial results', 'study results', 'interim analysis', 'final analysis',
            'primary endpoint', 'secondary endpoint', 'progression', 'remission',
            'objective response rate', 'overall survival', 'progression-free survival'
        ]
        
        # Remove the duplicate content scraping settings since they're now in __init__
        
    def _rate_limit(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()

    def _scrape_article_content(self, url: str, source: str) -> Dict[str, str]:
        """
        Scrape full content from an article URL.
        Returns dict with 'content', 'summary', and 'scraped_successfully' keys.
        """
        if not self.scrape_content or not url:
            return {'content': '', 'summary': '', 'scraped_successfully': False}
        
        try:
            logger.info(f"Scraping content from {source}: {url[:100]}...")
            content = self._get_page_content(url)
            
            if not content:
                return {'content': '', 'summary': '', 'scraped_successfully': False}
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside", "advertisement"]):
                script.decompose()
            
            # Extract content based on source
            if 'onclive.com' in url:
                content_text = self._extract_onclive_content(soup)
            elif 'enets.org' in url:
                content_text = self._extract_enets_content(soup)
            elif 'annalsofoncology.org' in url:
                content_text = self._extract_annals_content(soup)
            elif 'ascopubs.org' in url or 'asco.org' in url:
                content_text = self._extract_asco_content(soup)
            else:
                # Generic content extraction
                content_text = self._extract_generic_content(soup)
            
            # Clean and limit content
            content_text = re.sub(r'\s+', ' ', content_text.strip())
            
            # Create summary (first 500 characters)
            summary = content_text[:500] + "..." if len(content_text) > 500 else content_text
            
            # Limit full content
            if len(content_text) > self.max_content_length:
                content_text = content_text[:self.max_content_length] + "..."
            
            return {
                'content': content_text,
                'summary': summary,
                'scraped_successfully': True
            }
            
        except Exception as e:
            logger.warning(f"Failed to scrape content from {url}: {e}")
            return {'content': '', 'summary': '', 'scraped_successfully': False}

    def _extract_onclive_content(self, soup: BeautifulSoup) -> str:
        """Extract content specifically from Onclive articles."""
        content_selectors = [
            '.article-content',
            '.content-body',
            '.article-body',
            '.entry-content',
            '[class*="article"]',
            '[class*="content"]',
            'main',
            '.post-content'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                # Remove ads, sidebars, and navigation
                for unwanted in content_elem.select('.ad, .sidebar, .navigation, .related, .comments'):
                    unwanted.decompose()
                
                paragraphs = content_elem.find_all(['p', 'div'], recursive=True)
                text_content = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                
                if len(text_content) > 200:  # Ensure we got substantial content
                    return text_content
        
        # Fallback to generic extraction
        return self._extract_generic_content(soup)

    def _extract_enets_content(self, soup: BeautifulSoup) -> str:
        """Extract content specifically from ENETS articles."""
        content_selectors = [
            '.abstract-content',
            '.abstract-text',
            '.content',
            '.article-content',
            'main'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                return content_elem.get_text(separator=' ', strip=True)
        
        return self._extract_generic_content(soup)

    def _extract_annals_content(self, soup: BeautifulSoup) -> str:
        """Extract content specifically from Annals of Oncology articles."""
        content_selectors = [
            '.article-full-text',
            '.article-body',
            '.fulltext',
            '.abstract',
            '.article-content',
            'main'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                return content_elem.get_text(separator=' ', strip=True)
        
        return self._extract_generic_content(soup)

    def _extract_asco_content(self, soup: BeautifulSoup) -> str:
        """Extract content specifically from ASCO articles."""
        content_selectors = [
            '.abstract-content',
            '.article-content',
            '.fulltext-view',
            '.abstract',
            'main'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                return content_elem.get_text(separator=' ', strip=True)
        
        return self._extract_generic_content(soup)

    def _extract_generic_content(self, soup: BeautifulSoup) -> str:
        """Generic content extraction for any website."""
        # Try common content containers
        content_selectors = [
            'article',
            'main',
            '[role="main"]',
            '.content',
            '.post',
            '.entry',
            '#content',
            '#main'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                text = content_elem.get_text(separator=' ', strip=True)
                if len(text) > 200:
                    return text
        
        # Fallback: get all paragraphs from body
        paragraphs = soup.find_all('p')
        return ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])

    def _get_page_content(self, url: str) -> Optional[str]:
        """Get page content using HTTP requests."""
        self._rate_limit()
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.warning(f"Failed to load {url}: {e}")
            return None

    def _make_request_with_session(self, url: str):
        """Make a request with the session and return response object."""
        self._rate_limit()
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response
        except Exception as e:
            logger.warning(f"Failed to load {url}: {e}")
            return None

    def _has_results_content(self, text: str) -> bool:
        return any(keyword in text.lower() for keyword in self.results_keywords)

    def _calculate_relevance_score(self, text: str, nct_id: str) -> float:
        score = 0.0
        text_lower = text.lower()
        if nct_id.lower() in text_lower:
            score += 10.0
        score += sum(2.0 for keyword in self.results_keywords if keyword in text_lower)
        return score

    def _search_bing(self, nct_id: str, site_domain: str, source_name: str) -> List[Dict]:
        """Search Bing for articles from a specific site domain."""
        articles = []
        
        # Use different search strategies for better results
        if site_domain == 'onclive.com':
            # For onclive, use the simple query that works best
            search_query = f"onclive {nct_id}"
        else:
            # For other sites, use site: prefix
            search_query = f"site:{site_domain} {nct_id}"
        
        bing_url = f"https://www.bing.com/search?q={urllib.parse.quote(search_query)}"
        
        logger.info(f"Searching {source_name} via Bing: {search_query}")
        
        content = self._get_page_content(bing_url)
        if not content:
            return articles
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Parse Bing algorithmic results
        result_items = soup.select('#b_results li.b_algo')
        
        for result in result_items:
            try:
                # Get title and URL from the main link
                title_link = result.select_one('h2 a')
                if not title_link:
                    continue
                    
                title = title_link.get_text(strip=True)
                url = title_link.get('href', '')
                
                # Skip if not from target domain (for site-specific searches)
                if site_domain != 'onclive.com' and site_domain not in url:
                    continue
                elif site_domain == 'onclive.com' and 'onclive.com' not in url:
                    continue
                
                # Get snippet from the result description
                snippet_elem = result.select_one('.b_caption p, .b_caption')
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                
                full_text = f"{title} {snippet}"
                
                # For onclive, be more permissive - include if it's from onclive domain
                # For other sites, require results content or NCT ID
                if site_domain == 'onclive.com':
                    # Include onclive results that contain relevant medical terms or NCT ID
                    is_relevant = (self._has_results_content(full_text) or 
                                 nct_id.lower() in full_text.lower() or
                                 any(term in full_text.lower() for term in ['trial', 'study', 'therapy', 'treatment', 'pfs', 'survival', 'compete']))
                else:
                    # For other sites, require results content or NCT ID
                    is_relevant = self._has_results_content(full_text) or nct_id.lower() in full_text.lower()
                
                if is_relevant:
                    # Scrape full content from the article
                    scraped_data = self._scrape_article_content(url, source_name)
                    
                    article_data = {
                        'title': title,
                        'url': url,
                        'source': f"{source_name} (via Bing)",
                        'abstract_text': snippet,
                        'relevance_score': self._calculate_relevance_score(full_text, nct_id),
                        'full_content': scraped_data['content'],
                        'content_summary': scraped_data['summary'],
                        'content_scraped': scraped_data['scraped_successfully']
                    }
                    
                    articles.append(article_data)
                    
            except Exception as e:
                logger.debug(f"Error processing Bing result: {e}")
                continue
        
        # Remove duplicates while preserving order
        seen_urls = set()
        unique_articles = []
        for article in articles:
            if article['url'] not in seen_urls:
                seen_urls.add(article['url'])
                unique_articles.append(article)
        
        logger.info(f"Found {len(unique_articles)} {source_name} articles via Bing")
        return unique_articles

    def search_congress_abstracts(self, nct_id: str, study_title: str) -> Tuple[bool, Dict]:
        """Search for congress abstracts from key sources."""
        logger.info(f"Searching congress abstracts for {nct_id}")
        
        abstracts = []
        abstracts.extend(self._search_bing(nct_id, 'ascopubs.org', 'ASCO'))
        abstracts.extend(self._search_annals_oncology_direct(nct_id))
        
        # Always search ENETS
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
                            # Scrape full content from the article
                            scraped_data = self._scrape_article_content(url, 'ENETS')
                            
                            abstracts.append({
                                'title': title,
                                'url': url,
                                'source': 'ENETS',
                                'abstract_text': snippet,
                                'relevance_score': self._calculate_relevance_score(full_text, nct_id),
                                'full_content': scraped_data['content'],
                                'content_summary': scraped_data['summary'],
                                'content_scraped': scraped_data['scraped_successfully']
                            })
        except Exception as e:
            logger.warning(f"ENETS search failed: {e}")
        
        return abstracts

    def _search_annals_oncology_direct(self, nct_id: str) -> List[Dict]:
        """Direct search of Annals of Oncology website."""
        articles = []
        try:
            search_url = f"https://www.annalsofoncology.org/action/doSearch?type=quicksearch&text1={urllib.parse.quote(nct_id)}&field1=AllField"
            response = self._make_request_with_session(search_url)
            
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for search results in various containers
                result_containers = soup.select('.search-result, .result-item, .article-item, .search-hit')
                
                for container in result_containers[:3]:  # Limit to first 3 results
                    try:
                        # Get title
                        title_elem = container.select_one('h3 a, .result-title a, .article-title a, h2 a, h4 a')
                        if not title_elem:
                            continue
                            
                        title = title_elem.get_text(strip=True)
                        url = title_elem.get('href', '')
                        
                        # Make URL absolute if needed
                        if url and not url.startswith('http'):
                            url = urllib.parse.urljoin("https://www.annalsofoncology.org/", url)
                        
                        # Get abstract/snippet
                        snippet_elem = container.select_one('.abstract, .summary, .result-summary, p')
                        snippet = snippet_elem.get_text(strip=True)[:200] if snippet_elem else ""
                        
                        full_text = f"{title} {snippet}"
                        if self._has_results_content(full_text) or nct_id.lower() in full_text.lower():
                            # Scrape full content from the article
                            scraped_data = self._scrape_article_content(url, 'Annals of Oncology')
                            
                            articles.append({
                                'title': title,
                                'url': url,
                                'source': 'Annals of Oncology',
                                'abstract_text': snippet,
                                'relevance_score': self._calculate_relevance_score(full_text, nct_id),
                                'full_content': scraped_data['content'],
                                'content_summary': scraped_data['summary'],
                                'content_scraped': scraped_data['scraped_successfully']
                            })
                    except Exception as e:
                        logger.debug(f"Error processing Annals result: {e}")
                        continue
                        
        except Exception as e:
            logger.warning(f"Annals of Oncology search failed: {e}")
        
        logger.info(f"Found {len(articles)} Annals of Oncology articles for {nct_id}")
        return articles

    def search_onclive_enhanced(self, nct_id: str, study_title: str) -> Tuple[bool, Dict]:
        """Search Onclive for relevant articles."""
        logger.info(f"Searching Onclive for {nct_id}")
        
        articles = self._search_bing(nct_id, 'onclive.com', 'Onclive')
        
        result_data = {
            'articles_found': len(articles),
            'articles': sorted(articles, key=lambda x: x.get('relevance_score', 0), reverse=True),
            'search_successful': True
        }
        
        logger.info(f"Found {len(articles)} Onclive articles for {nct_id}")
        return len(articles) > 0, result_data

    def _create_search_result_dict(self, articles: List[Dict]) -> Dict:
        return {
            'publications_found': len(articles),
            'publications': articles,
            'search_successful': len(articles) > 0
        }

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
                
                # For PubMed, we already have the abstract, but we can try to get full text if available
                pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                scraped_data = self._scrape_article_content(pubmed_url, 'PubMed')
                
                publications.append({
                    'pmid': pmid,
                    'title': title,
                    'url': pubmed_url,
                    'has_results_keywords': self._has_results_content(full_text),
                    'relevance_score': self._calculate_relevance_score(full_text, nct_id),
                    'abstract_text': abstract_text[:400] + "..." if len(abstract_text) > 400 else abstract_text,
                    'full_content': scraped_data['content'] if scraped_data['content'] else abstract_text,
                    'content_summary': scraped_data['summary'] if scraped_data['summary'] else abstract_text[:500],
                    'content_scraped': scraped_data['scraped_successfully']
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

    def set_content_scraping(self, enabled: bool, max_length: int = 2000):
        """Enable or disable content scraping functionality."""
        self.scrape_content = enabled
        self.max_content_length = max_length
        logger.info(f"Content scraping {'enabled' if enabled else 'disabled'}")

    def search_for_study_results(self, nct_id: str, study_title: str, progress_callback=None) -> Dict:
        logger.info(f"Comprehensive search for study results: {nct_id}")
        results = {
            'nct_id': nct_id, 'study_title': study_title, 'search_timestamp': time.time(),
            'pubmed': {}, 'congress_abstracts': {}, 'onclive': {}
        }
        
        # Track progress through the search sources
        total_sources = 3
        completed_sources = 0
        
        def update_progress(source_name: str):
            nonlocal completed_sources
            completed_sources += 1
            if progress_callback:
                progress_callback(completed_sources, total_sources, source_name)
        
        try:
            _, results['pubmed'] = self.search_pubmed_enhanced(nct_id, study_title)
            update_progress("PubMed")
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            update_progress("PubMed")
            
        try:
            _, results['congress_abstracts'] = self.search_congress_abstracts(nct_id, study_title)
            update_progress("Congress Abstracts")
        except Exception as e:
            logger.error(f"Congress search failed: {e}")
            update_progress("Congress Abstracts")
            
        try:
            _, results['onclive'] = self.search_onclive_enhanced(nct_id, study_title)
            update_progress("Onclive")
        except Exception as e:
            logger.error(f"Onclive search failed: {e}")
            update_progress("Onclive")
            
        return results

if __name__ == "__main__":
    # Simplified to use HTTP requests only with content scraping enabled
    checker = OnlineResultChecker(rate_limit_delay=2.0, scrape_content=True, max_content_length=1500)
    
    nct_id_example = "NCT03049189"
    title_example = "A Study of 177Lu-Edotreotide Versus Everolimus in GEP-NET (COMPETE)"
    
    print(f"\n--- Checking {nct_id_example} ---")
    results = checker.search_for_study_results(nct_id_example, title_example)
    
    print("\n--- Search Summary ---")
    total_scraped = 0
    total_items = 0
    
    for source, data in results.items():
        if isinstance(data, dict) and 'search_successful' in data:
            count = data.get('articles_found', data.get('publications_found', data.get('abstracts_found', 0)))
            items = data.get('articles') or data.get('publications') or data.get('abstracts')
            
            if items:
                scraped_count = sum(1 for item in items if item.get('content_scraped', False))
                total_scraped += scraped_count
                total_items += len(items)
                
                print(f"{source.capitalize()}: Found {count} results, {scraped_count} scraped.")
                print(f"  - Example: {items[0]['title'][:70]}... ({items[0]['url']})")
                
                if items[0].get('content_scraped'):
                    summary = items[0].get('content_summary', '')
                    print(f"  - Content: {summary[:100]}...")
            else:
                print(f"{source.capitalize()}: Found {count} results.")
    
    print(f"\nOverall: {total_scraped}/{total_items} articles had content successfully scraped.")
    
    # Save results with scraped content
    import json
    with open('example_results_with_content.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Detailed results with scraped content saved to: example_results_with_content.json")