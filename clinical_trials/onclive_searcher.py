#!/usr/bin/env python3
"""
Simplified Onclive Searcher for Clinical Trials

This script searches Google for "onclive" + NCT number and extracts onclive.com results.
Uses simple HTTP requests instead of browser automation.
"""

import requests
from bs4 import BeautifulSoup
import urllib.parse
import logging
import time
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OncliveSearcher:
    """
    Simple searcher for Onclive.com clinical trial content using HTTP requests.
    """
    
    def __init__(self, nct_number: str):
        self.nct_number = nct_number
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
    
    def search_bing_for_onclive(self):
        """Search Bing for 'onclive NCT_NUMBER' and extract content from onclive.com results."""
        logger.info(f"Searching Bing for: onclive {self.nct_number}")
        
        try:
            # Construct Bing search URL  
            query = f'onclive {self.nct_number}'
            bing_url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}"
            
            # Make request with delay to be respectful
            time.sleep(1)
            response = self.session.get(bing_url, timeout=30)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract onclive URLs from Bing results
            onclive_urls = self._extract_onclive_urls_from_bing(soup)
            
            if not onclive_urls:
                logger.warning("No onclive.com URLs found in Bing results")
                return []
            
            # Get content from each onclive URL
            all_results = []
            for url in onclive_urls[:5]:  # Limit to first 5 results
                try:
                    logger.info(f"Extracting content from: {url}")
                    content = self._extract_content_from_onclive_page(url)
                    if content:
                        all_results.append(content)
                    time.sleep(2)  # Be respectful to the server
                except Exception as e:
                    logger.error(f"Failed to extract content from {url}: {e}")
                    continue
            
            logger.info(f"Extracted content from {len(all_results)} onclive pages")
            return all_results
            
        except Exception as e:
            logger.error(f"Bing search failed: {e}")
            return []
    
    def _extract_onclive_urls_from_bing(self, soup):
        """Extract onclive.com URLs from Bing search results."""
        onclive_urls = []
        
        # Find links in Bing search results
        links = soup.find_all('a', href=True)
        
        for link in links:
            href = link.get('href', '')
            
            # Bing URLs are usually clean or have minimal prefixes
            if 'onclive.com' in href:
                # Clean up the URL
                if href.startswith('http'):
                    clean_url = href
                else:
                    continue
                    
                if clean_url not in onclive_urls:
                    onclive_urls.append(clean_url)
                    logger.debug(f"Found onclive URL: {clean_url}")
        
        logger.info(f"Found {len(onclive_urls)} unique onclive URLs in Bing results")
        return onclive_urls
    
    def _extract_content_from_onclive_page(self, url):
        """Extract content from an individual onclive page."""
        try:
            time.sleep(1)  # Be respectful
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_elem = soup.select_one('h1, .article-title, .post-title, title')
            title = title_elem.get_text(strip=True) if title_elem else "No title found"
            
            # Extract main content
            content_selectors = [
                'article',
                '.article-content',
                '.post-content', 
                '.content',
                'main',
                '.main-content',
                '#content'
            ]
            
            content_text = ""
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Remove script and style elements
                    for script in content_elem(["script", "style"]):
                        script.decompose()
                    
                    content_text = content_elem.get_text(separator=' ', strip=True)
                    break
            
            # If no main content found, get all paragraphs
            if not content_text:
                paragraphs = soup.select('p')
                content_text = ' '.join([p.get_text(strip=True) for p in paragraphs[:10]])
            
            # Check if content is relevant (contains NCT number)
            if self.nct_number.lower() in content_text.lower() or self.nct_number.upper() in content_text:
                # Extract relevant excerpts (paragraphs containing NCT number)
                relevant_excerpts = []
                for p in soup.select('p'):
                    p_text = p.get_text(strip=True)
                    if self.nct_number.lower() in p_text.lower() or self.nct_number.upper() in p_text:
                        relevant_excerpts.append(p_text)
                
                # Combine excerpts or use first 500 chars of content
                snippet = ' '.join(relevant_excerpts)[:500] if relevant_excerpts else content_text[:500]
                
                return {
                    'title': title,
                    'url': url,
                    'snippet': snippet,
                    'method': 'Bing Search + HTTP Request',
                    'source': 'Onclive',
                    'full_content_length': len(content_text),
                    'relevant_excerpts_count': len(relevant_excerpts)
                }
            else:
                logger.info(f"No relevant content found in {url}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to extract content from {url}: {e}")
            return None
    
    def search(self):
        """Perform the search and return results."""
        logger.info(f"Starting Onclive search for: {self.nct_number}")
        return self.search_bing_for_onclive()
    
    def print_results(self, results):
        """Print formatted results."""
        print(f"\n{'='*60}")
        print(f"ONCLIVE SEARCH RESULTS FOR {self.nct_number}")
        print(f"{'='*60}")
        
        if not results:
            print("No results found.")
            return
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['title']}")
            print(f"   URL: {result['url']}")
            print(f"   Method: {result['method']}")
            if result['snippet']:
                print(f"   Snippet: {result['snippet']}")
            
            # Print additional info if available
            if 'full_content_length' in result:
                print(f"   Content Length: {result['full_content_length']} characters")
            if 'relevant_excerpts_count' in result:
                print(f"   Relevant Excerpts: {result['relevant_excerpts_count']}")
            print()


def main():
    """Main execution function."""
    # Configuration
    NCT_NUMBER = "NCT03049189"
    
    # Initialize searcher
    searcher = OncliveSearcher(NCT_NUMBER)
    
    try:
        # Perform search
        results = searcher.search()
        
        # Display results
        searcher.print_results(results)
        
        # Save results to file
        import json
        with open(f'onclive_results_{NCT_NUMBER}.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to onclive_results_{NCT_NUMBER}.json")
        
    except Exception as e:
        logger.error(f"Search failed: {e}")


if __name__ == "__main__":
    main()
