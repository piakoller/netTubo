#!/usr/bin/env python3
"""
ClinicalTrials.gov API Interface

This module provides a clean interface to the ClinicalTrials.gov API v2.
"""

import logging
import requests
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ClinicalTrialsAPI:
    """Interface to ClinicalTrials.gov API v2."""
    
    BASE_URL = "https://clinicaltrials.gov/api/v2"
    
    def __init__(self, rate_limit_delay: float = 1.0):
        """Initialize with rate limiting."""
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Implement rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def search_studies(self, 
                      condition: str, 
                      intervention: Optional[str] = None,
                      status: Optional[List[str]] = None,
                      phase: Optional[List[str]] = None,
                      max_results: int = 100) -> List[Dict]:
        """
        Search for clinical trials based on condition and other criteria.
        
        Args:
            condition: Medical condition (e.g., "neuroendocrine tumor", "NET")
            intervention: Intervention/treatment (e.g., "PRRT", "everolimus")
            status: Study status (e.g., ["RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED"])
            phase: Study phases (e.g., ["PHASE1", "PHASE2", "PHASE3"])
            max_results: Maximum number of results to return
            
        Returns:
            List of study dictionaries
        """
        self._rate_limit()
        
        # Build query string manually for better control
        query_parts = [f"AREA[Condition]{condition}"]
        
        if intervention:
            query_parts.append(f"AREA[InterventionName]{intervention}")
            
        if status:
            # Convert status to proper format
            status_mapping = {
                "RECRUITING": "Recruiting",
                "ACTIVE_NOT_RECRUITING": "Active, not recruiting", 
                "COMPLETED": "Completed",
                "TERMINATED": "Terminated"
            }
            mapped_statuses = [status_mapping.get(s, s) for s in status]
            status_query = " OR ".join([f'AREA[OverallStatus]"{s}"' for s in mapped_statuses])
            query_parts.append(f"({status_query})")
        
        query_string = " AND ".join(query_parts)
        
        # Build parameters with simplified approach
        params = {
            "format": "json",
            "pageSize": min(max_results, 1000),
            "query.term": query_string
        }
        
        try:
            logger.info(f"Searching with query: {query_string}")
            response = requests.get(f"{self.BASE_URL}/studies", params=params, timeout=30)
            
            # Log the actual URL for debugging
            logger.debug(f"Request URL: {response.url}")
            
            if response.status_code == 400:
                logger.warning(f"Bad request for condition '{condition}'. Trying simpler query...")
                # Fallback to simpler query
                simple_params = {
                    "format": "json",
                    "pageSize": min(max_results, 1000),
                    "query.term": condition
                }
                response = requests.get(f"{self.BASE_URL}/studies", params=simple_params, timeout=30)
            
            response.raise_for_status()
            
            data = response.json()
            studies = data.get("studies", [])
            
            logger.info(f"Found {len(studies)} studies for condition: {condition}")
            return studies
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching studies: {e}")
            # Try one more fallback with basic search
            try:
                logger.info(f"Trying basic search for: {condition}")
                basic_params = {
                    "format": "json", 
                    "pageSize": min(max_results, 100),
                    "query.term": f'"{condition}"'
                }
                response = requests.get(f"{self.BASE_URL}/studies", params=basic_params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    studies = data.get("studies", [])
                    logger.info(f"Basic search found {len(studies)} studies")
                    return studies
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
            
            return []
    
    def get_study_details(self, nct_id: str) -> Optional[Dict]:
        """Get detailed information for a specific study."""
        self._rate_limit()
        
        try:
            response = requests.get(f"{self.BASE_URL}/studies/{nct_id}", 
                                  params={"format": "json"}, 
                                  timeout=30)
            response.raise_for_status()
            
            data = response.json()
            study = data.get("protocolSection", {})
            
            return {"protocolSection": study}
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting study details for {nct_id}: {e}")
            return None


def test_api_connection():
    """Test if the ClinicalTrials.gov API is working."""
    api = ClinicalTrialsAPI(rate_limit_delay=1.0)
    
    print("Testing ClinicalTrials.gov API connection...")
    
    # Test with a simple, common condition
    test_studies = api.search_studies("cancer", max_results=5)
    
    if test_studies:
        print(f"✅ API connection successful! Found {len(test_studies)} studies for 'cancer'")
        
        # Show first study as example
        if test_studies:
            first_study = test_studies[0]
            protocol = first_study.get("protocolSection", {})
            identification = protocol.get("identificationModule", {})
            print(f"Example study: {identification.get('briefTitle', 'No title')}")
            print(f"NCT ID: {identification.get('nctId', 'No ID')}")
        return True
    else:
        print("❌ API connection failed!")
        return False
