#!/usr/bin/env python3
"""
Data models for clinical trials matching system.

This module contains dataclasses and model definitions.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class ClinicalTrialMatch:
    """Represents a clinical trial match for a patient."""
    nct_id: str
    title: str
    status: str
    phase: str
    condition: str
    intervention: str
    brief_summary: str
    detailed_description: str
    eligibility_criteria: str
    start_date: str
    completion_date: str
    primary_outcome: str
    secondary_outcome: str
    sponsor: str
    relevance_score: float
    relevance_reason: str
    locations: List[str]
    url: str
    publications: Optional[Dict] = None  # Will contain publication information
