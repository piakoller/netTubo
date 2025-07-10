#!/usr/bin/env python3
"""
Clinical Trials Matching System

This package provides functionality for matching patients with relevant
clinical trials using AI-based evaluation.
"""

from .api import ClinicalTrialsAPI, test_api_connection
from .models import ClinicalTrialMatch
from .llm_matcher import LLMStudyMatcher
from .publications import PublicationDownloader
from .matcher import PatientStudyMatcher
from .reports import generate_study_report, generate_summary_report

__all__ = [
    'ClinicalTrialsAPI',
    'ClinicalTrialMatch', 
    'LLMStudyMatcher',
    'PublicationDownloader',
    'PatientStudyMatcher',
    'generate_study_report',
    'generate_summary_report',
    'test_api_connection'
]
