#!/usr/bin/env python3
"""
LLM-based study matching for clinical trials.

This module handles the AI evaluation of study relevance for patients.
"""

import logging
import requests
import time
import re
from typing import Dict, Tuple
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from shared_logic import format_patient_data_for_prompt, PATIENT_FIELDS_FOR_PROMPT

logger = logging.getLogger(__name__)


class LLMStudyMatcher:
    """Uses an LLM to match patients with clinical studies."""
    
    def __init__(self, api_key: str, model: str, base_url: str):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        
    def call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Call the LLM API with error handling and retries."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/piakoller/netTubo",
            "X-Title": "NET Clinical Trials Matcher",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.1
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    response_json = response.json()
                    if "choices" in response_json and len(response_json["choices"]) > 0:
                        return response_json["choices"][0]["message"]["content"]
                else:
                    logger.warning(f"API returned status {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"LLM API request failed (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return "ERROR: LLM API call failed after all retries"
    
    def evaluate_study_relevance(self, patient_data: Dict, study_info: Dict) -> Tuple[float, str]:
        """
        Use LLM to evaluate if a study is relevant for a patient.
        
        Returns:
            Tuple of (relevance_score, explanation)
        """
        # Format patient data using only the fields specified in PATIENT_FIELDS_FOR_PROMPT
        patient_data_formatted = format_patient_data_for_prompt(patient_data, PATIENT_FIELDS_FOR_PROMPT)
        
        # Extract study information
        protocol = study_info.get("protocolSection", {})
        identification = protocol.get("identificationModule", {})
        
        study_title = identification.get("briefTitle", "")
        # Get only the brief summary from descriptionModule
        description_module = protocol.get("descriptionModule", {})
        study_summary = description_module.get("briefSummary", "")
        
        # Get basic study information for context
        design_module = protocol.get("designModule", {})
        study_phases = "; ".join(design_module.get("phases", []))
        study_type = design_module.get("studyType", "")
        
        # Get status information
        status_module = protocol.get("statusModule", {})
        study_status = status_module.get("overallStatus", "")
        
        # Create prompt for LLM evaluation
        prompt = f"""
Du bist Facharzt für Onkologie, Facharzt für Endokrinologie und Facharzt für Nuklearmedizin. Bewerte, ob die folgende klinische Studie für den Patienten relevant sein könnte.

{patient_data_formatted}

STUDIE INFORMATION:
- Titel: {study_title}
- Status: {study_status}
- Studientyp: {study_type}
- Phase: {study_phases}
- Kurzzusammenfassung: {study_summary}

WICHTIGER HINWEIS:
Der Patient hat eine bestimmte Diagnose, aber die konkrete Behandlungsmethode ist noch NICHT festgelegt. Die Behandlung soll erst in einem zweiten Schritt basierend auf den gefundenen relevanten Studien entschieden werden.

WICHTIG: Bewerte die Relevanz AUSSCHLIESSLICH basierend auf der obigen Kurzzusammenfassung der Studie. Berücksichtige keine anderen Details oder Vermutungen über die Studie.

AUFGABE:
Bewerte die Relevanz dieser Studie für den Patienten auf einer Skala von 0.0 bis 1.0, wobei:
- 0.0 = Völlig irrelevant (andere Erkrankung, völlig unpassend)
- 0.3 = Teilweise relevant (ähnliche Erkrankung oder verwandte Indikation)
- 0.6 = Relevant (passende Erkrankung, Studie könnte für Behandlungsentscheidung hilfreich sein)
- 0.9 = Hochrelevant (sehr gut passende Erkrankung und Patientenprofil)
- 1.0 = Ideal geeignet (perfekte Übereinstimmung mit Diagnose und Patientencharakteristika)

Berücksichtige HAUPTSÄCHLICH:
1. Übereinstimmung der Diagnose/Erkrankung mit der Studienindikation (basierend auf der Kurzzusammenfassung)
2. Relevanz der Studie für die Behandlungsentscheidung (basierend auf der Kurzzusammenfassung)
3. Stadium/Schweregrad der Erkrankung (falls in der Kurzzusammenfassung erkennbar)

Antworte im folgenden Format:
RELEVANZ_SCORE: [0.0-1.0]
BEGRÜNDUNG: [Detaillierte Erklärung der Relevanz basierend auf der Kurzzusammenfassung und dem Potenzial für Behandlungsentscheidung]
"""

        try:
            llm_response = self.call_llm(prompt)
            
            if "ERROR:" in llm_response:
                logger.error(f"LLM evaluation failed: {llm_response}")
                return 0.0, "LLM evaluation failed"
            
            # Parse LLM response
            relevance_score = 0.0
            explanation = "No explanation provided"
            
            lines = llm_response.split('\n')
            for line in lines:
                if line.startswith("RELEVANZ_SCORE:"):
                    try:
                        score_text = line.split(":", 1)[1].strip()
                        relevance_score = float(score_text)
                        relevance_score = max(0.0, min(1.0, relevance_score))  # Clamp to 0-1
                    except (ValueError, IndexError):
                        logger.warning(f"Could not parse relevance score from: {line}")
                        
                elif line.startswith("BEGRÜNDUNG:"):
                    explanation = line.split(":", 1)[1].strip()
            
            # If no structured response, try to extract score from text
            if relevance_score == 0.0 and explanation == "No explanation provided":
                score_match = re.search(r'(\d+\.?\d*)', llm_response)
                if score_match:
                    try:
                        relevance_score = float(score_match.group(1))
                        if relevance_score > 1.0:
                            relevance_score = relevance_score / 10.0  # Convert 8.5 to 0.85
                        relevance_score = max(0.0, min(1.0, relevance_score))
                    except ValueError:
                        pass
                explanation = llm_response[:200] + "..." if len(llm_response) > 200 else llm_response
            
            return relevance_score, explanation
            
        except Exception as e:
            logger.error(f"Error in LLM evaluation: {e}")
            return 0.0, f"Evaluation error: {str(e)}"
    
    def generate_evaluation_prompt(self, patient_data: Dict, study_info: Dict) -> str:
        """
        Generate the evaluation prompt without calling the LLM (for testing purposes).
        
        Returns:
            The prompt string that would be sent to the LLM
        """
        # Format patient data using only the fields specified in PATIENT_FIELDS_FOR_PROMPT
        patient_data_formatted = format_patient_data_for_prompt(patient_data, PATIENT_FIELDS_FOR_PROMPT)
        
        # Extract study information
        protocol = study_info.get("protocolSection", {})
        identification = protocol.get("identificationModule", {})
        
        study_title = identification.get("briefTitle", "")
        # Get only the brief summary from descriptionModule
        description_module = protocol.get("descriptionModule", {})
        study_summary = description_module.get("briefSummary", "")
        
        # Get basic study information for context
        design_module = protocol.get("designModule", {})
        study_phases = "; ".join(design_module.get("phases", []))
        study_type = design_module.get("studyType", "")
        
        # Get status information
        status_module = protocol.get("statusModule", {})
        study_status = status_module.get("overallStatus", "")
        
        # Create prompt for LLM evaluation
        prompt = f"""
Du bist Facharzt für Onkologie, Facharzt für Endokrinologie und Facharzt für Nuklearmedizin. Bewerte, ob die folgende klinische Studie für den Patienten relevant sein könnte.

{patient_data_formatted}

STUDIE INFORMATION:
- Titel: {study_title}
- Status: {study_status}
- Studientyp: {study_type}
- Phase: {study_phases}
- Kurzzusammenfassung: {study_summary}

WICHTIGER HINWEIS:
Der Patient hat eine bestimmte Diagnose, aber die konkrete Behandlungsmethode ist noch NICHT festgelegt. Die Behandlung soll erst in einem zweiten Schritt basierend auf den gefundenen relevanten Studien entschieden werden.

WICHTIG: Bewerte die Relevanz AUSSCHLIESSLICH basierend auf der obigen Kurzzusammenfassung der Studie. Berücksichtige keine anderen Details oder Vermutungen über die Studie.

AUFGABE:
Bewerte die Relevanz dieser Studie für den Patienten auf einer Skala von 0.0 bis 1.0, wobei:
- 0.0 = Völlig irrelevant (andere Erkrankung, völlig unpassend)
- 0.3 = Teilweise relevant (ähnliche Erkrankung oder verwandte Indikation)
- 0.6 = Relevant (passende Erkrankung, Studie könnte für Behandlungsentscheidung hilfreich sein)
- 0.9 = Hochrelevant (sehr gut passende Erkrankung und Patientenprofil)
- 1.0 = Ideal geeignet (perfekte Übereinstimmung mit Diagnose und Patientencharakteristika)

Berücksichtige HAUPTSÄCHLICH:
1. Übereinstimmung der Diagnose/Erkrankung mit der Studienindikation (basierend auf der Kurzzusammenfassung)
2. Relevanz der Studie für die Behandlungsentscheidung (basierend auf der Kurzzusammenfassung)
3. Stadium/Schweregrad der Erkrankung (falls in der Kurzzusammenfassung erkennbar)

Antworte im folgenden Format:
RELEVANZ_SCORE: [0.0-1.0]
BEGRÜNDUNG: [Detaillierte Erklärung der Relevanz basierend auf der Kurzzusammenfassung und dem Potenzial für Behandlungsentscheidung]
"""
        
        return prompt
