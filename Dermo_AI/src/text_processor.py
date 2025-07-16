import numpy as np
from typing import Dict, Any
import logging
import re
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.medical_keywords = self._load_medical_keywords()
        self._load_models()

    def _load_models(self):
        try:
            logger.info("Loading sentence transformer for text processing...")
            model_name = 'all-MiniLM-L6-v2'

            self.model = SentenceTransformer(model_name)

            if self.device == 'cuda' and torch.cuda.is_available():
                self.model = self.model.to(self.device)

            logger.info(f"Sentence transformer loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Error loading sentence transformer: {e}")
            self.model = None

    def _load_medical_keywords(self) -> Dict[str, list]:
        return {
            'skin_conditions': [
                'rash', 'lesion', 'spot', 'mole', 'bump', 'growth', 'patch',
                'discoloration', 'scaling', 'peeling', 'cracking', 'blistering'
            ],
            'symptoms': [
                'itching', 'burning', 'pain', 'tenderness', 'swelling',
                'redness', 'warmth', 'numbness', 'tingling'
            ],
            'temporal': [
                'sudden', 'gradual', 'recent', 'chronic', 'acute',
                'days', 'weeks', 'months', 'years'
            ],
            'severity': [
                'mild', 'moderate', 'severe', 'intense', 'slight',
                'worsening', 'improving', 'stable'
            ]
        }

    def extract_features(self, text: str) -> np.ndarray:
        try:
            if self.model is None:
                return np.random.randn(384)

            processed_text = self._preprocess_text(text)
            embeddings = self.model.encode(processed_text, convert_to_numpy=True)

            return embeddings

        except Exception as e:
            logger.error(f"Error extracting text features: {e}")
            return np.random.randn(384)

    def _preprocess_text(self, text: str) -> str:
        try:
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            text = re.sub(r'\s+', ' ', text).strip()
            text = self._standardize_medical_terms(text)
            return text

        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return text

    def _standardize_medical_terms(self, text: str) -> str:
        replacements = {
            'itchy': 'pruritic',
            'rash': 'skin eruption',
            'bump': 'papule',
            'red': 'erythematous',
            'swollen': 'edematous'
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    def analyze_symptoms(self, text: str) -> Dict[str, Any]:
        try:
            text_lower = text.lower()
            analysis = {}

            for category, keywords in self.medical_keywords.items():
                found_keywords = [kw for kw in keywords if kw in text_lower]
                analysis[category] = found_keywords

            urgent_keywords = ['sudden', 'severe', 'intense', 'bleeding', 'spreading']
            urgency_score = sum(1 for kw in urgent_keywords if kw in text_lower)

            if urgency_score >= 2:
                urgency = 'high'
            elif urgency_score == 1:
                urgency = 'medium'
            else:
                urgency = 'low'

            analysis['urgency'] = urgency
            analysis['urgency_score'] = urgency_score

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing symptoms: {e}")
            return {'urgency': 'unknown'}