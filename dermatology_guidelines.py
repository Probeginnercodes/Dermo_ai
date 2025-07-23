from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DermatologyGuidelines:
    def __init__(self):
        self.guidelines = self._load_guidelines()

    def _load_guidelines(self) -> List[str]:
        return [
            "ABCDE criteria for melanoma screening: Asymmetry, Border irregularity, Color variation, Diameter >6mm, Evolving characteristics",
            "Any pigmented lesion showing asymmetry should be evaluated by a dermatologist",
            "Irregular, notched, or blurred borders in moles require professional assessment",
            "Rapidly spreading rash with fever requires immediate medical attention",
            "Skin lesions with signs of infection need prompt treatment",
            "Eczema typically presents as itchy, red, dry patches often in flexural areas",
            "Contact dermatitis shows localized reaction pattern matching exposure source",
            "Psoriasis commonly appears as well-demarcated, scaly plaques on extensor surfaces",
            "Cellulitis presents as spreading erythema, warmth, and tenderness",
            "Fungal infections show characteristic ring-like appearance with central clearing",
            "Non-healing ulcers lasting >4-6 weeks require biopsy consideration",
            "Skin lesions that bleed easily or spontaneously need evaluation",
            "New growths in elderly patients should be assessed for malignancy"
        ]
    
    def get_all_guidelines(self) -> List[str]:
        return self.guidelines

    def search_guidelines_by_keyword(self, keyword: str) -> List[str]:
        keyword_lower = keyword.lower()
        return [g for g in self.guidelines if keyword_lower in g.lower()]