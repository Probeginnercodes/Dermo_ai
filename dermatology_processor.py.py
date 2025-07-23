import numpy as np
from typing import Dict, List, Any, Optional
import logging
from image_processor import ImageProcessor
from audio_processor import AudioProcessor
from text_processor import TextProcessor
from fusion_layer import AttentionFusionLayer
from dermatology_guidelines import DermatologyGuidelines

logger = logging.getLogger(__name__)

class MultimodalDermatologyProcessor:
    def __init__(self, device='cpu'):  # Changed from 'cuda' to 'cpu'
        """Initialize all processor components with specified device"""
        self.device = device
        self.image_processor = ImageProcessor(device)
        self.audio_processor = AudioProcessor(device)
        self.text_processor = TextProcessor(device)
        self.fusion_layer = AttentionFusionLayer(device=device)
        self.guidelines = DermatologyGuidelines()

        logger.info(f"Multimodal processor initialized on {device.upper()}")

    def process_case(self, image=None, audio=None, text_data=None, patient_info=None) -> Dict[str, Any]:
        """Main processing pipeline for dermatology cases"""
        try:
            results = {
                'image_analysis': self._process_image(image) if image else None,
                'audio_analysis': self._process_audio(audio) if audio else None,
                'text_analysis': self._process_text(text_data, patient_info) if text_data else None,
                'fusion_results': None,
                'clinical_assessment': None,
                'recommendations': [],
                'urgency': 'unknown',
                'confidence': 0.0,
                'error': False
            }

            # Only proceed with fusion if at least one modality succeeded
            if any([results['image_analysis'], results['audio_analysis'], results['text_analysis']]):
                results.update({
                    'fusion_results': self._perform_fusion(results),
                    'clinical_assessment': self._generate_clinical_assessment(results),
                    'recommendations': self._generate_recommendations(results),
                    'urgency': self._assess_urgency(results),
                    'confidence': self._calculate_confidence(results)
                })

            return results

        except Exception as e:
            logger.exception(f"Case processing failed: {str(e)}")
            return self._generate_error_response(str(e))

    def _process_image(self, image) -> Dict[str, Any]:
        """Process skin lesion image"""
        try:
            return {
                'features': self.image_processor.process_image(image),
                'caption': self.image_processor.generate_caption(image),
                'quality': self.image_processor.analyze_image_quality(image),
                'available': True
            }
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            return {'available': False, 'error': str(e)}

    def _process_audio(self, audio) -> Dict[str, Any]:
        """Process patient audio description"""
        try:
            return {
                'transcript': self.audio_processor.transcribe_audio(audio),
                'features': self.audio_processor.extract_features(audio),
                'quality': self.audio_processor.analyze_audio_quality(audio),
                'available': True
            }
        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}")
            return {'available': False, 'error': str(e)}

    def _process_text(self, text_data, patient_info=None) -> Dict[str, Any]:
        """Process textual symptom description"""
        try:
            combined_text = f"{self._format_patient_info(patient_info)} {text_data}" if patient_info else text_data
            return {
                'features': self.text_processor.extract_features(combined_text),
                'symptom_analysis': self.text_processor.analyze_symptoms(combined_text),
                'combined_text': combined_text,
                'available': True
            }
        except Exception as e:
            logger.error(f"Text processing failed: {str(e)}")
            return {'available': False, 'error': str(e)}

    def _perform_fusion(self, results) -> Dict[str, Any]:
        """Fuse multimodal features using attention"""
        try:
            modalities = {
                'image': results['image_analysis']['features'] if results['image_analysis'] else None,
                'text': results['text_analysis']['features'] if results['text_analysis'] else None,
                'audio': results['audio_analysis']['features'] if results['audio_analysis'] else None
            }

            # Ensure at least one modality exists
            if not any(modalities.values()):
                return {'fusion_successful': False, 'reason': 'No valid modalities'}

            # Fill missing modalities with noise if needed
            for mod in ['image', 'text']:
                if modalities[mod] is None:
                    modalities[mod] = np.random.randn(384) * 0.1
                    logger.warning(f"Using synthetic {mod} features")

            return {
                'fused_features': self.fusion_layer.fuse_modalities(**modalities),
                'fusion_successful': True
            }
        except Exception as e:
            logger.error(f"Fusion failed: {str(e)}")
            return {'fusion_successful': False, 'error': str(e)}

    def _generate_clinical_assessment(self, results) -> Dict[str, Any]:
        """Generate clinical interpretation"""
        findings = []
        
        # Image findings
        if results['image_analysis']:
            findings.append(f"Visual: {results['image_analysis']['caption']}")
        
        # Audio findings
        if results['audio_analysis']:
            findings.append(f"Patient report: {results['audio_analysis']['transcript']}")
        
        # Text findings
        if results['text_analysis']:
            findings.extend(f"Reported: {s}" for s in results['text_analysis']['symptom_analysis'].get('symptoms', []))

        return {
            'primary_findings': findings,
            'differential_diagnoses': self.guidelines.get_differential_diagnoses(findings),
            'clinical_description': '. '.join(findings) if findings else "Insufficient clinical data"
        }

    def _generate_recommendations(self, results) -> List[str]:
        """Generate patient-specific recommendations"""
        urgency = results['urgency']
        return self.guidelines.get_recommendations(
            urgency=urgency,
            findings=results['clinical_assessment']['primary_findings']
        )

    def _assess_urgency(self, results) -> str:
        """Determine clinical urgency"""
        score = 0
        
        # Text urgency signals
        if results['text_analysis']:
            score += 3 if 'pain' in results['text_analysis']['combined_text'].lower() else 1
        
        # Image concerning features
        if results['image_analysis']:
            concerning_terms = ['bleeding', 'ulcer', 'irregular']
            score += sum(1 for term in concerning_terms 
                        if term in results['image_analysis']['caption'].lower())
        
        return 'high' if score >= 4 else 'medium' if score >= 2 else 'low'

    def _calculate_confidence(self, results) -> float:
        """Calculate overall confidence score"""
        scores = []
        if results['image_analysis']:
            scores.append(results['image_analysis']['quality'].get('score', 0.5))
        if results['text_analysis']:
            scores.append(0.7)  # Base confidence for text
        if results['audio_analysis']:
            scores.append(results['audio_analysis']['quality'].get('score', 0.5))
        
        return round(sum(scores)/len(scores), 2) if scores else 0.3

    def _generate_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Generate error response template"""
        return {
            'error': True,
            'message': f"System error: {error_msg}",
            'recommendations': [
                "Please try again",
                "Contact support if error persists"
            ],
            'urgency': 'medium',
            'confidence': 0.0
        }

    def _format_patient_info(self, info: Dict) -> str:
        """Format patient metadata for text processing"""
        return f"Patient: {info.get('age', 'unknown')}yo {info.get('gender', 'unspecified')}. History: {info.get('medical_history', 'none')}."