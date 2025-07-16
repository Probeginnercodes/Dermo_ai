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
    def __init__(self, device='cuda'):
        self.device = device
        self.image_processor = ImageProcessor(device)
        self.audio_processor = AudioProcessor(device)
        self.text_processor = TextProcessor(device)
        self.fusion_layer = AttentionFusionLayer(device=device)
        self.guidelines = DermatologyGuidelines()

        logger.info("Multimodal dermatology processor initialized")

    def process_case(self, image=None, audio=None, text_data=None, patient_info=None) -> Dict[str, Any]:
        try:
            results = {
                'image_analysis': None,
                'audio_analysis': None,
                'text_analysis': None,
                'fusion_results': None,
                'clinical_assessment': None,
                'recommendations': [],
                'urgency': 'unknown',
                'confidence': 0.0
            }

            if image is not None:
                results['image_analysis'] = self._process_image(image)

            if audio is not None:
                results['audio_analysis'] = self._process_audio(audio)

            if text_data is not None:
                results['text_analysis'] = self._process_text(text_data, patient_info)

            if any([results['image_analysis'], results['audio_analysis'], results['text_analysis']]):
                results['fusion_results'] = self._perform_fusion(results)
                results['clinical_assessment'] = self._generate_clinical_assessment(results)
                results['recommendations'] = self._generate_recommendations(results)
                results['urgency'] = self._assess_urgency(results)
                results['confidence'] = self._calculate_confidence(results)

            return results

        except Exception as e:
            logger.error(f"Error processing case: {e}")
            return self._generate_error_response(str(e))

    def _process_image(self, image) -> Dict[str, Any]:
        try:
            features = self.image_processor.process_image(image)
            caption = self.image_processor.generate_caption(image)
            quality = self.image_processor.analyze_image_quality(image)

            return {
                'features': features,
                'caption': caption,
                'quality': quality,
                'available': True
            }

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {'available': False, 'error': str(e)}

    def _process_audio(self, audio) -> Dict[str, Any]:
        try:
            transcript = self.audio_processor.transcribe_audio(audio)
            features = self.audio_processor.extract_features(audio)
            quality = self.audio_processor.analyze_audio_quality(audio)

            return {
                'transcript': transcript,
                'features': features,
                'quality': quality,
                'available': True
            }

        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return {'available': False, 'error': str(e)}

    def _process_text(self, text_data, patient_info=None) -> Dict[str, Any]:
        try:
            combined_text = text_data
            if patient_info:
                patient_text = f"Age: {patient_info.get('age', 'unknown')}, Gender: {patient_info.get('gender', 'unknown')}"
                combined_text = f"{patient_text}. {text_data}"

            features = self.text_processor.extract_features(combined_text)
            symptom_analysis = self.text_processor.analyze_symptoms(combined_text)

            return {
                'features': features,
                'symptom_analysis': symptom_analysis,
                'combined_text': combined_text,
                'available': True
            }

        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return {'available': False, 'error': str(e)}

    def _perform_fusion(self, results) -> Dict[str, Any]:
        try:
            image_features = None
            text_features = None
            audio_features = None

            if results['image_analysis'] and results['image_analysis'].get('available'):
                image_features = results['image_analysis']['features']

            if results['text_analysis'] and results['text_analysis'].get('available'):
                text_features = results['text_analysis']['features']

            if results['audio_analysis'] and results['audio_analysis'].get('available'):
                audio_features = results['audio_analysis']['features']

            if sum([f is not None for f in [image_features, text_features, audio_features]]) >= 1:
                if image_features is None:
                    image_features = np.random.randn(384) * 0.1
                if text_features is None:
                    text_features = np.random.randn(384) * 0.1

                fused_features = self.fusion_layer.fuse_modalities(
                    image_features, text_features, audio_features
                )

                return {
                    'fused_features': fused_features,
                    'fusion_successful': True
                }

            return {'fusion_successful': False, 'reason': 'Insufficient modalities'}

        except Exception as e:
            logger.error(f"Error in fusion: {e}")
            return {'fusion_successful': False, 'error': str(e)}

    def _generate_clinical_assessment(self, results) -> Dict[str, Any]:
        assessment = {
            'primary_findings': [],
            'differential_diagnoses': [],
            'clinical_description': ''
        }

        if results['image_analysis'] and results['image_analysis'].get('available'):
            assessment['primary_findings'].append(
                f"Visual: {results['image_analysis']['caption']}"
            )

        if results['audio_analysis'] and results['audio_analysis'].get('available'):
            assessment['primary_findings'].append(
                f"Patient report: {results['audio_analysis']['transcript']}"
            )

        if results['text_analysis'] and results['text_analysis'].get('available'):
            symptoms = results['text_analysis']['symptom_analysis']
            if symptoms.get('skin_conditions'):
                assessment['primary_findings'].extend(
                    [f"Reported: {cond}" for cond in symptoms['skin_conditions']]
                )

        assessment['differential_diagnoses'] = [
            {'condition': 'Contact dermatitis', 'likelihood': 'moderate'},
            {'condition': 'Eczema', 'likelihood': 'moderate'},
            {'condition': 'Fungal infection', 'likelihood': 'low'}
        ]

        assessment['clinical_description'] = '. '.join(assessment['primary_findings']) if assessment['primary_findings'] else "Limited clinical information available"

        return assessment

    def _generate_recommendations(self, results) -> List[str]:
        recommendations = []
        urgency = self._assess_urgency(results)

        if urgency == 'high':
            recommendations.extend([
                "Seek immediate medical attention",
                "Do not delay evaluation by a healthcare provider"
            ])
        elif urgency == 'medium':
            recommendations.extend([
                "Schedule appointment with dermatologist within 1-2 weeks",
                "Monitor for changes in size, color, or symptoms"
            ])
        else:
            recommendations.extend([
                "Consider routine dermatologic consultation",
                "Practice good skin hygiene"
            ])

        return recommendations

    def _assess_urgency(self, results) -> str:
        urgency_score = 0

        if results['text_analysis'] and results['text_analysis'].get('available'):
            symptoms = results['text_analysis']['symptom_analysis']
            text_urgency = symptoms.get('urgency', 'low')

            if text_urgency == 'high':
                urgency_score += 3
            elif text_urgency == 'medium':
                urgency_score += 2
            else:
                urgency_score += 1

        if results['image_analysis'] and results['image_analysis'].get('available'):
            caption = results['image_analysis']['caption'].lower()
            concerning_terms = ['bleeding', 'ulcerated', 'irregular', 'asymmetric']
            concern_count = sum(1 for term in concerning_terms if term in caption)
            urgency_score += concern_count

        if urgency_score >= 4:
            return 'high'
        elif urgency_score >= 2:
            return 'medium'
        else:
            return 'low'

    def _calculate_confidence(self, results) -> float:
        confidence_factors = []

        if results['image_analysis'] and results['image_analysis'].get('available'):
            quality = results['image_analysis']['quality']
            confidence_factors.append(quality.get('score', 50) / 100)

        if results['text_analysis'] and results['text_analysis'].get('available'):
            confidence_factors.append(0.7)

        if results['audio_analysis'] and results['audio_analysis'].get('available'):
            quality = results['audio_analysis']['quality']
            confidence_factors.append(quality.get('score', 50) / 100)

        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.3

    def _generate_error_response(self, error_msg: str) -> Dict[str, Any]:
        return {
            'error': True,
            'message': error_msg,
            'recommendations': ["Please try again or consult with healthcare provider"],
            'urgency': 'medium',
            'confidence': 0.0
        }