import torch
import numpy as np
from PIL import Image
import cv2
from typing import Dict, Any
import logging
from transformers import BlipProcessor, BlipForConditionalGeneration

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.processor = None
        self._load_models()

    def _load_models(self):
        try:
            logger.info("Loading BLIP model for image analysis...")
            model_name = "Salesforce/blip-image-captioning-base"

            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)

            if torch.cuda.is_available() and self.device == 'cuda':
                self.model = self.model.to(self.device)

            self.model.eval()
            logger.info(f"BLIP model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Error loading BLIP model: {e}")
            self.model = None
            self.processor = None

    def process_image(self, image: Image.Image) -> np.ndarray:
        try:
            if self.model is None:
                return np.random.randn(768)

            processed_image = self._preprocess_image(image)
            inputs = self.processor(processed_image, return_tensors="pt")

            if torch.cuda.is_available() and self.device == 'cuda':
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.vision_model(**{k: v for k, v in inputs.items() if k in ['pixel_values']})
                features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]

            return features

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return np.random.randn(768)

    def generate_caption(self, image: Image.Image) -> str:
        try:
            if self.model is None:
                return "Image analysis unavailable - model not loaded"

            prompt = "A detailed medical description of this skin lesion showing"
            processed_image = self._preprocess_image(image)

            inputs = self.processor(processed_image, prompt, return_tensors="pt")

            if torch.cuda.is_available() and self.device == 'cuda':
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=100, num_beams=5, early_stopping=True)
                caption = self.processor.decode(outputs[0], skip_special_tokens=True)

            enhanced_caption = self._enhance_medical_terminology(caption)
            return enhanced_caption

        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return f"Basic image analysis: {self._basic_visual_analysis(image)}"

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')

            max_size = 512
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            return image

        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return image

    def _enhance_medical_terminology(self, caption: str) -> str:
        medical_terms = {
            'spot': 'lesion',
            'mark': 'marking',
            'bump': 'papule',
            'red': 'erythematous',
            'dark': 'hyperpigmented',
            'light': 'hypopigmented'
        }

        enhanced = caption.lower()
        for common, medical in medical_terms.items():
            enhanced = enhanced.replace(common, medical)

        return enhanced.capitalize()

    def _basic_visual_analysis(self, image: Image.Image) -> str:
        try:
            img_array = np.array(image)
            mean_color = img_array.mean(axis=(0, 1))

            if mean_color[0] > 150:
                color_desc = "reddish lesion"
            elif mean_color.mean() < 100:
                color_desc = "dark pigmented lesion"
            else:
                color_desc = "skin lesion"

            return f"Visible {color_desc} requiring medical evaluation"

        except Exception:
            return "Skin lesion visible in image"

    def analyze_image_quality(self, image: Image.Image) -> Dict[str, Any]:
        try:
            img_array = np.array(image)
            sharpness = cv2.Laplacian(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
            brightness = img_array.mean()

            quality_score = 100
            issues = []

            if sharpness < 100:
                issues.append("Image may be blurry")
                quality_score -= 20

            if brightness < 50:
                issues.append("Image is too dark")
                quality_score -= 15
            elif brightness > 200:
                issues.append("Image is too bright")
                quality_score -= 15

            if min(image.size) < 200:
                issues.append("Image resolution is low")
                quality_score -= 25

            quality = "good" if quality_score >= 70 else "fair" if quality_score >= 50 else "poor"

            return {
                "quality": quality,
                "score": max(0, quality_score),
                "sharpness": sharpness,
                "brightness": brightness,
                "resolution": image.size,
                "issues": issues
            }

        except Exception as e:
            logger.error(f"Error analyzing image quality: {e}")
            return {"quality": "unknown", "issues": ["Quality analysis failed"]}
