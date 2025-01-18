from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
import numpy as np

class SDXLModel:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['model']['device'])
        self.processor = None
        self.model = None
        
    def load_model(self):
        """Load BLIP model for caption generation"""
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ).to(self.device)
        
    def generate_caption(self, image):
        """Generate detailed caption for the given image"""
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Process image
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        # Generate caption with beam search
        outputs = self.model.generate(
            **inputs,
            max_length=self.config['pipeline']['max_length'],
            num_beams=5,
            num_return_sequences=1,
            temperature=0.7,
            repetition_penalty=1.2,
            length_penalty=1.0
        )
        
        # Decode caption
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        return caption 