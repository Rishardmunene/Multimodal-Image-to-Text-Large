from PIL import Image
import torch
import numpy as np
from torchvision import transforms

def preprocess_image(image_path, config):
    """Enhanced image preprocessing"""
    # Load and convert to RGB
    image = Image.open(image_path).convert('RGB')
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((config['processing']['image_size'], 
                         config['processing']['image_size']),
                        interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms and convert back to PIL
    img_tensor = transform(image)
    img_normalized = transforms.ToPILImage()(img_tensor)
    
    return img_normalized 