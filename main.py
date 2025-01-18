import yaml
import os
from pycocotools.coco import COCO
import urllib.request
import subprocess
from tqdm import tqdm
import json
from src.pipeline.caption_pipeline import CaptionPipeline
from nltk.translate.bleu_score import sentence_bleu
import nltk
import numpy as np
import ssl

def setup_nltk():
    """Set up NLTK resources"""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

def setup_project_structure(code_root, data_root):
    """Set up project directories and config files"""
    # Create necessary directories
    directories = [
        os.path.join(code_root, 'config'),
        os.path.join(data_root, 'data/annotations'),
        os.path.join(data_root, 'data/images/val2017'),
        os.path.join(data_root, 'results')
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
    
    # Create config.yaml if it doesn't exist
    config_path = os.path.join(code_root, 'config/config.yaml')
    if not os.path.exists(config_path):
        config = {
            'model': {
                'sdxl_model_path': 'stabilityai/stable-diffusion-xl-base-1.0',
                'device': 'cuda'
            },
            'pipeline': {
                'max_length': 100,
                'num_inference_steps': 50,
                'guidance_scale': 7.5
            },
            'processing': {
                'image_size': 384,
                'batch_size': 1
            }
        }
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        print(f"Created config file at {config_path}")

def evaluate_captions(generated_caption, ground_truth_captions):
    """Evaluate caption quality using BLEU score"""
    # Tokenize captions
    reference_tokens = [nltk.word_tokenize(cap.lower()) for cap in ground_truth_captions]
    candidate_tokens = nltk.word_tokenize(generated_caption.lower())
    
    # Calculate BLEU score
    bleu_score = sentence_bleu(reference_tokens, candidate_tokens)
    return bleu_score

def main():
    # Add this line near the start of main(), before any NLTK operations
    setup_nltk()
    
    # Define root directories
    code_root = '/content/project/Multimodal_Image_to_Text_Exp_2/Multimodal Image-to-Text Exp 2'
    data_root = '/content/project/Multimodal_Image_to_Text_Exp_2'
    print(f"Code root: {code_root}")
    print(f"Data root: {data_root}")
    
    # Set up project structure
    setup_project_structure(code_root, data_root)
    
    # Verify config file exists
    config_path = os.path.join(code_root, 'config/config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize pipeline
    pipeline = CaptionPipeline(config)
    pipeline.initialize()
    
    # Initialize results dictionary
    results = {
        'captions': [],
        'metrics': {
            'bleu_scores': []
        }
    }
    
    # Set up absolute paths
    annotations_path = os.path.join(data_root, 'data/annotations/captions_val2017.json')
    images_dir = os.path.join(data_root, 'data/images/val2017')
    
    # Verify paths exist
    print(f"Checking paths...")
    print(f"Annotations path: {annotations_path}")
    print(f"Images directory: {images_dir}")
    
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Annotations file not found at {annotations_path}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found at {images_dir}")
    
    # Process COCO validation set
    coco = COCO(annotations_path)
    img_ids = coco.getImgIds()
    
    # Process images with progress bar
    for img_id in tqdm(img_ids[:100], desc="Processing images"):
        img_info = coco.loadImgs(img_id)[0]
        # Use absolute path for images
        image_path = os.path.join(images_dir, img_info['file_name'])
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
            
        # Get ground truth captions
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        gt_captions = [ann['caption'] for ann in anns]
        
        try:
            # Generate caption
            generated_caption = pipeline.generate_caption(image_path)
            
            # Calculate BLEU score
            bleu_score = evaluate_captions(generated_caption, gt_captions)
            
            # Store results
            results['captions'].append({
                'image_id': img_id,
                'image_file': img_info['file_name'],
                'generated_caption': generated_caption,
                'ground_truth_captions': gt_captions,
                'bleu_score': bleu_score
            })
            results['metrics']['bleu_scores'].append(bleu_score)
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            continue
    
    # Calculate and print average BLEU score
    if results['metrics']['bleu_scores']:
        avg_bleu = np.mean(results['metrics']['bleu_scores'])
        print(f"\nAverage BLEU score: {avg_bleu:.4f}")
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(data_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results
    results_path = os.path.join(results_dir, 'caption_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()



    