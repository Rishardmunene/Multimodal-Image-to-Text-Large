import yaml
import os
from pycocotools.coco import COCO
import urllib.request
import subprocess
from tqdm import tqdm
import json
from src.pipeline.caption_pipeline import CaptionPipeline
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import numpy as np
import ssl
import time
from datetime import datetime, timedelta

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
    """Evaluate caption quality using BLEU score with enhanced smoothing"""
    # Initialize smoothing function with method7 (more aggressive smoothing)
    smoother = SmoothingFunction()
    
    # Tokenize captions
    reference_tokens = [nltk.word_tokenize(cap.lower()) for cap in ground_truth_captions]
    candidate_tokens = nltk.word_tokenize(generated_caption.lower())
    
    # Calculate BLEU scores with focus on lower n-grams and enhanced smoothing
    bleu_scores = {
        # Unigram BLEU (focus on word overlap)
        'bleu-1': sentence_bleu(
            reference_tokens, 
            candidate_tokens, 
            weights=(1.0, 0, 0, 0),
            smoothing_function=smoother.method7
        ),
        # Bigram BLEU (basic phrase matching)
        'bleu-2': sentence_bleu(
            reference_tokens, 
            candidate_tokens, 
            weights=(0.6, 0.4, 0, 0),
            smoothing_function=smoother.method7
        ),
        # Combined lower-order n-grams
        'bleu-combined': sentence_bleu(
            reference_tokens, 
            candidate_tokens, 
            weights=(0.4, 0.3, 0.2, 0.1),
            smoothing_function=smoother.method7
        )
    }
    
    # Add additional metrics
    word_overlap = len(set(candidate_tokens) & set([word for ref in reference_tokens for word in ref]))
    total_words = len(set(candidate_tokens))
    if total_words > 0:
        bleu_scores['word_overlap_ratio'] = word_overlap / total_words
    else:
        bleu_scores['word_overlap_ratio'] = 0.0
    
    return bleu_scores

def format_time(seconds):
    """Format time duration"""
    return str(timedelta(seconds=int(seconds)))

def main():
    setup_nltk()
    
    # Define root directories
    code_root = '/content/project/Multimodal_Image_to_Text_Exp_2/Multimodal Image-to-Text Exp 2'
    data_root = '/content/project/Multimodal_Image_to_Text_Exp_2'
    
    # Initialize timing and progress tracking
    start_time = time.time()
    processed_images = 0
    warnings = []
    
    print("\nInitializing Image-to-Text Pipeline...")
    
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
    
    # Initialize results with more detailed metrics
    results = {
        'captions': [],
        'metrics': {
            'bleu_scores': {
                'bleu-1': [],
                'bleu-2': [],
                'bleu-combined': [],
                'word_overlap_ratio': []
            }
        },
        'warnings': [],
        'timing': {
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'total_duration': None,
            'images_per_second': None
        },
        'generation_params': {
            'do_sample': True,
            'temperature': 0.7,
            'top_k': 50,
            'top_p': 0.9,
            'num_beams': 5
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
    
    # Process images with enhanced progress tracking
    total_images = min(100, len(img_ids))
    with tqdm(total=total_images, desc="Processing images") as pbar:
        for img_id in img_ids[:total_images]:
            loop_start = time.time()
            
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
                generated_caption = pipeline.generate_caption(image_path)
                bleu_scores = evaluate_captions(generated_caption, gt_captions)
                
                # Update metrics
                for score_type, score in bleu_scores.items():
                    results['metrics']['bleu_scores'][score_type].append(score)
                
                # Store detailed results
                results['captions'].append({
                    'image_id': img_id,
                    'image_file': img_info['file_name'],
                    'generated_caption': generated_caption,
                    'ground_truth_captions': gt_captions,
                    'bleu_scores': bleu_scores,
                    'processing_time': time.time() - loop_start
                })
                
                processed_images += 1
                pbar.update(1)
                
                # Calculate and display progress statistics
                elapsed_time = time.time() - start_time
                images_per_second = processed_images / elapsed_time
                pbar.set_postfix({
                    'Speed': f'{images_per_second:.2f} img/s',
                    'Elapsed': format_time(elapsed_time)
                })
                
            except Exception as e:
                warning_msg = f"Error processing image {image_path}: {str(e)}"
                warnings.append(warning_msg)
                print(f"\nWarning: {warning_msg}")
                continue
    
    # Calculate final statistics
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Update timing information
    results['timing'].update({
        'end_time': datetime.now().isoformat(),
        'total_duration': total_duration,
        'images_per_second': processed_images / total_duration
    })
    
    # Calculate and display final metrics
    print("\nProcessing Complete!")
    print(f"\nProgress Summary:")
    print(f"Started at 0% (0/{total_images} images)")
    print(f"Finished at 100% ({processed_images}/{total_images} images)")
    print(f"Total processing time: {format_time(total_duration)}")
    print(f"Final processing speed: {processed_images/total_duration:.2f} images per second")
    
    print("\nBLEU Score Summary:")
    for score_type in results['metrics']['bleu_scores']:
        scores = results['metrics']['bleu_scores'][score_type]
        if scores:
            avg_score = np.mean(scores)
            print(f"Average {score_type}: {avg_score:.4f}")
    
    if warnings:
        print("\nWarnings encountered:")
        for warning in warnings:
            print(f"- {warning}")
    
    # Save detailed results
    results_dir = os.path.join(data_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'caption_results.json')
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_path}")

if __name__ == "__main__":
    main()



    