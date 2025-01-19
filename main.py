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
from collections import defaultdict

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

def validate_pipeline(pipeline, coco, data_root, batch_size=32, max_samples=1000):
    """Comprehensive validation on larger dataset"""
    results = {
        'metrics': {
            'bleu_scores': {
                'bleu-1': [],
                'bleu-2': [],
                'bleu-combined': [],
                'word_overlap_ratio': []
            },
            'timing': {
                'start_time': datetime.now().isoformat(),
                'batch_times': [],
                'average_time_per_image': None
            }
        },
        'samples': []
    }
    
    # Get validation image IDs
    img_ids = coco.getImgIds()
    if max_samples:
        img_ids = img_ids[:max_samples]
    
    # Process in batches
    batches = [img_ids[i:i + batch_size] for i in range(0, len(img_ids), batch_size)]
    
    with tqdm(total=len(img_ids), desc="Validating") as pbar:
        for batch_idx, batch_ids in enumerate(batches):
            batch_start = time.time()
            batch_results = process_batch(
                batch_ids, pipeline, coco, data_root, pbar
            )
            
            # Update metrics
            for metric_name, scores in batch_results['metrics'].items():
                if metric_name in results['metrics']['bleu_scores']:
                    results['metrics']['bleu_scores'][metric_name].extend(scores)
            
            # Store sample results
            results['samples'].extend(batch_results['samples'])
            
            # Update timing
            batch_time = time.time() - batch_start
            results['metrics']['timing']['batch_times'].append(batch_time)
            
            # Display batch statistics
            display_batch_stats(batch_results, batch_idx + 1, len(batches))
    
    # Calculate final statistics
    compute_final_statistics(results)
    
    return results

def process_batch(batch_ids, pipeline, coco, data_root, pbar):
    """Process a batch of images"""
    batch_results = {
        'metrics': {
            'bleu-1': [],
            'bleu-2': [],
            'bleu-combined': [],
            'word_overlap_ratio': []
        },
        'samples': []
    }
    
    for img_id in batch_ids:
        try:
            # Get image info and ground truth
            img_info = coco.loadImgs(img_id)[0]
            image_path = os.path.join(data_root, 'data/images/val2017', img_info['file_name'])
            
            # Get ground truth captions
            ann_ids = coco.getAnnIds(imgIds=img_id)
            gt_captions = [ann['caption'] for ann in coco.loadAnns(ann_ids)]
            
            # Generate caption
            generated_caption = pipeline.generate_caption(image_path)
            
            # Calculate metrics
            metrics = calculate_metrics(generated_caption, gt_captions)
            
            # Update batch results
            for metric_name, score in metrics.items():
                if metric_name in batch_results['metrics']:
                    batch_results['metrics'][metric_name].append(score)
            
            batch_results['samples'].append({
                'image_id': img_id,
                'file_name': img_info['file_name'],
                'generated_caption': generated_caption,
                'ground_truth': gt_captions,
                'metrics': metrics
            })
            
            pbar.update(1)
            
        except Exception as e:
            print(f"\nError processing image {img_id}: {str(e)}")
            continue
    
    return batch_results

def calculate_metrics(generated_caption, reference_captions):
    """Calculate comprehensive metrics"""
    metrics = {}
    
    # BLEU scores with smoothing
    smoother = SmoothingFunction()
    ref_tokens = [nltk.word_tokenize(cap.lower()) for cap in reference_captions]
    gen_tokens = nltk.word_tokenize(generated_caption.lower())
    
    # Calculate BLEU scores
    for n in range(1, 5):
        weights = tuple([1.0/n] * n + [0.0] * (4-n))
        metrics[f'bleu-{n}'] = sentence_bleu(
            ref_tokens, gen_tokens,
            weights=weights,
            smoothing_function=smoother.method7
        )
    
    # Word overlap ratio
    unique_gen_words = set(gen_tokens)
    unique_ref_words = set(word for ref in ref_tokens for word in ref)
    metrics['word_overlap_ratio'] = len(unique_gen_words & unique_ref_words) / len(unique_gen_words) if unique_gen_words else 0
    
    return metrics

def display_batch_stats(batch_results, batch_num, total_batches):
    """Display batch statistics"""
    print(f"\nBatch {batch_num}/{total_batches} Statistics:")
    for metric_name, scores in batch_results['metrics'].items():
        if scores:
            avg_score = np.mean(scores)
            print(f"Average {metric_name}: {avg_score:.4f}")

def compute_final_statistics(results):
    """Compute and add final statistics to results"""
    results['final_metrics'] = {}
    
    # Calculate averages for all metrics
    for metric_name, scores in results['metrics']['bleu_scores'].items():
        if scores:
            results['final_metrics'][f'avg_{metric_name}'] = float(np.mean(scores))
            results['final_metrics'][f'std_{metric_name}'] = float(np.std(scores))
    
    # Calculate timing statistics
    batch_times = results['metrics']['timing']['batch_times']
    if batch_times:
        results['metrics']['timing']['average_time_per_batch'] = float(np.mean(batch_times))
        results['metrics']['timing']['total_processing_time'] = float(np.sum(batch_times))
        total_samples = len([s for s in results['samples'] if 'metrics' in s])
        results['metrics']['timing']['average_time_per_image'] = results['metrics']['timing']['total_processing_time'] / total_samples if total_samples > 0 else 0

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
    
    # Run validation
    validation_results = validate_pipeline(
        pipeline=pipeline,
        coco=coco,
        data_root=data_root,
        batch_size=32,
        max_samples=1000  # Increase this for larger validation
    )
    
    # Save results
    results_path = os.path.join(data_root, 'results/validation_results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"\nValidation results saved to: {results_path}")

if __name__ == "__main__":
    main()



    