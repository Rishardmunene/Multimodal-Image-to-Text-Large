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
nltk.download('punkt')
import numpy as np

def evaluate_captions(generated_caption, ground_truth_captions):
    """Evaluate caption quality using BLEU score"""
    # Tokenize captions
    reference_tokens = [nltk.word_tokenize(cap.lower()) for cap in ground_truth_captions]
    candidate_tokens = nltk.word_tokenize(generated_caption.lower())
    
    # Calculate BLEU score
    bleu_score = sentence_bleu(reference_tokens, candidate_tokens)
    return bleu_score

def main():
    # Get the absolute path to the project root
    project_root = '/content/project/Multimodal_Image_to_Text_Exp_2'
    
    # Load configuration
    config = yaml.safe_load(open(os.path.join(project_root, 'config/config.yaml')))
    
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
    annotations_path = os.path.join(project_root, 'data/annotations/captions_val2017.json')
    images_dir = os.path.join(project_root, 'data/images/val2017')
    
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
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results
    results_path = os.path.join(results_dir, 'caption_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()



    