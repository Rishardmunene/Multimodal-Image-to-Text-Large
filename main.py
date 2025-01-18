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
    # Load configuration
    config = yaml.safe_load(open('config/config.yaml'))
    
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
    
    # Process COCO validation set
    coco = COCO('data/annotations/captions_val2017.json')
    img_ids = coco.getImgIds()
    
    # Process images with progress bar
    for img_id in tqdm(img_ids[:100], desc="Processing images"):
        img_info = coco.loadImgs(img_id)[0]
        image_path = os.path.join('data/images/val2017', img_info['file_name'])
        
        # Get ground truth captions
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        gt_captions = [ann['caption'] for ann in anns]
        
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
    
    # Calculate and print average BLEU score
    avg_bleu = np.mean(results['metrics']['bleu_scores'])
    print(f"\nAverage BLEU score: {avg_bleu:.4f}")
    
    # Save results
    with open('results/caption_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()