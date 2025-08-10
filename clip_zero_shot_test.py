#!/usr/bin/env python3
"""
CLIP Zero-Shot Classification Test
Tests CLIP's ability to classify objects with various prompts and analyzes strengths/weaknesses
"""

import os
import torch
import clip
from PIL import Image
import numpy as np
from pathlib import Path

def load_images(image_dir):
    """Load all images from the test_images directory"""
    images = {}
    image_dir = Path(image_dir)
    
    for img_path in image_dir.glob("*.jpg"):
        try:
            image = Image.open(img_path)
            images[img_path.stem] = image
            print(f"Loaded: {img_path.stem}")
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")
    
    return images

def run_clip_classification(model, preprocess, images, text_prompts):
    """Run CLIP classification on images with given text prompts"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Preprocess images
    image_inputs = []
    image_names = []
    
    for name, image in images.items():
        try:
            processed_image = preprocess(image).unsqueeze(0).to(device)
            image_inputs.append(processed_image)
            image_names.append(name)
        except Exception as e:
            print(f"Failed to process {name}: {e}")
    
    if not image_inputs:
        print("No valid images to process!")
        return {}
    
    # Tokenize text prompts
    text_tokens = clip.tokenize(text_prompts).to(device)
    
    # Get predictions
    results = {}
    
    with torch.no_grad():
        # Encode images and text
        image_features = torch.cat(image_inputs, dim=0)
        image_features = model.encode_image(image_features)
        text_features = model.encode_text(text_tokens)
        
        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarities
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # Store results
        for i, image_name in enumerate(image_names):
            results[image_name] = {}
            for j, prompt in enumerate(text_prompts):
                results[image_name][prompt] = similarity[i, j].item()
    
    return results

def analyze_results(results, images, text_prompts):
    """Analyze CLIP results and identify strengths/weaknesses"""
    print("\n" + "="*80)
    print("CLIP ZERO-SHOT CLASSIFICATION RESULTS")
    print("="*80)
    
    # Show top matches for each image
    print("\nTOP MATCHES FOR EACH IMAGE:")
    print("-" * 50)
    
    for image_name in results.keys():
        print(f"\n{image_name.upper()}:")
        # Sort prompts by confidence
        sorted_prompts = sorted(
            results[image_name].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for prompt, confidence in sorted_prompts[:3]:
            print(f"  {confidence:.3f}: {prompt}")
    
    # Show top matches for each prompt
    print("\n\nTOP IMAGES FOR EACH PROMPT:")
    print("-" * 50)
    
    for prompt in text_prompts:
        print(f"\n{prompt}:")
        # Sort images by confidence for this prompt
        image_scores = [(name, results[name][prompt]) for name in results.keys()]
        image_scores.sort(key=lambda x: x[1], reverse=True)
        
        for image_name, confidence in image_scores[:3]:
            print(f"  {confidence:.3f}: {image_name}")
    
    # Analyze strengths and weaknesses
    print("\n\nANALYSIS OF CLIP'S STRENGTHS & WEAKNESSES:")
    print("-" * 50)
    
    # Find best matches (strengths)
    best_matches = []
    for image_name in results.keys():
        best_prompt = max(results[image_name].items(), key=lambda x: x[1])
        best_matches.append((image_name, best_prompt[0], best_prompt[1]))
    
    best_matches.sort(key=lambda x: x[2], reverse=True)
    
    print("\nSTRENGTHS (High confidence matches):")
    for image_name, prompt, confidence in best_matches[:3]:
        print(f"  ✓ {image_name} → '{prompt}' (confidence: {confidence:.3f})")
    
    # Find worst matches (weaknesses)
    worst_matches = []
    for image_name in results.keys():
        worst_prompt = min(results[image_name].items(), key=lambda x: x[1])
        worst_matches.append((image_name, worst_prompt[0], worst_prompt[1]))
    
    worst_matches.sort(key=lambda x: x[2])
    
    print("\nWEAKNESSES (Low confidence matches):")
    for image_name, prompt, confidence in worst_matches[:3]:
        print(f"  ✗ {image_name} → '{prompt}' (confidence: {confidence:.3f})")
    
    # Find ambiguous cases (multiple high confidence matches)
    print("\nAMBIGUOUS CASES (Multiple high confidence matches):")
    for image_name in results.keys():
        scores = list(results[image_name].values())
        high_conf_count = sum(1 for s in scores if s > 0.3)
        if high_conf_count > 1:
            print(f"  ? {image_name} has {high_conf_count} high-confidence matches")
            sorted_scores = sorted(results[image_name].items(), key=lambda x: x[1], reverse=True)
            for prompt, conf in sorted_scores[:2]:
                print(f"    - '{prompt}': {conf:.3f}")

def main():
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    print("CLIP model loaded successfully!")
    
    # Load test images
    images = load_images("demo_images")
    print(f"Loaded {len(images)} images")
    
    # Define text prompts to test
    text_prompts = [
        "a photo of a hammer",
        "a photo of something with a handle", 
        "a photo of a screwdriver",
        "a photo of a tool",
        "a photo of an apple",
        "a photo of a banana",
        "a photo of fruit",
        "a photo of a cat",
        "a photo of a dog",
        "a photo of an animal",
        "a photo of a pet",
        "a photo of something you can eat",
        "a photo of something made of metal",
        "a photo of something made of wood",
        "a photo of something sharp",
        "a photo of something round",
        "a photo of something small",
        "a photo of something large",
        "a photo of something colorful",
        "a photo of something orange",
        "a photo of something red",
        "a photo of something brown",
        "a photo of something with fur",
        "a photo of something with a tail",
        "a photo of something you can hold",
        "a photo of something you can use",
        "a photo of something in the kitchen",
        "a photo of something in the garage",
        "a photo of something in the garden"
    ]
    
    print(f"Testing with {len(text_prompts)} text prompts")
    
    # Run classification
    results = run_clip_classification(model, preprocess, images, text_prompts)
    
    # Analyze results
    analyze_results(results, images, text_prompts)
    
    print("\n" + "="*80)
    print("TEST COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main() 