#!/usr/bin/env python3
"""
CLIP Improvement Strategies
Demonstrates various approaches to improve CLIP performance on identified weaknesses
"""

import torch
import clip
from clip_zero_shot_test import load_images, run_clip_classification
import json
from pathlib import Path

def test_prompt_engineering():
    """Test different prompt engineering strategies"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load CLIP model
    model, preprocess = clip.load("ViT-B/32", device=device)
    print("CLIP model loaded successfully!")
    
    # Load test images
    test_images = load_images("demo_images")
    print(f"Loaded {len(test_images)} test images")
    
    # Test different prompt strategies
    prompt_strategies = {
        "Original": [
            "a photo of a hammer",
            "a photo of a screwdriver",
            "a photo of a tool"
        ],
        "Detailed": [
            "a photo of a metal hammer with wooden handle",
            "a photo of a metal screwdriver with plastic handle", 
            "a photo of a hand tool"
        ],
        "Contextual": [
            "a photo of a hammer in a toolbox",
            "a photo of a screwdriver in a toolbox",
            "a photo of a tool in a workshop"
        ],
        "Functional": [
            "a photo of a hammer used for hitting nails",
            "a photo of a screwdriver used for turning screws",
            "a photo of a tool used for construction"
        ],
        "Descriptive": [
            "a photo of a heavy hammer with a long handle",
            "a photo of a thin screwdriver with a pointed tip",
            "a photo of a handheld tool"
        ]
    }
    
    results = {}
    
    for strategy_name, prompts in prompt_strategies.items():
        print(f"\n" + "="*80)
        print(f"TESTING PROMPT STRATEGY: {strategy_name}")
        print("="*80)
        
        strategy_results = run_clip_classification(model, preprocess, test_images, prompts)
        results[strategy_name] = strategy_results
        
        # Show results for problematic cases
        print(f"\nResults for {strategy_name} strategy:")
        print("-" * 50)
        
        for image_name in ["hammer", "screwdriver"]:
            if image_name in strategy_results:
                print(f"\n{image_name.upper()}:")
                sorted_prompts = sorted(strategy_results[image_name].items(), key=lambda x: x[1], reverse=True)
                for prompt, confidence in sorted_prompts:
                    print(f"  {confidence:.3f}: {prompt}")
    
    return results

def test_ensemble_approach():
    """Test ensemble approach using multiple prompts"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Load test images
    test_images = load_images("demo_images")
    
    # Multiple prompts for each concept
    ensemble_prompts = {
        "hammer": [
            "a photo of a hammer",
            "a photo of a metal hammer",
            "a photo of a tool with a handle",
            "a photo of a hammer in a toolbox",
            "a photo of a construction tool"
        ],
        "screwdriver": [
            "a photo of a screwdriver",
            "a photo of a metal screwdriver", 
            "a photo of a tool with a tip",
            "a photo of a screwdriver in a toolbox",
            "a photo of a repair tool"
        ],
        "tool": [
            "a photo of a tool",
            "a photo of a hand tool",
            "a photo of a metal tool",
            "a photo of a construction tool",
            "a photo of a workshop tool"
        ]
    }
    
    print(f"\n" + "="*80)
    print("ENSEMBLE APPROACH RESULTS")
    print("="*80)
    
    ensemble_results = {}
    
    for concept, prompts in ensemble_prompts.items():
        print(f"\nTesting ensemble for '{concept}':")
        print("-" * 50)
        
        # Get results for all prompts
        concept_results = run_clip_classification(model, preprocess, test_images, prompts)
        
        # Calculate ensemble scores (average across all prompts)
        ensemble_scores = {}
        for image_name in test_images.keys():
            ensemble_scores[image_name] = {}
            for prompt in prompts:
                if image_name in concept_results and prompt in concept_results[image_name]:
                    score = concept_results[image_name][prompt]
                    if concept not in ensemble_scores[image_name]:
                        ensemble_scores[image_name][concept] = []
                    ensemble_scores[image_name][concept].append(score)
        
        # Show ensemble results
        for image_name, scores in ensemble_scores.items():
            if concept in scores:
                avg_score = sum(scores[concept]) / len(scores[concept])
                max_score = max(scores[concept])
                print(f"  {image_name}: avg={avg_score:.3f}, max={max_score:.3f}")
                ensemble_results[f"{image_name}_{concept}"] = {
                    "average": avg_score,
                    "maximum": max_score,
                    "all_scores": scores[concept]
                }
    
    return ensemble_results

def test_attribute_focus():
    """Test focusing on specific attributes that CLIP struggled with"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Load test images
    test_images = load_images("demo_images")
    
    # Focus on problematic attributes
    attribute_prompts = {
        "material": [
            "a photo of something made of metal",
            "a photo of something made of wood", 
            "a photo of something made of plastic"
        ],
        "color": [
            "a photo of something red",
            "a photo of something yellow",
            "a photo of something brown"
        ],
        "texture": [
            "a photo of something with fur",
            "a photo of something smooth",
            "a photo of something rough"
        ],
        "size": [
            "a photo of something small",
            "a photo of something large",
            "a photo of something medium-sized"
        ]
    }
    
    print(f"\n" + "="*80)
    print("ATTRIBUTE FOCUS RESULTS")
    print("="*80)
    
    attribute_results = {}
    
    for attribute_type, prompts in attribute_prompts.items():
        print(f"\nTesting {attribute_type} attributes:")
        print("-" * 50)
        
        attr_results = run_clip_classification(model, preprocess, test_images, prompts)
        attribute_results[attribute_type] = attr_results
        
        # Show results for each image
        for image_name in test_images.keys():
            if image_name in attr_results:
                print(f"\n{image_name.upper()}:")
                sorted_prompts = sorted(attr_results[image_name].items(), key=lambda x: x[1], reverse=True)
                for prompt, confidence in sorted_prompts[:2]:
                    print(f"  {confidence:.3f}: {prompt}")
    
    return attribute_results

def generate_improvement_report():
    """Generate a comprehensive improvement report"""
    
    print("="*80)
    print("CLIP IMPROVEMENT STRATEGIES ANALYSIS")
    print("="*80)
    
    # Test different strategies
    prompt_results = test_prompt_engineering()
    ensemble_results = test_ensemble_approach()
    attribute_results = test_attribute_focus()
    
    # Generate recommendations
    print(f"\n" + "="*80)
    print("IMPROVEMENT RECOMMENDATIONS")
    print("="*80)
    
    print("\n1. PROMPT ENGINEERING STRATEGIES:")
    print("- Use more descriptive prompts (e.g., 'metal hammer with wooden handle')")
    print("- Add contextual information (e.g., 'hammer in a toolbox')")
    print("- Include functional descriptions (e.g., 'hammer used for hitting nails')")
    
    print("\n2. ENSEMBLE APPROACHES:")
    print("- Combine multiple prompts for the same concept")
    print("- Use both specific and general descriptions")
    print("- Average or take maximum confidence across prompts")
    
    print("\n3. ATTRIBUTE-FOCUSED APPROACHES:")
    print("- Train separate models for different attribute types")
    print("- Use specialized prompts for materials, colors, textures")
    print("- Combine attribute predictions for better understanding")
    
    print("\n4. DATA AUGMENTATION:")
    print("- Collect more diverse images of tools and objects")
    print("- Include images from different angles and contexts")
    print("- Add synthetic data for underrepresented categories")
    
    print("\n5. MODEL ARCHITECTURE IMPROVEMENTS:")
    print("- Use larger CLIP models (ViT-L/14, ViT-H/14)")
    print("- Implement attention mechanisms for better attribute focus")
    print("- Add specialized heads for different attribute types")
    
    # Save results
    all_results = {
        "prompt_engineering": prompt_results,
        "ensemble_approach": ensemble_results,
        "attribute_focus": attribute_results
    }
    
    with open("improvement_strategies_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nSaved all results to improvement_strategies_results.json")
    
    return all_results

if __name__ == "__main__":
    generate_improvement_report() 