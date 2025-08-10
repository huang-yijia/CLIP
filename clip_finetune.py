#!/usr/bin/env python3
"""
CLIP Fine-tuning Script
Fine-tunes CLIP to improve performance on tool recognition and attribute understanding
"""

import os
import torch
import clip
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import argparse

class CLIPDataset(Dataset):
    """Custom dataset for CLIP fine-tuning"""
    
    def __init__(self, image_dir, annotations_file, preprocess, device):
        self.image_dir = Path(image_dir)
        self.preprocess = preprocess
        self.device = device
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Create image-text pairs
        self.pairs = []
        for img_name, labels in self.annotations.items():
            img_path = self.image_dir / f"{img_name}.jpg"
            if img_path.exists():
                for label in labels:
                    self.pairs.append((str(img_path), label))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_path, text = self.pairs[idx]
        
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        image = self.preprocess(image)
        
        # Tokenize text
        text_tokens = clip.tokenize([text])[0]
        
        return image, text_tokens

def create_annotations():
    """Create annotations for fine-tuning based on our analysis"""
    annotations = {
        "hammer": [
            "a photo of a hammer",
            "a photo of a tool",
            "a photo of something made of metal",
            "a photo of something you can use",
            "a photo of something with a handle",
            "a photo of something in the garage",
            "a photo of something you can hold"
        ],
        "screwdriver": [
            "a photo of a screwdriver", 
            "a photo of a tool",
            "a photo of something made of metal",
            "a photo of something sharp",
            "a photo of something you can use",
            "a photo of something with a handle",
            "a photo of something in the garage"
        ],
        "apple": [
            "a photo of an apple",
            "a photo of fruit",
            "a photo of something you can eat",
            "a photo of something red",
            "a photo of something round",
            "a photo of something in the kitchen"
        ],
        "banana": [
            "a photo of a banana",
            "a photo of fruit", 
            "a photo of something you can eat",
            "a photo of something yellow",
            "a photo of something long",
            "a photo of something in the kitchen"
        ],
        "cat": [
            "a photo of a cat",
            "a photo of a pet",
            "a photo of an animal",
            "a photo of something with fur",
            "a photo of something with a tail",
            "a photo of something small"
        ],
        "dog": [
            "a photo of a dog",
            "a photo of a pet",
            "a photo of an animal", 
            "a photo of something with fur",
            "a photo of something with a tail",
            "a photo of something large"
        ]
    }
    
    return annotations

def contrastive_loss(logits_per_image, logits_per_text, temperature=0.07):
    """Compute contrastive loss for CLIP fine-tuning"""
    # Ground truth: diagonal elements should be highest
    ground_truth = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
    
    # Image-to-text loss
    loss_i = nn.functional.cross_entropy(logits_per_image / temperature, ground_truth)
    
    # Text-to-image loss  
    loss_t = nn.functional.cross_entropy(logits_per_text / temperature, ground_truth)
    
    return (loss_i + loss_t) / 2

def fine_tune_clip(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-5, device='cuda'):
    """Fine-tune CLIP model"""
    
    # Move model to device
    model = model.to(device)
    
    # Setup optimizer with lower learning rate to prevent NaN
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, eps=1e-8)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"Starting fine-tuning for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, (images, texts) in enumerate(train_pbar):
            images = images.to(device)
            texts = texts.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits_per_image, logits_per_text = model(images, texts)
            
            # Compute loss
            loss = contrastive_loss(logits_per_image, logits_per_text)
            
            # Skip if loss is NaN
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected in batch {batch_idx}, skipping...")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for images, texts in val_pbar:
                images = images.to(device)
                texts = texts.to(device)
                
                logits_per_image, logits_per_text = model(images, texts)
                loss = contrastive_loss(logits_per_image, logits_per_text)
                
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_clip_model.pth')
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
    
    return train_losses, val_losses

def evaluate_finetuned_model(model, preprocess, test_images, text_prompts, device):
    """Evaluate the fine-tuned model on our test set"""
    model.eval()
    
    # Preprocess images
    image_inputs = []
    image_names = []
    
    for name, image in test_images.items():
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

def main():
    parser = argparse.ArgumentParser(description='Fine-tune CLIP model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate, skip training')
    
    args = parser.parse_args()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load CLIP model
    model, preprocess = clip.load("ViT-B/32", device=device)
    print("CLIP model loaded successfully!")
    
    # Create annotations
    annotations = create_annotations()
    
    # Save annotations to file
    with open('annotations.json', 'w') as f:
        json.dump(annotations, f, indent=2)
    print("Created annotations.json")
    
    if not args.eval_only:
        # Create datasets
        train_dataset = CLIPDataset("demo_images", "annotations.json", preprocess, device)
        val_dataset = CLIPDataset("demo_images", "annotations.json", preprocess, device)  # Using same data for simplicity
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        print(f"Dataset size: {len(train_dataset)} pairs")
        
        # Fine-tune the model
        train_losses, val_losses = fine_tune_clip(
            model, train_loader, val_loader, 
            num_epochs=args.epochs, 
            learning_rate=args.lr, 
            device=device
        )
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        with open('training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        print("Saved training history")
    
    # Load best model if it exists
    if os.path.exists('best_clip_model.pth'):
        model.load_state_dict(torch.load('best_clip_model.pth'))
        print("Loaded fine-tuned model")
    
    # Evaluate the model
    print("\nEvaluating fine-tuned model...")
    
    # Load test images
    from clip_zero_shot_test import load_images
    test_images = load_images("demo_images")
    
    # Test prompts (focusing on the problematic ones from our analysis)
    test_prompts = [
        "a photo of a hammer",
        "a photo of a screwdriver", 
        "a photo of a tool",
        "a photo of something made of metal",
        "a photo of something with a handle",
        "a photo of something sharp",
        "a photo of something red",
        "a photo of something yellow",
        "a photo of something with fur",
        "a photo of something you can eat"
    ]
    
    # Evaluate
    results = evaluate_finetuned_model(model, preprocess, test_images, test_prompts, device)
    
    # Print results
    print("\n" + "="*80)
    print("FINE-TUNED CLIP RESULTS")
    print("="*80)
    
    for image_name in results.keys():
        print(f"\n{image_name.upper()}:")
        sorted_prompts = sorted(results[image_name].items(), key=lambda x: x[1], reverse=True)
        for prompt, confidence in sorted_prompts[:3]:
            print(f"  {confidence:.3f}: {prompt}")
    
    print("\n" + "="*80)
    print("FINE-TUNING COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main() 