"""
Inference script for video-language alignment model
Generate text descriptions from video representations
"""
import torch
import numpy as np
import pickle
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from models.alignment.model import VideoLanguageAlignment
from transformers import MBartTokenizer

def load_model(checkpoint_path, device='cuda'):
    """Load trained alignment model"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model_config = checkpoint.get('model_config', {
        'video_repr_dim': 1536,
        'mbart_model_name': 'facebook/mbart-large-50',
        'mbart_model_path': None,
        'd_model': 1024,
        'dropout': 0.1
    })
    
    model = VideoLanguageAlignment(
        video_repr_dim=model_config['video_repr_dim'],
        mbart_model_name=model_config.get('mbart_model_name', 'facebook/mbart-large-50'),
        mbart_model_path=model_config.get('mbart_model_path', None),
        d_model=model_config['d_model'],
        dropout=model_config['dropout'],
        freeze_mbart=False
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, model_config

def generate_from_video_reprs(
    model,
    global_repr,
    local_reprs,
    device='cuda',
    max_length=128,
    num_beams=4
):
    """
    Generate text from video representations
    
    Args:
        model: Trained alignment model
        global_repr: Global representation [512] or [batch, 512]
        local_reprs: Local representations [2, 512] or [batch, 2, 512]
        device: Device
        max_length: Maximum generation length
        num_beams: Beam search width
    
    Returns:
        generated_texts: List of generated text strings
    """
    # Convert to tensor first
    if isinstance(global_repr, np.ndarray):
        global_repr = torch.FloatTensor(global_repr)
    if isinstance(local_reprs, np.ndarray):
        local_reprs = torch.FloatTensor(local_reprs)
    
    # Ensure batch dimension
    if global_repr.ndim == 1:
        global_repr = global_repr.unsqueeze(0)
    if local_reprs.ndim == 2:
        local_reprs = local_reprs.unsqueeze(0)
    
    # Move to device
    global_repr = global_repr.to(device)
    local_reprs = local_reprs.to(device)
    
    generated_texts = model.generate(
        global_repr=global_repr,
        local_reprs=local_reprs,
        max_length=max_length,
        num_beams=num_beams
    )
    
    return generated_texts

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate text from video representations")
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (will load ALIGNMENT config)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to alignment model checkpoint')
    parser.add_argument('--video_reprs_path', type=str, default=None,
                        help='Path to video representations npz file')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to generate')
    parser.add_argument('--max_length', type=int, default=None,
                        help='Maximum generation length')
    parser.add_argument('--num_beams', type=int, default=None,
                        help='Beam search width')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--text_data_path', type=str, default=None,
                        help='Path to text data file (for comparison)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save generation results')
    
    args = parser.parse_args()
    
    # Load from config if provided
    if args.config:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", args.config)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config_dict = config_module.ALIGNMENT
        
        inference_config = config_dict.get('inference', {})
        args.checkpoint = args.checkpoint or inference_config.get('checkpoint')
        args.video_reprs_path = args.video_reprs_path or inference_config.get('video_reprs_path')
        args.num_samples = args.num_samples or inference_config.get('num_samples', 10)
        args.max_length = args.max_length or inference_config.get('max_length', 128)
        args.num_beams = args.num_beams or inference_config.get('num_beams', 4)
        args.text_data_path = args.text_data_path or inference_config.get('text_data_path')
        args.output_path = args.output_path or inference_config.get('output_path')
    
    # Validate required arguments
    if not args.checkpoint or not args.video_reprs_path:
        parser.error("--checkpoint and --video_reprs_path are required")
    
    # Load model
    print("Loading alignment model...")
    model, model_config = load_model(args.checkpoint, device=args.device)
    print("Model loaded successfully!")
    print(f"Model config: {model_config}")
    print(f"Decoder start token ID: {model.decoder_start_token_id}")
    print(f"EOS token ID: {model.tokenizer.eos_token_id}")
    print(f"PAD token ID: {model.tokenizer.pad_token_id}")
    
    # Load video representations
    print(f"\nLoading video representations from {args.video_reprs_path}...")
    video_data = np.load(args.video_reprs_path)
    global_reprs = video_data['global_reprs']  # [num_samples, 512]
    local_reprs = video_data['local_reprs']  # [num_samples, 2, 512]
    
    num_samples = min(args.num_samples, len(global_reprs))
    print(f"Processing {num_samples} samples...")
    
    # Load ground truth texts if provided
    ground_truth_texts = None
    if args.text_data_path:
        import json
        print(f"\nLoading ground truth texts from {args.text_data_path}...")
        with open(args.text_data_path, 'r', encoding='utf-8') as f:
            text_data = json.load(f)
        if isinstance(text_data, dict) and 'texts' in text_data:
            ground_truth_texts = text_data['texts']
        elif isinstance(text_data, list):
            ground_truth_texts = text_data
        print(f"Loaded {len(ground_truth_texts)} ground truth texts")
    
    # Generate text for each sample
    print("\n" + "="*60)
    print("GENERATING TEXT DESCRIPTIONS")
    print("="*60)
    
    results = []
    
    for idx in range(num_samples):
        print(f"\nSample {idx+1}/{num_samples}:")
        
        global_repr = global_reprs[idx]  # [512]
        local_repr = local_reprs[idx]  # [2, 512]
        
        generated_texts = generate_from_video_reprs(
            model,
            global_repr,
            local_repr,
            device=args.device,
            max_length=args.max_length,
            num_beams=args.num_beams
        )
        
        generated_text = generated_texts[0] if generated_texts else ""
        
        # Debug: Check if text is empty or suspicious
        if not generated_text or len(generated_text.strip()) == 0:
            print(f"  Generated text: (empty)")
            print(f"  Warning: Generated empty text. Model may need more training.")
        elif len(set(generated_text)) < 3:  # Only 1-2 unique characters
            print(f"  Generated text: {generated_text[:100]}...")
            print(f"  Warning: Generated repetitive text. Model may need more training.")
        else:
            print(f"  Generated text: {generated_text}")
        
        # Compare with ground truth if available
        if ground_truth_texts and idx < len(ground_truth_texts):
            gt_text = ground_truth_texts[idx]
            print(f"  Ground truth:   {gt_text}")
        
        results.append({
            'sample_id': idx,
            'generated_text': generated_text,
            'ground_truth': ground_truth_texts[idx] if ground_truth_texts and idx < len(ground_truth_texts) else None
        })
    
    # Save results if output path provided
    if args.output_path:
        import json
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {args.output_path}")
    
    print("\nDone!")

