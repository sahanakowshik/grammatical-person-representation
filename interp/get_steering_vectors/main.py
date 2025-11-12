"""
Main script to compute steering vectors.
"""

import argparse
import os
import sys
import torch
from omegaconf import OmegaConf

# # Handle both direct execution and module execution
# try:
#     from .model_loader import load_model_and_tokenizer
#     from .steering_vectors import compute_steering_vectors
# except ImportError:
#     # Add parent directory (interp) to path for direct execution
#     # This allows importing steering_vectors as a package
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     parent_dir = os.path.dirname(current_dir)  # interp directory
#     if parent_dir not in sys.path:
#         sys.path.insert(0, parent_dir)
#     from steering_vectors.model_loader import load_model_and_tokenizer
#     from steering_vectors.steering_vectors import compute_steering_vectors
    
from model_loader import load_model_and_tokenizer
from steering_vectors import compute_steering_vectors


def main():
    parser = argparse.ArgumentParser(description="Compute steering vectors from I/You sentence pairs")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Path to config YAML file"
    )
    
    args = parser.parse_args()
    
    # Load config
    cfg = OmegaConf.load(args.config)
    
    # Get parameters from config (with overrides from command line)
    model_id = cfg.get("model_id", "meta-llama/Llama-3.1-8B-Instruct")
    data_path = cfg.get("data_path")
    cache_dir = cfg.get("cache_dir", "/projectnb/vkolagrp/skowshik/.cache/")
    output_path = cfg.get("output_path", None)
    output_name = cfg.get("output_name", None)
    model_type = cfg.get("model_type", "instruct")
    max_unique_prompts = cfg.get("max_unique_prompts", 200)
    
    if data_path is None:
        raise ValueError("data_path must be provided either in config or via --data_path argument")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model: {model_id}")
    model, tokenizer = load_model_and_tokenizer(
        model_id=model_id,
        cache_dir=cache_dir
    )
    
    # Compute steering vectors
    results = compute_steering_vectors(
        model=model,
        tokenizer=tokenizer,
        data_path=data_path,
        model_type=model_type,
        device=device,
        max_unique_prompts=max_unique_prompts
    )
    
    for layer_idx, result in results.items():
        
        output_dir = os.path.join(output_path, f"layer_{layer_idx}")
        os.makedirs(output_dir, exist_ok=True)

        # Save if output path is provided
        if output_dir:
            torch.save({
                'steering_vec': result["steering_vec"],
                'avg_vec_i': result["avg_vec_i"],
                'avg_vec_you': result["avg_vec_you"],
                'layer_idx': layer_idx,
                'model_id': model_id,
            }, f"{output_dir}/{output_name}")
            print(f"Saved steering vector to {output_path}")
            
        # break
        
        print("Done!")


if __name__ == "__main__":
    main()

