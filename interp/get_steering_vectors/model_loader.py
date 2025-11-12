"""
Model loading utilities.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_and_tokenizer(
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    cache_dir: str = "/projectnb/vkolagrp/skowshik/.cache/",
    dtype: str = "auto",
    device_map: str = "auto",
):
    """
    Load a model and tokenizer.
    
    Args:
        model_id: HuggingFace model identifier
        cache_dir: Directory to cache model files
        dtype: Data type for model weights (default: "auto")
        device_map: Device mapping strategy (default: "auto")
    
    Returns:
        tuple: (model, tokenizer)
    """
    # Set cache directory if provided
    if cache_dir:
        os.environ["HF_HOME"] = cache_dir
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        dtype=dtype,
        device_map=device_map
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    return model, tokenizer

