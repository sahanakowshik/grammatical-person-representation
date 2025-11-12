"""
Utility functions for steering vectors computation.
"""

from collections import OrderedDict
import pandas as pd
import torch
from transformers import AutoTokenizer


def get_prompt(
    prompt: str,
    tokenizer: AutoTokenizer,
    device=None,
    model_type: str = "instruct",
    add_system: bool = False,
):
    """
    Format a prompt for the model.
    
    Args:
        prompt: The input prompt text
        tokenizer: The tokenizer to use
        device: Device to move tensors to (if None, tensors stay on CPU)
        model_type: Type of model ("base" or "instruct")
        add_system: Whether to add a system message
    
    Returns:
        tuple: (formatted_text, model_inputs)
    """
    if model_type == "base":
        # No chat template: pass the raw text straight to the tokenizer
        text = prompt if isinstance(prompt, str) else str(prompt)
        model_inputs = tokenizer([text], return_tensors="pt")
        if device is not None:
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        return text, model_inputs
    else:
        if add_system:
            messages = [
                {"role": "system", "content": prompt.split(".", 1)[0]},
                {"role": "user", "content": prompt.split(".", 1)[1] if "." in prompt else ""}
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt")
        if device is not None:
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        return text, model_inputs


def load_data(data_path: str):
    """
    Load data from a JSONL file.
    
    Args:
        data_path: Path to JSONL file with 'i' and 'you' columns
    
    Returns:
        DataFrame with the loaded data
    """
    return pd.read_json(data_path, lines=True)


def load_steering_vector(file_path: str, device=None):
    """
    Load a saved steering vector from a .pt file.
    
    Args:
        file_path: Path to the saved .pt file
        device: Device to load tensors to (if None, keeps original device)
    
    Returns:
        dict: Dictionary containing:
            - 'steering_vec': The steering vector tensor
            - 'avg_vec_i': Average vector for 'i' sentences
            - 'avg_vec_you': Average vector for 'you' sentences
            - 'layer_idx': Layer index used
            - 'model_id': Model identifier used
    """
    data = torch.load(file_path, map_location=device)
    return data

