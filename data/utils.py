import random
import json
from typing import List, Dict, Any
import torch
from transformers import set_seed



def make_fewshot_block(seed_pairs: List[Dict[str, str]], max_pairs: int = 20) -> str:
    """
    Format seed pairs as a few-shot examples block for prompts.
    
    Takes a list of contrastive pairs and formats them as a JSON string
    prefixed with "EXAMPLES (JSON):". Limits the number of pairs to keep
    prompt size reasonable.
    
    Args:
        seed_pairs: List of dictionaries with 'i' and 'you' keys.
        max_pairs: Maximum number of pairs to include (default: 20).
    
    Returns:
        Formatted string containing the examples block.
    """
    # Use up to max_pairs to keep prompt size reasonable
    chosen = seed_pairs[:max_pairs]
    as_json = json.dumps(chosen, ensure_ascii=True, indent=2)
    return f"EXAMPLES (JSON):\n{as_json}"


def set_all_seeds(seed: int):
    """
    Set random seeds for reproducibility across all libraries.
    
    Sets seeds for Python's random module, NumPy (if available),
    transformers, and PyTorch (including CUDA if available).
    
    Args:
        seed: Integer seed value to use for all random number generators.
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        
def parse_model_json(output_text: str) -> List[Dict[str, str]]:
    """
    Extract the first top-level JSON array in the text and parse it.
    We keep it simple and robust: find the first '[' and last ']' and parse.
    """
    start = output_text.find("[")
    end = output_text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array found in model output.")
    raw = output_text[start:end+1]
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("Parsed JSON is not a list.")
    # Validate shape
    cleaned = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            continue
        i_sent = row.get("i")
        you_sent = row.get("you")
        if isinstance(i_sent, str) and isinstance(you_sent, str):
            cleaned.append({"i": i_sent.strip(), "you": you_sent.strip()})
    return cleaned

