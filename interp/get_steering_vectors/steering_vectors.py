"""
Compute steering vectors from data.
"""

from collections import OrderedDict
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_prompt, load_data


def make_cache_hook(layer_idx, cache):
    """
    Create a hook function that caches activations from a specific layer.
    
    Args:
        layer_idx: Index or identifier for the layer
        cache: Dictionary to store cached activations
    
    Returns:
        Hook function that can be registered with register_forward_hook
    """
    def hook(module, inputs, output):
        act = output[0] if isinstance(output, tuple) else output
        cache[layer_idx] = act.detach().cpu()
    return hook


def get_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    df,
    column: str,
    layer_idx: int,
    model_type: str = "instruct",
    device=None,
    description: str = None,
):
    """
    Get activations for sentences from a specific column.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        df: DataFrame with sentence data
        column: Column name to extract sentences from ('i' or 'you')
        layer_idx: Layer index to extract activations from
        model_type: Type of model ("base" or "instruct")
        device: Device to run on (defaults to model.device)
        description: Description for progress bar
    
    Returns:
        OrderedDict: Dictionary mapping example indices to cached activations
    """
    if device is None:
        device = next(model.parameters()).device
    
    cache = OrderedDict()
    block = model.model.layers[layer_idx]
    
    unique_prompts = set()
    
    if description:
        print(f"Computing activations for {description}...")
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc=description):
        handle = block.register_forward_hook(
            make_cache_hook(f"Example: {i}", cache)
        )
        if row[column].split(".")[0] in unique_prompts:
            continue
        else:
            unique_prompts.add(row[column].split(".")[0])
        
        text, inputs = get_prompt(
            row[column].split(".")[0],
            tokenizer,
            device=device,
            model_type=model_type,
            add_system=False
        )
        
        _ = model(**inputs)
        handle.remove()
        
    print(f"Total number of unique prompts: {len(unique_prompts)}")
    
    return cache


def compute_average_vector(cache):
    """
    Compute average vector from cached activations.
    
    Args:
        cache: OrderedDict mapping example indices to activation tensors
               Each tensor has shape (B, T, D)
    
    Returns:
        torch.Tensor: Average vector with shape (1, D)
    """
    vecs = []
    for k, v in cache.items():
        # v shape: (B, T, D), take last token: (B, D)
        last = v[:, -1, :]  # (1, D)
        vecs.append(last)
    
    # Stack and average: (N, 1, D) -> (1, D)
    avg_vec = torch.mean(torch.stack(vecs, dim=0), dim=0)
    return avg_vec


def compute_steering_vector(avg_vec_i, avg_vec_you, normalize: bool = True):
    """
    Compute steering vector from average vectors.
    
    Args:
        avg_vec_i: Average activation vector for 'i' sentences (shape: [1, hidden_dim])
        avg_vec_you: Average activation vector for 'you' sentences (shape: [1, hidden_dim])
        normalize: Whether to normalize the steering vector
    
    Returns:
        torch.Tensor: Steering vector (normalized if normalize=True)
    """
    # Compute steering vector: difference between 'i' and 'you' vectors
    steering_vec = avg_vec_i - avg_vec_you
    steering_vec_norm = steering_vec.norm().item()
    
    print(f"Steering vector shape: {steering_vec.shape}")
    print(f"Steering vector norm: {steering_vec_norm:.2f}")
    
    # Normalize steering vector if requested
    if normalize:
        steering_vec = steering_vec / steering_vec.norm()
    
    return steering_vec


def compute_steering_vectors(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    data_path: str,
    layer_idx: int = 10,
    model_type: str = "instruct",
    device: str = None,
):
    """
    Compute steering vectors from I/You sentence pairs.
    
    This function:
    1. Loads data from a JSONL file with 'i' and 'you' columns
    2. Extracts activations from the specified layer for both sentence types
    3. Computes average vectors for 'i' and 'you' sentences
    4. Returns the normalized steering vector (avg_i - avg_you)
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        data_path: Path to JSONL file with 'i' and 'you' columns
        layer_idx: Layer index to extract activations from
        model_type: Type of model ("base" or "instruct")
        device: Device to run on (defaults to model.device)
    
    Returns:
        tuple: (steering_vec, avg_vec_i, avg_vec_you)
            - steering_vec: Normalized steering vector (shape: [1, hidden_dim])
            - avg_vec_i: Average activation vector for 'i' sentences
            - avg_vec_you: Average activation vector for 'you' sentences
    """
    # Load data
    df = load_data(data_path)
    
    # Get activations for 'i' sentences
    cache_i = get_activations(
        model=model,
        tokenizer=tokenizer,
        df=df,
        column="i",
        layer_idx=layer_idx,
        model_type=model_type,
        device=device,
        description="'I' sentences"
    )
    
    # Compute average vector for 'i' sentences
    avg_vec_i = compute_average_vector(cache_i)
    print(f"Average 'I' vector shape: {avg_vec_i.shape}")
    
    # Get activations for 'you' sentences
    cache_you = get_activations(
        model=model,
        tokenizer=tokenizer,
        df=df,
        column="you",
        layer_idx=layer_idx,
        model_type=model_type,
        device=device,
        description="'You' sentences"
    )
    
    # Compute average vector for 'you' sentences
    avg_vec_you = compute_average_vector(cache_you)
    print(f"Average 'You' vector shape: {avg_vec_you.shape}")
    
    # Compute steering vector
    steering_vec = compute_steering_vector(avg_vec_i, avg_vec_you, normalize=True)
    
    return steering_vec, avg_vec_i, avg_vec_you

