"""
Compute steering vectors from data (collect activations from all layers).
"""

from collections import OrderedDict
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

from utils import get_prompt, load_data


def _get_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError("Could not locate model blocks; adjust `_get_blocks`.")


def make_layer_hook(layer_idx, cache_dict):
    """
    Create a hook that saves activations for a given layer.
    """
    def hook(module, inputs, output):
        act = output[0] if isinstance(output, tuple) else output
        cache_dict[layer_idx] = act.detach().cpu()
    return hook


def get_activations_all_layers(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    df,
    column: str,
    model_type: str = "instruct",
    device=None,
    description: str = None,
    # max_unique_prompts: int = 200,
):
    """
    Loop over texts. For each text:
      - register hooks for all layers
      - run one forward pass
      - store per-layer activations in a dict
      - append dict to outer list
    """
    if device is None:
        device = next(model.parameters()).device

    blocks = _get_blocks(model)
    all_examples = []  # outer list of per-example activation dicts
    seen = set()

    print(f"Collecting activations for {description or column}...")

    for i, row in tqdm(df.iterrows(), total=len(df), desc=description or column):
        text_str = row[column]
        if text_str in seen:
            continue
        seen.add(text_str)

        # per-example cache
        example_cache = {}

        # register hooks for all layers
        handles = []
        for layer_idx, block in enumerate(blocks):
            h = block.register_forward_hook(make_layer_hook(layer_idx, example_cache))
            handles.append(h)

        # tokenize and forward
        _, inputs = get_prompt(
            text_str,
            tokenizer,
            device=device,
            model_type=model_type,
            add_system=False,
        )

        with torch.no_grad():
            _ = model(**inputs)

        # remove hooks after forward pass
        for h in handles:
            h.remove()
            
        print(f"Got {len(example_cache)} activations for row {i}")

        all_examples.append(example_cache)

        # if len(seen) >= max_unique_prompts:
        #     break

    print(f"Total unique prompts processed: {len(seen)}")
    return all_examples  # list of dicts: [{layer_idx: tensor, ...}, ...]


def compute_average_vector(cache_list, layer_idx: int):
    """
    Compute average vector from cached activations for a specific layer.
    cache_list: list of dicts (each dict: layer_idx -> activation tensor)
    """
    vecs = []
    for cache in cache_list:
        if layer_idx not in cache:
            continue
        v = cache[layer_idx]
        last = v[:, -1, :]  # (B, D)
        vecs.append(last)
    if not vecs:
        raise ValueError(f"No activations found for layer {layer_idx}")
    avg_vec = torch.mean(torch.stack(vecs, dim=0), dim=0)
    return avg_vec


def compute_steering_vector(avg_vec_i, avg_vec_you, normalize: bool = True):
    """
    Compute steering vector from average vectors.
    """
    steering_vec = avg_vec_i - avg_vec_you
    if normalize:
        steering_vec = steering_vec / steering_vec.norm()
    return steering_vec


def compute_steering_vectors(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    data_path: str,
    model_type: str = "instruct",
    device: str = None,
    max_unique_prompts: int = 10,
):
    """
    Compute steering vectors across all layers.
    Returns dict: layer_idx -> (steering_vec, avg_i, avg_you)
    """
    df = load_data(data_path)

    cache_i = get_activations_all_layers(
        model=model,
        tokenizer=tokenizer,
        df=df,
        column="i",
        model_type=model_type,
        device=device,
        description="'I' sentences",
        max_unique_prompts=max_unique_prompts,
    )

    cache_you = get_activations_all_layers(
        model=model,
        tokenizer=tokenizer,
        df=df,
        column="you",
        model_type=model_type,
        device=device,
        description="'You' sentences",
        max_unique_prompts=max_unique_prompts,
    )

    results = {}
    blocks = _get_blocks(model)
    for layer_idx, _ in enumerate(blocks):
        avg_i = compute_average_vector(cache_i, layer_idx)
        avg_you = compute_average_vector(cache_you, layer_idx)
        steering_vec = compute_steering_vector(avg_i, avg_you, normalize=True)
        results[layer_idx] = {
            "steering_vec": steering_vec,
            "avg_vec_i": avg_i,
            "avg_vec_you": avg_you
        }

    return results
