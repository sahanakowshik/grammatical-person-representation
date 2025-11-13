
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")

model_id = "meta-llama/Llama-3.1-8B-Instruct"

# load model
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    #cache_dir = "/projectnb/vkolagrp/skowshik/.cache/",
    dtype="auto",
    device_map="auto")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def act_add(steering_vec):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h, *rest = output
        else:
            h, rest = output, None
        steer = steering_vec.to(device=h.device, dtype=h.dtype)

        h = h + steer
        
        return (h, *rest) if rest is not None else h
    return hook

def generate_with_steering(
    model,
    tokenizer,
    model_inputs,
    layer_idx,
    steering_vec,
    coeff=5,
    max_new_tokens=50,
    device="cuda"
):
    """
    Generate text while steering model activations in both directions.

    Args:
        model: The transformer model (e.g., Llama, GPT, etc.)
        tokenizer: The tokenizer used with the model
        model_inputs: Tokenized input (output of tokenizer(..., return_tensors="pt"))
        layer_idx: The index of the layer to apply steering on
        steering_vec: The steering vector tensor
        coeff: Magnitude of steering (default: 5)
        max_new_tokens: Number of tokens to generate (default: 50)
        device: Device where tensors are stored (default: "cuda")

    Returns:
        dict with 'positive' and 'negative' generated texts
    """

    results = {}

    for direction, scale in [("positive", coeff), ("negative", -coeff)]:
        # Register hook
        handle = model.model.layers[layer_idx].register_forward_hook(
            act_add(scale * steering_vec)
        )

        # Generate
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )

        # Extract only the newly generated tokens
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)

        # Cleanup hook
        handle.remove()

        # Store result
        results[direction] = generated_text

    return results


emotion_steer_prompts = [
    "Feeling angry", 
    "Am sad", 
    "So scared", 
    "Feel happy", 
    "Kind of jealous", 
    "Feeling silly", 
    "Very interested", 
    "Am excited",
    "Am anxious", 
    "So nervous"
]

creative_steer_prompts = [
    "Interpret the dream", 
    "Tell a story", 
    "Sing a song", 
    "Write a poem", 
    "Make a wish", 
    "Explain the meaning of life", 
    "Provide advice", 
    "Give therapy", 
    "Write a blog", 
    "Give an idea"
]

question_steer_prompts = [
    "What was that sound?", 
    "Why did it snow?", 
    "Where to go from here?", 
    "How did the fire start?", 
    "When was the last baseball game?", 
    "What is the point of learning?", 
    "Why dance?", 
    "Where did it come from?", 
    "How can it be?", 
    "When will the rain stop?"
]

all_steer_prompts = emotion_steer_prompts + creative_steer_prompts + question_steer_prompts

pos_steering_result = {}
neg_steering_result = {}
for layer_idx in tqdm(range(0, 32, 2)):
    #load vec for layer
    steering_vec_path = f"./interp/get_steering_vectors/vectors_llama/layer_{layer_idx}/emotions.pt"
    steering_vec = torch.load(steering_vec_path, map_location=device)["steering_vec"]
    pos_steering_prompt_result = {}
    neg_steering_prompt_result = {}
    for prompt in all_steer_prompts:

        messages = [
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        texts = generate_with_steering(
            model=model,
            tokenizer=tokenizer,
            model_inputs=model_inputs,
            layer_idx=layer_idx,
            steering_vec=steering_vec,
            coeff=5,
            max_new_tokens=100,
            device=device
        )

        pos_steering_prompt_result[prompt] = texts["positive"]
        neg_steering_prompt_result[prompt] = texts["negative"]

    pos_steering_result[layer_idx] = pos_steering_prompt_result
    neg_steering_result[layer_idx] = neg_steering_prompt_result

pd.DataFrame(pos_steering_result).to_csv('./data/steer_results/pos_steer.csv')
pd.DataFrame(neg_steering_result).to_csv('./data/steer_results/neg_steer.csv')