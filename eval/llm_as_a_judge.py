
import pandas as pd
import os

os.environ["HF_HOME"] = "/projectnb/vkolagrp/skowshik/.cache/"
import json

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import gc
import contextlib, argparse
from omegaconf import OmegaConf
import ray
from typing import List, Optional
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)

parser = argparse.ArgumentParser(description="Compute steering vectors from I/You sentence pairs")
parser.add_argument(
    "--config",
    type=str,
    default="config.yml",
    help="Path to config YAML file"
)

args = parser.parse_args()




# cfg = {
#     "model_id": "Qwen/Qwen3-32B",
#     "n_devices": 1,
#     "max_model_len": 10000,
#     "max_new_tokens": 500,
#     "temperature": 0.7,
#     "top_p": 0.8,
#     "top_k": 20
# }
 
# data_path = "/projectnb/cs599m1/projects/grammatical-person/grammatical-person-representation/data/steer_results/pos_steer.csv"
# save_name = "llm_as_a_judge_results/llama31_8B_instruct/pos/run_5.jsonl"

# data_path = "/projectnb/cs599m1/projects/grammatical-person/grammatical-person-representation/data/steer_results/neg_steer.csv"
# save_name = "llm_as_a_judge_results/llama31_8B_instruct/neg/run_5.jsonl"

# data_path = "/projectnb/cs599m1/projects/grammatical-person/grammatical-person-representation/data/steer_results/all_llama_base_steer.csv"
# save_name = "llm_as_a_judge_results/llama31_8B_instruct/neutral/run_5.jsonl"


cfg = OmegaConf.load(args.config)

LLM_prompt = """You will be given the prompt given to a large language model (LLM) and it's response. Analyze the response and output your result as a JSON object with the following structure:
    
{{
  "analyzing": "brief explanation of your reasoning",
  "label": "user or llm or not coherent"
}}

Classification rules:

- user: The response is written as though the assistant is directly engaging with the user, often by offering assistance, asking questions, explaining concepts, or providing guidance (e.g., "Would you like me to tell you a story?" or "I can help you with that if you'd like." or explaining or defining a word). The focus is on the user-assistant interaction, not the assistant's inner persona. The response can be incomplete.
- llm: The response is written from the assistant's own persona, character, or internal perspective, as if it's expressing its own feelings, style, or personality (e.g., "I'm so excited to talk about this!" or "I love learning new things!"). The tone is more self-referential or role-play-like rather than service-oriented. The response can be incomplete.
- not coherent: The response is not in English. Choose this only if most of the words are gibberish and are not English words.

Output only the JSON object and nothing else.

Prompt given to the LLM: "{prompt}"; 
The LLM response: "{response}" """

# Load the data
df = pd.read_csv(cfg["data_path"])
df["prompt"] = df["Unnamed: 0"]
df.drop(["Unnamed: 0"], axis=1, inplace=True)
# df.head()

# Load the model


class LLMWrapper:
    def __init__(self, config):
        """
        Initialize the LLM wrapper from a config mapping.

        Args:
            config: Dict-like object with keys `model_id`, `n_devices`,
                `max_model_len`, and `max_new_tokens`.
        """
        self.model_id = config['model_id']
        self.n_devices = config['n_devices']
        self.max_model_len = config['max_model_len']
        self.max_new_tokens = config['max_new_tokens']
        self.llm = self._load_llm()
        self.tokenizer = self._load_tokenizer()

    def _load_llm(self):
        """
        Create and return an initialized vLLM `LLM` instance using
        the current configuration.

        Returns:
            LLM: Initialized language model ready for inference.
        """
        return LLM(
            model=self.model_id,
            tokenizer=self.model_id,
            tensor_parallel_size=self.n_devices,
            gpu_memory_utilization=0.9,
            max_model_len=self.max_model_len,
            enable_lora=False,
            distributed_executor_backend='mp',
        )

    def _load_tokenizer(self):
        """
        Load and configure the tokenizer corresponding to `model_id`.

        Ensures a valid `pad_token` is set (falls back to `eos_token`).

        Returns:
            transformers.PreTrainedTokenizer: Configured tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side='left')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def generate(self, messages, temperature=0.7, top_p=0.8, top_k=20):
        """
        Run a chat completion without schema enforcement.

        Args:
            messages: Chat messages in the format expected by `llm.chat`.
            temperature: Sampling temperature.
            top_p: Top-p nucleus sampling parameter.
            top_k: Top-k sampling parameter.

        Returns:
            The completions object returned by `vllm.LLM.chat`.
        """
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=self.max_new_tokens,
        )
        completions = self.llm.chat(
            messages, 
            sampling_params,
            chat_template_kwargs={"enable_thinking": False},
        )
        return completions

    def destroy_instance(self):
        """
        Delete the LLM and related distributed state, freeing GPU memory.

        This destroys model/distributed contexts, clears CUDA cache, and
        shuts down Ray.
        """
        # Delete LLM instance
        destroy_model_parallel()
        destroy_distributed_environment()
        del self.llm
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()
        ray.shutdown()
        print("Successfully deleted the llm pipeline and freed the GPU memory.\n\n\n\n")


llm = LLMWrapper(cfg)



# Generate prompts
def build_messages(user_prompt: str):
    return [
        # {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def parse_model_json(output_text: str):
    """
    Extract the first top-level JSON array in the text and parse it.
    """
    try:
        data = json.loads(output_text)
        return data
    except Exception as e:
        return {"analyzing": f"parse_error: {e}", "label": "[Not coherent]"}


# all_columns = {}

with open(cfg["save_name"], "w", encoding="utf-8") as f:
    for i, column in enumerate(df.columns):
        # skip prompt column if needed
        
        if "neutral" in cfg["save_name"] and column != "emotions":
            continue
        
        if column == "prompt":
            continue

        responses = [
            (row['prompt'], row[column]) for i, row in df.iterrows()
        ]

        messages = [
            build_messages(LLM_prompt.format(prompt=response[0], response=response[1]))
            for response in responses
        ]

        completions = llm.generate(
            messages,
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            top_k=cfg["top_k"]
        )

        texts = []
        for request_output in completions:
            if hasattr(request_output, "outputs") and len(request_output.outputs) > 0:
                text = request_output.outputs[0].text
            elif hasattr(request_output, "text"):
                text = request_output.text
            else:
                text = str(request_output)
            texts.append(text)

        labels = [parse_model_json(text) for text in texts]
        prompts = list(df['prompt'])

        # all_columns[column] = labels

        for label, prompt in zip(labels, prompts):
            json.dump({"layer_index": column, "prompt": prompt, **label}, f, ensure_ascii=False)
            f.write("\n")


print(f"Saved all results to {cfg["save_name"]}")






