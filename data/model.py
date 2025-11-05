from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from vllm.sampling_params import StructuredOutputsParams
from transformers import AutoTokenizer
import torch
import gc
import contextlib
import ray
from typing import List, Optional
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)


class ContrastivePair(BaseModel):
    """A single contrastive pair with 'I' and 'You' perspectives."""
    i: str = Field(
        ..., 
        description="Sentence using first-person pronoun 'I'",
        max_length=160
    )
    you: str = Field(
        ..., 
        description="Sentence using second-person pronoun 'You'",
        max_length=160
    )


def create_contrastive_pairs_schema(exact_count: int = None, min_count: int = 1, max_count: int = 100):
    """
    Create a ContrastivePairs schema with dynamic list constraints.
    
    Args:
        exact_count: If provided, both min and max will be set to this value
        min_count: Minimum number of pairs (ignored if exact_count is set)
        max_count: Maximum number of pairs (ignored if exact_count is set)
    """
    if exact_count is not None:
        min_count = max_count = exact_count
    
    class ContrastivePairs(BaseModel):
        """Collection of contrastive pairs for guided decoding."""
        pairs: List[ContrastivePair] = Field(
            ...,
            description="List of contrastive sentence pairs",
            min_length=min_count,
            max_length=max_count
        )
    
    return ContrastivePairs


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

    def generate_json_schema(
        self, 
        messages, 
        temperature=0.7, 
        top_p=0.8, 
        top_k=20,
        count: int = None,
    ):
        """
        Generate contrastive pairs with guided decoding to enforce JSON schema.
        
        Args:
            messages: Chat messages in the format expected by llm.chat()
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            guided_decoding_backend: Backend for guided decoding ("outlines" or "lm-format-enforcer")
        
        Returns:
            List of completions with structured output conforming to ContrastivePairs schema
        """
        
        # Create sampling params with guided decoding
        json_schema = create_contrastive_pairs_schema(exact_count=count).model_json_schema()
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=self.max_new_tokens,
            structured_outputs=StructuredOutputsParams(
                json=json_schema
            )
            # guided_decoding=GuidedDecodingParams(
            #     json=json_schema,
            # )
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