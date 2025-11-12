from typing import List, Dict
from transformers import AutoTokenizer


GEN_INSTRUCTION = """You are given some EXAMPLES of contrastive pairs. 
Each pair contains two sentences that ask the EXACT SAME THING; the ONLY difference is perspective:
- First sentence uses first-person subject pronoun "I".
- Second sentence uses second-person subject pronoun "You".
- The rest of the sentence MUST be IDENTICAL (including punctuation), except the person change.

TASK:
For the CATEGORY: "{category}", generate EXACTLY {count} NEW contrastive pairs.
Rules:
1) Keep semantics and requests identical across each pair; change ONLY the subject pronoun ("I" ↔ "You").
2) Avoid repeating the seed examples verbatim.
3) Stay safe and non-sensitive; no private data, hate, or instructions for harm.
4) Keep each sentence concise (<= 160 characters).
5) Use plain ASCII quotes and punctuation.

OUTPUT FORMAT (strict JSON list):
[
  {{"i": "...", "you": "..."}},
  ...
]
Do not include any commentary outside the JSON array."""

# For demographics
# GEN_INSTRUCTION = """You are given some EXAMPLES of contrastive pairs. 
# Each pair contains two sentences that ask the EXACT SAME THING; the ONLY difference is perspective:
# - First sentence uses first-person subject pronoun "I".
# - Second sentence uses second-person subject pronoun "You".
# - The rest of the sentence MUST be IDENTICAL (including punctuation), except the person change.

# TASK:
# For the CATEGORY: "{category}", generate EXACTLY {count} NEW contrastive pairs. Only include sentences having age, gender, income, ethnicity, education level, and location. Occupations does not come under demographics.
# Rules:
# 1) Keep semantics and requests identical across each pair; change ONLY the subject pronoun ("I" ↔ "You").
# 2) Avoid repeating the seed examples verbatim.
# 3) Stay safe and non-sensitive; no private data, hate, or instructions for harm.
# 4) Keep each sentence concise (<= 160 characters).
# 5) Use plain ASCII quotes and punctuation.

# OUTPUT FORMAT (strict JSON list):
# [
#   {{"i": "...", "you": "..."}},
#   ...
# ]
# Do not include any commentary outside the JSON array."""


def has_chat_template(tok: AutoTokenizer) -> bool:
    tmpl = getattr(tok, "chat_template", None)
    return tmpl is not None and len(tmpl) > 0


def build_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    return [
        # {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def format_prompt(tokenizer: AutoTokenizer, system_prompt: str, user_prompt: str) -> str:
    """
    If the tokenizer has a chat template (typical for *-Instruct models like Qwen),
    format messages via the template; otherwise, fall back to a plain concatenated prompt.
    """
    if has_chat_template(tokenizer):
        messages = build_messages(system_prompt, user_prompt)
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        return f"<<SYSTEM>>\n{system_prompt}\n\n<<USER>>\n{user_prompt}\n\n<<ASSISTANT>>"