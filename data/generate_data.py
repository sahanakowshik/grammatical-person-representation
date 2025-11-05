import argparse
import json
import os
from typing import List, Dict, Any

from prompt import GEN_INSTRUCTION, build_messages
from utils import set_all_seeds, make_fewshot_block, parse_model_json
from model import LLMWrapper
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yml", help="Path to config JSON.")
    parser.add_argument("--examples", type=str, default="seed_examples.json", help="Path to seed examples JSON.")
    parser.add_argument("--out", type=str, default="generated_examples/", help="Output JSONL file.")
    args = parser.parse_args()

    # cfg = load_config(args.config)
    cfg = OmegaConf.load(args.config)
    set_all_seeds(cfg.get("seed", 42))

    # Load model
    print(f"Loading model {cfg.get("model_id")}")
    model_gen = LLMWrapper(cfg)
    
    # INSERT_YOUR_CODE
    # Ensure output directory exists
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Load seed examples (already grouped by category)
    # grouped = load_seed_examples(args.examples)
    with open(args.examples, "r", encoding="utf-8") as f:
        grouped = json.load(f)

    # Generate text with vLLM
    temperature = cfg.get("temperature", 0.7)
    top_p = cfg.get("top_p", 0.8)
    top_k = cfg.get("top_k", 20)
    
    targets_per_category = int(cfg.get("targets_per_category", 50))
    targets_per_run = int(cfg.get("targets_per_run", 50))
    n_runs = int(cfg.get("n_runs", 50))
    sys_prompt = cfg.get("system_prompt", "You are a helpful assistant.")

    total_written = 0
    for category, seed_pairs in grouped.items():
        out_path = os.path.join(out_dir, f"{category}.jsonl")
        out_f = open(out_path, "w", encoding="utf-8")
    
        fewshot = make_fewshot_block(seed_pairs, max_pairs=20)
        # print(fewshot)
        user_prompt = GEN_INSTRUCTION.format(category=category, count=targets_per_run) + "\n\n" + fewshot
        # print(user_prompt)

        # Build messages for vLLM
        message = build_messages(sys_prompt, user_prompt)
        
        messages = [message for _ in range(n_runs)]
        
        # raise ValueError

        print(f"\n== Category: {category} | targets_per_run: {targets_per_run} ==")
        
        
        completions = model_gen.generate(messages, temperature=temperature, top_p=top_p, top_k=top_k)
        # completions = model_gen.generate_json_schema(messages, temperature=temperature, top_p=top_p, top_k=top_k, count=targets_per_run) # To use vLLM structured outputs using pydantic JSON schema (it's toos slow)
        
        # print(completions)
        # break
        
        # Extract text from vLLM completions
        # vLLM's llm.chat() returns RequestOutput objects
        generated = []
        texts = []

        # Each item in 'completions' corresponds to an output for a given prompt.
        if isinstance(completions, list):
            for idx, request_output in enumerate(completions):
                text = None
                if hasattr(request_output, "outputs") and len(request_output.outputs) > 0:
                    text = request_output.outputs[0].text
                elif hasattr(request_output, "text"):
                    text = request_output.text
                else:
                    text = str(request_output)
                texts.append(text)
        else:
            # Just in case completions is not a list (fallback)
            if hasattr(completions, "outputs") and len(completions.outputs) > 0:
                texts.append(completions.outputs[0].text)
            elif hasattr(completions, "text"):
                texts.append(completions.text)
            else:
                texts.append(str(completions))

        for idx, text in enumerate(texts):
            try:
                this_generated = parse_model_json(text)
                if isinstance(this_generated, list):
                    generated.extend(this_generated)
                else:
                    generated.append(this_generated)
            except Exception as e:
                print(f"[WARN] JSON parse failed for category '{category}', index {idx}: {e}")
                # Save raw for debugging
                dbg_dir = "debug"
                os.makedirs(dbg_dir, exist_ok=True)
                with open(os.path.join(dbg_dir, f"debug_{category}.txt"), "a", encoding="utf-8") as dbg:
                    dbg.write(text + "\n")

        # Post-filter: enforce the pronoun-only constraint
        cleaned: List[Dict[str, str]] = []
        for row in generated:
            i_sent = row["i"]
            you_sent = row["you"]
            # quick heuristic: replace first-person start and compare remainder sans pronoun
            # Ensure they only differ by "I" vs "You" as the subject (case-sensitive to keep it simple)
            if i_sent.startswith("I ") and you_sent.startswith("You "):
                # compare substrings after the first space
                if i_sent.split(".")[-1] == you_sent.split(".")[-1]:
                    cleaned.append(row)

        # Deduplicate within category
        seen = set()
        deduped = []
        for row in cleaned:
            key = (row["i"], row["you"])
            if key not in seen:
                seen.add(key)
                deduped.append(row)

        # Truncate to target_n
        deduped = deduped[:targets_per_category]
                
        print(f"Before removing duplicates: {len(cleaned)}")
        print(f"After removing duplicates: {len(deduped)}")

        # Write to JSONL with category tag
        for row in deduped:
            out = {"category": category, "i": row["i"], "you": row["you"]}
            out_f.write(json.dumps(out, ensure_ascii=True) + "\n")
            total_written += 1

        print(f"Wrote {len(deduped)} pairs for '{category}'. (Total so far: {total_written})")

    out_f.close()
    print(f"\nDone. Wrote {total_written} pairs to {args.out}")
    
    # Clean up model resources
    model_gen.destroy_instance()

if __name__ == "__main__":
    main()



