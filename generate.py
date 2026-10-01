"""
Generates continuations from a trained model.
Usage: python generate.py output_dir/runs/<run>/final ["Свой промпт" ...] [--max-new-tokens 100]
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from your_solution import GENERATION_PROMPTS, generate_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('prompts', nargs='*', default=GENERATION_PROMPTS)
    parser.add_argument('--max-new-tokens', type=int, default=80)
    parser.add_argument('--out', default=None, help="optional path of a JSON file to save the samples to")
    args = parser.parse_args()

    on_gpu = torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16 if on_gpu else torch.float32,
        attn_implementation='flash_attention_2' if on_gpu else 'sdpa',
    ).to('cuda' if on_gpu else 'cpu')

    samples = generate_samples(model, tokenizer, args.prompts, max_new_tokens=args.max_new_tokens)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
