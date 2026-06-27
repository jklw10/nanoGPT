import os
import tiktoken
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

enc = tiktoken.get_encoding("gpt2")

def process_and_save(dataset_dict, output_dir, format_fn):
    os.makedirs(output_dir, exist_ok=True)
    
    for split, dset in dataset_dict.items():
        print(f"Processing {split} split for {output_dir}...")
        
        all_ids = []
        for example in tqdm(dset):
            text = format_fn(example)
            ids = enc.encode_ordinary(text)
            ids.append(enc.eot_token) # Delimit patterns/sessions with <|endoftext|>
            all_ids.extend(ids)
            
        arr = np.array(all_ids, dtype=np.uint16)
        filename = os.path.join(output_dir, f'{split}.bin')
        arr.tofile(filename)
        print(f"Saved {filename} ({len(arr):,} tokens).")

if __name__ == '__main__':
    # -------------------------------------------------------------
    # 1. Supervised Fine-Tuning (SFT) - Alpaca
    # -------------------------------------------------------------
    print("Loading Alpaca SFT dataset...")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    # Split into 98% train, 2% val
    alpaca_splits = alpaca.train_test_split(test_size=0.02, seed=42, shuffle=True)
    alpaca_splits['val'] = alpaca_splits.pop('test')
    
    def format_sft(example):
        instr = example.get('instruction', '')
        inp = example.get('input', '')
        out = example.get('output', '')
        if inp:
            return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
        else:
            return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instr}\n\n### Response:\n{out}"

    process_and_save(alpaca_splits, os.path.join('data', 'sft'), format_sft)

    # -------------------------------------------------------------
    # 2. Chain-of-Thought (CoT) - GSM8K
    # -------------------------------------------------------------
    print("\nLoading GSM8K CoT dataset...")
    gsm8k_train = load_dataset("openai/gsm8k", "main", split="train")
    gsm8k_val = load_dataset("openai/gsm8k", "main", split="test") # Using standard test split as val
    gsm8k_splits = {"train": gsm8k_train, "val": gsm8k_val}
    
    def format_cot(example):
        q = example.get('question', '')
        a = example.get('answer', '')
        return f"Below is a math word problem. Think step-by-step to find the answer.\n\n### Question:\n{q}\n\n### Thought & Solution:\n{a}"

    process_and_save(gsm8k_splits, os.path.join('data', 'cot'), format_cot)