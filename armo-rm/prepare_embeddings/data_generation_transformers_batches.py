import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
from tqdm.auto import tqdm
import json
import gc
from safetensors.torch import save_file


# Add these arguments to your argument parser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_index", type=int, default = 0, help="Index of the current batch (starting from 0)")
parser.add_argument("--num_batches", type=int, default = 10, help="how many batches to split the data generation in ")
args = parser.parse_args()

# === Configuration and Define the concepts to label ===
cache_dir = "/cluster/dataset/vogtlab/Group/slaguna/huggingface/"
# model_path = cache_dir + "models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a/" #Running out of memory with this one, changing to a smaller model instead
#  model_name = "Meta-Llama-3-8B-Instruct"

# model_path = cache_dir + "mistral-7b-instruct"  # Smaller memory footprint
# model_name = "mistral-7b-instruct"

model_path = cache_dir + "phi-3-mini-4k-instruct" #Smaller model
model_name = "phi-3-mini-4k-instruct"


dataset_path = cache_dir + "datasets/RLHFlow___ultra_feedback-preference-standard/default/0.0.0/caad75bface3d66c59a14e1d40147a8608a383b0/"
save_path_dir = "/cluster/dataset/vogtlab/Group/slaguna/"
dataset_name = "UltraFeedback-preference-standard"
save_path =  save_path_dir + f"data_RLHF/ArmoRM/labels/{model_name}/{dataset_name}"
torch.backends.cuda.matmul.allow_tf32 = True

concepts = ["helpfulness", "correctness", "coherence", "complexity", "verbosity", "overall_score", "instruction_following", "truthfulness", "honesty",  "is_safe", "score", "overall_quality", "judge_lm", "style", "explanation", "instruction-following", "readability"]
split = "train"  # or "validation"
max_tokens = 190
batch_size = 4
num_batches = args.num_batches
output_file = cache_dir + "datasets/RLHFlow___ultra_feedback-preference-standard/default/0.0.0/caad75bface3d66c59a14e1d40147a8608a383b0/labeled_prompts_llama3.jsonl"

# === Load model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    # local_files_only=True,
    use_fast=False  # explicitly specify this to avoid fast-tokenizer parsing errors, needed for mistral, not for llama
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)
model.eval() 
# llm = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")


# No gradient tracking
def generate(batch, max_tokens):
    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False
        ).cpu()

    # Slice off the prompt to get *only* new generated tokens
    # generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
    # decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    # return decoded
    decoded_full =  tokenizer.batch_decode(outputs, skip_special_tokens=True)
    trimmed_outputs = [full[len(prompt):] if full.startswith(prompt) else full
                   for prompt, full in zip(batch, decoded_full)]
    del inputs, decoded_full, outputs
    torch.cuda.empty_cache()
    gc.collect()    
    return trimmed_outputs

# === Load dataset ===
ds_full = load_dataset(dataset_path, split=split)
batch_len = len(ds_full) // num_batches
start_idx = args.batch_index * batch_len
end_idx = (args.batch_index + 1) * batch_len if args.batch_index < num_batches - 1 else len(ds_full)
ds = ds_full.select(range(start_idx, end_idx))

# === Labeling prompt template ===
def make_label_prompt(prompt_text, concepts):
    concept_str = ", ".join(concepts)
    return (
        f"Please rate the following assistant response for each of the following concepts: {concept_str}.\n\n"
        f"Input user prompt:\n{prompt_text[0]['content']}\n\n"
        f"Assistant Response:\n{prompt_text[1]['content']}\n\n"
        f"For each concept, give a score from 0 (very poor) to 1 (excellent), without an explanation, like this:\n"
        f"helpfulness: 0.2\ncorrectness: 0.5\n...\n\n"
        f"Scores:\n"
    )

def parse_scores(response, concepts):
    scores = {}
    for line in response.strip().split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip().lower()
            val = val.strip()
            try:
                val = float(val)
                scores[key] = val
            except ValueError:
                continue
    return [scores.get(c.lower(), 0.0) for c in concepts]  # Default 0.0 if not found

# === Main loop ===
all_prompt_chosen, all_prompt_rejected = [], []
for example in tqdm(ds, desc="Preparing prompts"):
    chosen, rejected = example["chosen"], example["rejected"]
    if "prompt" in example:
        chosen = [{"role": "user", "content": example["prompt"]}, {"role": "assistant", "content": chosen}]
        rejected = [{"role": "user", "content": example["prompt"]}, {"role": "assistant", "content": rejected}]

    all_prompt_chosen.append(make_label_prompt(chosen, concepts))
    all_prompt_rejected.append(make_label_prompt(rejected, concepts))

# Generate and parse responses
def batch_infer(prompts):
    results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Inference"):
        batch_prompts = prompts[i:i+batch_size]
        outputs = generate(batch_prompts, max_tokens)
        results.extend([parse_scores(o, concepts) for o in outputs])
    return torch.tensor(results)

chosen_scores = batch_infer(all_prompt_chosen)
rejected_scores = batch_infer(all_prompt_rejected)

concepts_label = torch.stack([chosen_scores, rejected_scores], dim=1)

batch_save_path = f"{save_path}_batch_{args.batch_index}"
os.makedirs(os.path.dirname(batch_save_path), exist_ok=True)

# Save the embeddings and prompt embeddings using safetensors
save_file(
    {"concepts_label": concepts_label},
    f"{batch_save_path}.safetensors",
)

# Print a confirmation message with the path to the saved embeddings
print(f"Saved labels to {batch_save_path}.safetensors")