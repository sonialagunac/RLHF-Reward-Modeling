import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
from tqdm.auto import tqdm
import json
from safetensors.torch import save_file

# === Configuration and Define the concepts to label ===
cache_dir = "/cluster/dataset/vogtlab/Group/slaguna/huggingface/"
model_path = cache_dir + "/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a/"
dataset_path = cache_dir + "datasets/RLHFlow___ultra_feedback-preference-standard/default/0.0.0/caad75bface3d66c59a14e1d40147a8608a383b0/"
save_path_dir = "/cluster/dataset/vogtlab/Group/slaguna/"
dataset_name = "UltraFeedback-preference-standard"
save_path =  save_path_dir + f"data_RLHF/ArmoRM/labels/{dataset_name}"

concepts = ["helpfulness", "correctness", "coherence", "complexity", "verbosity", "overall_score", "instruction_following", "truthfulness", "honesty",  "is_safe", "score", "overall_quality", "judge_lm", "style", "explanation", "instruction-following", "readability"]
split = "train"  # or "validation"
max_tokens = 2048
output_file = cache_dir + "datasets/RLHFlow___ultra_feedback-preference-standard/default/0.0.0/caad75bface3d66c59a14e1d40147a8608a383b0/labeled_prompts_llama3.jsonl"

# === Load model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype=torch.bfloat16
)
model.eval() 
# llm = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)


# No gradient tracking
def generate(prompt, max_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "")

# === Load dataset ===
ds = load_dataset(dataset_path, split=split)

# === Labeling prompt template ===
def make_label_prompt(prompt_text, concepts):
    concept_str = ", ".join(concepts)
    return (
        f"Please rate the following assistant response for each of the following concepts: {concept_str}.\n\n"
        f"Input user prompt:\n{prompt_text[0]['content']}\n\n"
        f"Assistant Response:\n{prompt_text[1]['content']}\n\n"
        f"For each concept, give a score from 0 (very poor) to 1 (excellent), like this:\n"
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

concepts_label = []
# === Main loop ===
for example in tqdm(ds, desc="Labeling prompts"):
    chosen = example["chosen"]
    rejected = example["rejected"]
    if "prompt" in example:
        # Format the data with the standard chat template if prompt is available
        chosen = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": chosen},
        ]
        rejected = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": rejected},
        ]

    prompt_chosen = make_label_prompt(chosen, concepts)
    prompt_rejected = make_label_prompt(rejected, concepts)
    response_chosen = generate(prompt_chosen, max_tokens)
    response_rejected = generate(prompt_rejected, max_tokens)
    concepts_label.append([parse_scores(response_chosen, concepts),parse_scores(response_rejected, concepts) ])    
    
concepts_label = torch.tensor(concepts_label)
# concepts_label = torch.stack(concepts_label)
os.makedirs(os.path.dirname(save_path), exist_ok=True)
file_name = (save_path)
# Save the embeddings and prompt embeddings using safetensors
save_file(
    {"concepts_label": concepts_label},
    f"{save_path}.safetensors",
)

# Print a confirmation message with the path to the saved embeddings
print(f"Saved labels to {save_path}.safetensors")