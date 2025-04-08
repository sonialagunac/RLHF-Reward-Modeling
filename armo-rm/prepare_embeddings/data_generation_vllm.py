import os, re
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
import gc
from safetensors.torch import save_file
import openai
import argparse
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
MAX_RETRY = 5 # in case when you prompt an llm the api returns an error (it may sometime happen if you do multiple LLM calls in parallel)
parser = argparse.ArgumentParser()
parser.add_argument("--batch_index", type=int, default=0, help="Index of the current batch (starting from 0)")
parser.add_argument("--num_batches", type=int, default=10, help="How many batches to split the data generation in")
parser.add_argument("--system_prompt", type=str, default=None)
parser.add_argument("--port", type=str, default='8000')
args = parser.parse_args()

# === Config ===
cache_dir = "/cluster/dataset/vogtlab/Group/slaguna/huggingface/"

dataset_path = cache_dir + "datasets/RLHFlow___ultra_feedback-preference-standard/default/0.0.0/caad75bface3d66c59a14e1d40147a8608a383b0/"
save_path_dir = "/cluster/dataset/vogtlab/Group/slaguna/"
dataset_name = "UltraFeedback-preference-standard"
model_id = "/cluster/dataset/vogtlab/Group/slaguna/huggingface/phi-3-mini-4k-instruct" #Smaller model
model_name = "phi-3-mini-4k-instruct"
# model_id = "/cluster/dataset/vogtlab/Group/slaguna/huggingface/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a/" #Running out of memory with this one, changing to a smaller model instead
# model_name = "Meta-Llama-3-8B-Instruct"
save_path = save_path_dir + f"data_RLHF/ArmoRM/labels/{model_name}/{dataset_name}"

# dataset_path = "RLHFlow/UltraFeedback-preference-standard"
# dataset_name = "UltraFeedback-preference-standard"
# model_name = "Qwen1.5-1.8B-Chat"
# model_id = "Qwen/Qwen1.5-1.8B-Chat"
# save_path = "/local/home/slaguna/Projects/datasets/RLHF_UltraFeedback"

concepts = ["helpfulness", "correctness", "coherence", "complexity", "verbosity", "overall_score", "instruction_following", "truthfulness", "honesty", "is_safe", "score", "overall_quality", "judge_lm", "style", "explanation_score", "readability"]
split = "train"
max_tokens = 300
batch_size = 10
num_batches = args.num_batches
openai_api_base = f"http://localhost:{args.port}/v1"
openai_api_key = "EMPTY"
client = openai.OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# === Load dataset ===
ds_full = load_dataset(dataset_path, split=split)
batch_len = len(ds_full) // num_batches
start_idx = args.batch_index * batch_len
end_idx = (args.batch_index + 1) * batch_len if args.batch_index < num_batches - 1 else len(ds_full)
ds = ds_full.select(range(start_idx, end_idx))

def call_model_safe(messages):
    try:
        return call_model(messages)
    except Exception as e:
        print(f"Error: {e}")
        return ''


def batch_infer(prompts):
    results = []
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(call_model_safe, p) for p in prompts]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Inference"):
            output = future.result()

            scores = parse_scores(output, concepts)
            results.append(scores)
    return results


def build_rating_chat(system_prompt, user_prompt, response_a, response_b, concepts):
    concept_str = ", ".join(concepts)
    instruction = (
        f"Rate which of the two assistant responses is better for each of the following concepts: {concept_str}.\n"
        f"Provide one score per concept, between 1 (first response is clearly better) and 0 (second response is clearly better).\n"
        f"Respond strictly in JSON format without any explanation.\n"
        f"Example:\n"
        f"{{\n"
        + "\n".join([f'  "{c}": <score>,' for c in concepts]) + "\n"
        f"}}\n\n"
        f"Now rate the two responses."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": response_a},
        {"role": "assistant", "content": response_b},
        {"role": "user", "content": instruction}
    ]
    return messages

def parse_scores(text, concepts):
    """
    Extract and parse JSON object from possibly messy text.
    Removes ```json blocks and extra text around JSON.
    """
    # Remove markdown ticks if present
    scores = {}
    text = text.strip()
    if text == '':
        return torch.tensor([0.5 for _ in concepts])
    
    if text.startswith("```json"):
        text = text.lstrip("```json").rstrip("```").strip()
    elif text.startswith("```"):
        text = text.lstrip("```").rstrip("```").strip()
    

    # Extract JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            json_obj =  json.loads(json_str)
            for c in concepts:
                scores[c.lower()] = json_obj.get(c, 0.5)
        except json.JSONDecodeError as e:
            pass
    if not scores:
        for line in text.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                # key = key.strip().lower().strip('"') 
                key = key.strip().lower().strip('"').replace(" ", "_").replace("-", "_")
                val = val.strip().replace(",", "")
                try:
                    val = float(val)
                    scores[key] = val
                except ValueError:
                    continue  # skip non-numeric
    # Build tensor
    if len(scores) != len(concepts):
        print(f"Missing concepts in output, parsing them with 0.5")
        print(text)
    result = [scores.get(c.lower().replace("-", "_"), 0.5) for c in concepts]
    try: 
        result = torch.tensor(result)
    except:
        result_clean = [x if isinstance(x, (float, int)) else 0.5 for x in result]
        result = torch.tensor(result_clean)
        print(f"Parsing string score with 0.5 because of text present")
    if result.shape != torch.Size([len(concepts)]):
        print(f"Invalid scores shape: {result.shape}. Expected: [{len(concepts)}]")
        result = torch.full((len(concepts),), 0.5)
    return result


def call_model(
        messages=None,
        n_used=1,
        seed=None,
        temperature=0.7,
        top_p=0.95,
        llm_name='llama-3-70B'
    ):

    success = False
    it = 0
    llm_name_used = model_id

    while not success and it < MAX_RETRY:
        it += 1
        try: 
            response = client.chat.completions.create(
                    model=llm_name_used,
                    messages=messages,
                    max_tokens=max_tokens,
                    n=n_used,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                )
            output = response.choices[0].message.content

        except:
            output = ''
            print("not getting the full list of concepts bc max_tokens reached")
        try:
            output is not None
            success = True
        except:
            print("---------- NOT PASS -------------")
            pass

    if not success:
        raise RuntimeError("Failed after 5 attempts.")
    return  output

if __name__ == "__main__":
    # === Main loop ===
    all_prompts = []
    for example in tqdm(ds, desc="Preparing prompts"):
        if args.system_prompt is None:
            args.system_prompt = "You are an AI assistant that helps scoring a system reponse." 
        user_prompt = example["chosen"][0]['content']
        response_a = example["chosen"][1]['content']
        response_b = example["rejected"][1]['content']
        all_prompts.append(build_rating_chat( args.system_prompt, user_prompt, response_a, response_b, concepts))

    print(f"Starting chosen batch {args.batch_index}")
    scores = batch_infer(all_prompts)

    concepts_label = torch.stack(scores, dim=1)

    batch_save_path = f"{save_path}_batch_{args.batch_index}"
    os.makedirs(os.path.dirname(batch_save_path), exist_ok=True)

    save_file(
        {"concepts_label": concepts_label},
        f"{batch_save_path}.safetensors",
    )

    print(f"Saved labels to {batch_save_path}.safetensors")
