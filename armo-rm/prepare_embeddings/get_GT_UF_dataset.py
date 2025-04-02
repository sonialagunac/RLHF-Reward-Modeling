from datasets import Dataset
from datasets import load_dataset


ds = load_dataset("openbmb/UltraFeedback", split="train")
import itertools
data = []
for example in ds:
    prompt = example['instruction']
    responses = {}
    annotations = {}
    fine_annots = {}
    for row in example['completions']:
        model_name, response = row['model'], row['response']
        responses[model_name] = response
        annotations[model_name] = row['fine-grained_score']
        fine_annots[model_name] = row['annotations'] 
    
    #preference = 'code-' + example['preference']
    all_models = []
    for model_name in responses.keys():
        if annotations[model_name] == 'N/A':
            continue
        all_models.append(model_name)

    all_combinations = list(itertools.combinations(all_models, 2))

    assert len(all_combinations) == len(all_models) * (len(all_models) - 1) / 2

    def extract_ratings(annots):
        keys = ["helpfulness", "honesty", "instruction_following", "truthfulness"]
        ratings = []
        for k in keys:
            rating = annots.get(k, {}).get("Rating", None)
            if rating is not None and rating != "N/A":
                ratings.append(int(rating))
            else:
                ratings.append(-1)  # Or None
        return ratings
    
    for combination in all_combinations:
        response1 = responses[combination[0]]
        rating1 = annotations[combination[0]]
        response2 = responses[combination[1]]
        rating2 = annotations[combination[1]]
        
        if rating1 == rating2:
            continue

        annots1 = fine_annots[combination[0]]
        annots2 = fine_annots[combination[1]]

        if rating1 > rating2: 
            chosen_message = [
                {"content": prompt, "role": "user"},
                        {"content": response1, "role": "assistant"},
            ]
            rejected_message = [
                {"content": prompt, "role": "user"},
                        {"content": response2, "role": "assistant"},
            ]
            chosen_rating = rating1
            rejected_rating = rating2
            chosen_scores = extract_ratings(annots1)
            rejected_scores = extract_ratings(annots2)
    
        elif rating1 < rating2: 
            chosen_message = [
                {"content": prompt, "role": "user"},
                        {"content": response2, "role": "assistant"},
            ]
            rejected_message = [
                {"content": prompt, "role": "user"},
                        {"content": response1, "role": "assistant"},
            ]
            chosen_rating = rating2
            rejected_rating = rating1
            chosen_scores = extract_ratings(annots2)
            rejected_scores = extract_ratings(annots1)
        else:
            print("error")
        
        data.append({"rejected": rejected_message, "chosen": chosen_message, "rejected_score": rejected_rating, "chosen_score": chosen_rating, "rejected_ratings": rejected_scores,
            "chosen_ratings": chosen_scores,})
    
        
dict_data = {
    "rejected": [d['rejected'] for d in data],
    "chosen": [d['chosen'] for d in data],
    "chosen_score": [d['chosen_score'] for d in data],
    "rejected_score": [d['rejected_score'] for d in data],
    "chosen_ratings": [d['chosen_ratings'] for d in data],
    "rejected_ratings": [d['rejected_ratings'] for d in data],
}

dataset = Dataset.from_dict(dict_data)
# Push to hub or save locally
# dataset.push_to_hub("RLHFcollection/UltraFeedback-preference-with-annotations")
# Save to local Hugging Face cache directory
save_path = "/local/home/slaguna/.cache/huggingface/datasets/ultrafeedback_preference_with_annotations"
dataset.save_to_disk(save_path)
print(f"Dataset saved to {save_path}")

