from typing import Dict, List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class ArmoRMPipeline:
    def __init__(self, model_id, cache_dir, device_map="auto", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=False, max_length=4096):
        self.cache_dir = cache_dir
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            cache_dir=self.cache_dir
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            cache_dir=self.cache_dir
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        """
        messages: OpenAI chat messages to be scored
        Note: no batching since due to length differences, the model will have to pad to the max length which is not efficient
        Returns: a dictionary with the score between 0 and 1
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            score = output.score.float().item()
        return {"score": score}

# Define cache directory path
cache_dir = "/cluster/dataset/vogtlab/Group/slaguna/huggingface/" 
path = cache_dir + "models--RLHFlow--ArmoRM-Llama3-8B-v0.1/snapshots/eb2676d20da2f2d41082289d23c59b9f7427f955/"  # Ensure this is the right path

# Create Reward Model Pipeline 
prompt = 'What are some synonyms for the word "beautiful"?'
rm = ArmoRMPipeline(
    path, #Loading it from cache for cluster "RLHFlow/ArmoRM-Llama3-8B-v0.1",
    cache_dir=cache_dir,
    trust_remote_code=True
)

# score the messages
response1 = 'Nicely, Beautifully, Handsome, Stunning, Wonderful, Gorgeous, Pretty, Stunning, Elegant'
score1 = rm([{"role": "user", "content": prompt}, {"role": "assistant", "content": response1}])
print(score1)

response2 = '''Certainly! Here are some synonyms for the word "beautiful":

1. Gorgeous
2. Lovely
3. Stunning
4. Attractive
5. Pretty
6. Elegant
7. Exquisite
8. Handsome
9. Charming
10. Alluring
11. Radiant
12. Magnificent
13. Graceful
14. Enchanting
15. Dazzling

These synonyms can be used in various contexts to convey the idea of beauty.'''
score2 = rm([{"role": "user", "content": prompt}, {"role": "assistant", "content": response2}])
print(score2)

response3 = 'Sorry i cannot answer this.'
score3 = rm([{"role": "user", "content": prompt}, {"role": "assistant", "content": response3}])
print(score3)
