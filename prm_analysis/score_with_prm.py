#!/usr/bin/env python3
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModel, AutoConfig

class PRMScorer:
    def __init__(self, model_name: str = "Gen-Verse/ReasonFlux-PRM-7B", device: Optional[str] = None, max_length: int = 4096):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading PRM model: {model_name}")
        print(f"Device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map="auto" if torch.cuda.is_available() else None)
        self.model.eval()
        print(f"Model loaded: {self.model.__class__.__name__}")

    def score_single(self, problem: str, solution: str) -> Dict[str, float]:
        text = f"Problem: {problem}\n\nSolution: {solution}"
        inputs = self.tokenizer(text, max_length=self.max_length, truncation=True, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=False)
            logits = outputs.logits  # [1, seq_len, 2]
            # Softmax over last token's 2 classes
            probs = torch.softmax(logits[0, -1, :], dim=-1)
            score = probs[1].item()  # P(good)
        
        return {'raw_score': logits[0, -1, 1].item(), 'probability': score}

    def score_batch(self, items: List[Dict], batch_size: int = 8) -> List[Dict]:
        results = []
        for i in tqdm(range(0, len(items), batch_size), desc="Scoring"):
            batch = items[i:i+batch_size]
            for item in batch:
                try:
                    scores = self.score_single(item.get('problem', ''), item.get('solution', ''))
                except Exception as e:
                    print(f"Error: {e}")
                    scores = {'raw_score': 0.0, 'probability': 0.5}
                results.append({**item, 'prm_score': scores['raw_score'], 'prm_probability': scores['probability']})
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Gen-Verse/ReasonFlux-PRM-7B")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("PRM SCORING")
    print("=" * 60)
    
    items = [json.loads(l) for l in open(args.input) if l.strip()]
    if args.limit:
        items = items[:args.limit]
    print(f"Loaded {len(items)} items")

    scorer = PRMScorer(model_name=args.model, max_length=args.max_length)
    results = scorer.score_batch(items, batch_size=args.batch_size)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    scores = [r['prm_probability'] for r in results]
    print(f"\nScored {len(results)} | Min:{min(scores):.3f} Max:{max(scores):.3f} Mean:{sum(scores)/len(scores):.3f}")

if __name__ == "__main__":
    main()
