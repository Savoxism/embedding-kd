import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

def load_student_model(model_name="google-bert/bert-base-uncased", checkpoint_path="./twmd_checkpoints/best_model.pt"):
    print(f"Loading tokenizer {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading base model {model_name}...")
    model = AutoModel.from_pretrained(model_name)
    
    print(f"Loading distilled weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Load the student state dict
    if "student_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["student_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return tokenizer, model

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    # Get CLS token embedding
    return outputs.last_hidden_state[:, 0, :]

def calculate_similarity(text1, text2, tokenizer, model):
    emb1 = get_embedding(text1, tokenizer, model)
    emb2 = get_embedding(text2, tokenizer, model)
    
    # Compute Cosine Similarity
    sim = F.cosine_similarity(emb1, emb2).item()
    return sim

import argparse

def main():
    parser = argparse.ArgumentParser(description="Test TWMD Student Model Inference")
    parser.add_argument("--checkpoint", type=str, default="./twmd_checkpoints/student_epoch_3.pt", help="Path to the saved checkpoint (.pt file)")
    parser.add_argument("--model", type=str, default="google-bert/bert-base-uncased", help="Base model name")
    args = parser.parse_args()
    
    checkpoint_path = args.checkpoint
    model_name = args.model
    
    try:
        tokenizer, model = load_student_model(model_name, checkpoint_path)
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first or update the --checkpoint argument.")
        return

    print("\n" + "="*50)
    print("TEST 1: Semantic Textual Similarity (STS)")
    print("="*50)
    
    pairs = [
        ("The cat sits outside", "A cat is resting outdoors"),
        ("The cat sits outside", "A man is playing guitar"),
        ("A woman is slicing a potato", "A woman is cutting a potato"),
        ("A woman is slicing a potato", "The dog is chasing a frisbee"),
    ]
    
    for text1, text2 in pairs:
        sim = calculate_similarity(text1, text2, tokenizer, model)
        print(f"Sentence 1 : {text1}")
        print(f"Sentence 2 : {text2}")
        print(f"Similarity : {sim:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    main()
