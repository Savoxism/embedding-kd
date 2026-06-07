import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')

class STSDataset(Dataset):
    def __init__(self, file_path):
        full_path = BASE_DIR / file_path if not os.path.isabs(file_path) else file_path
        self.dataset = pd.read_csv(full_path)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # instruction = "Given a text, Retrieve semantically similar text: "
        instruction=""
        return {
            "sentence1": instruction + self.dataset.iloc[idx]['sentence1'],
            "sentence2": instruction + self.dataset.iloc[idx]['sentence2'],
            "label": torch.tensor(self.dataset.iloc[idx]['score'], dtype=torch.float),
        }
        
def collate_fn(batch, tokenizer, max_len=128):
    s1_list = [item["sentence1"] for item in batch]
    s2_list = [item["sentence2"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    enc1 = tokenizer(
        s1_list,
        truncation=True,
        padding=True,       # chỉ pad theo câu dài nhất trong batch
        max_length=max_len,
        return_tensors="pt"
    )
    enc2 = tokenizer(
        s2_list,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt"
    )

    return {
        "input_ids1": enc1["input_ids"],
        "attention_mask1": enc1["attention_mask"],
        "input_ids2": enc2["input_ids"],
        "attention_mask2": enc2["attention_mask"],
        "labels": labels,
    }

def eval_sts(model, eval_loader):
    preds, labels = [], []
    device = model.device
    
    with torch.amp.autocast('cuda', dtype=torch.float16):
        with torch.no_grad():
            for batch in tqdm(eval_loader):
                input_ids1 = batch["input_ids1"].to(device)
                attn1 = batch["attention_mask1"].to(device)
                input_ids2 = batch["input_ids2"].to(device)
                attn2 = batch["attention_mask2"].to(device)
                label = batch["labels"]


                out1 = model(input_ids=input_ids1, attention_mask=attn1)
                out2 = model(input_ids=input_ids2, attention_mask=attn2)

                # Support both StellaModel (dict) and AutoModel (object)
                if isinstance(out1, dict) and 'pooled' in out1:
                    emb1 = out1['pooled']
                    emb2 = out2['pooled']
                else:
                    emb1 = out1.last_hidden_state[:, 0, :] if hasattr(out1, 'last_hidden_state') else out1['last_hidden_state'][:, 0, :]
                    emb2 = out2.last_hidden_state[:, 0, :] if hasattr(out2, 'last_hidden_state') else out2['last_hidden_state'][:, 0, :]
        
                # cosine similarity
                sim = F.cosine_similarity(emb1, emb2)
                score = (sim + 1) * 2.5  # scale [-1,1] -> [0,5]
        
                preds.extend(score.cpu().numpy())
                labels.extend(label.numpy())
    
    spearman_corr, _ = spearmanr(preds, labels)
    print(f"Spearman: {spearman_corr:.4f}")

    return spearman_corr


def eval_sts_task(model, path_list):
    model.eval()
    print(' eval_sts_task')
    for path in path_list:
        print(path)
        eval_dataset = STSDataset(path)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=lambda x: collate_fn(x, tokenizer)
        )
        eval_sts(model, eval_loader)
    model.train()

from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
import datasets
import numpy as np
import torch

def eval_cls(model, eval_loader):
    preds, labels = [], []
    device = model.device
    
    with torch.amp.autocast('cuda', dtype=torch.float16):
        with torch.no_grad():
            for batch in tqdm(eval_loader):
                input_ids1 = batch["input_ids1"].to(device)
                attn1 = batch["attention_mask1"].to(device)
                label = batch["labels"]

                out1 = model(input_ids=input_ids1, attention_mask=attn1)
                
                # Support both StellaModel (dict with 'pooled') and AutoModel (object/dict with last_hidden_state)
                if isinstance(out1, dict) and 'pooled' in out1:
                    emb1 = out1['pooled']
                else:
                    emb1 = out1.last_hidden_state[:, 0, :] if hasattr(out1, 'last_hidden_state') else out1['last_hidden_state'][:, 0, :]
        
                preds.extend(emb1.cpu().numpy())
                labels.extend(label.numpy())
    
    return preds, labels

class ClasssifyDataset(Dataset):
    def __init__(self, file_path):
        full_path = BASE_DIR / file_path if not os.path.isabs(file_path) else file_path
        self.dataset = pd.read_csv(full_path)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return {
            "text": self.dataset.iloc[idx]['text'],
            "label": torch.tensor(self.dataset.iloc[idx]['label'], dtype=torch.long),
        }

def clf_collate_fn(batch, tokenizer, max_len=512):
    s1_list = [item["text"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    enc1 = tokenizer(
        s1_list,
        truncation=True,
        padding=True,       # chỉ pad theo câu dài nhất trong batch
        max_length=max_len,
        return_tensors="pt"
    )

    return {
        "input_ids1": enc1["input_ids"],
        "attention_mask1": enc1["attention_mask"],
        "labels": labels,
    }


def eval_classification_task(model, path_list):
    model.eval()
    print(' eval classifier')

    for train_path, dev_path in path_list:
        print(dev_path)
        eval_dataset = ClasssifyDataset(dev_path)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=lambda x: clf_collate_fn(x, tokenizer)
        )
        
        train_dataset = ClasssifyDataset(train_path)
        train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=lambda x: clf_collate_fn(x, tokenizer)
        )

        X_train, y_train = eval_cls(model, train_loader)
        X_test, y_test = eval_cls(model, eval_loader)

        clf = LogisticRegression(
            random_state=42,
            n_jobs=1,
            max_iter=200,
            verbose=0,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        scores = {}
        accuracy = accuracy_score(y_test, y_pred)
        scores["accuracy"] = accuracy
        f1 = f1_score(y_test, y_pred, average="macro")
        scores["f1"] = f1
        print(scores)
        
    model.train()


class PairDataset(Dataset):
    def __init__(self, file_path):
        full_path = BASE_DIR / file_path if not os.path.isabs(file_path) else file_path
        self.dataset = pd.read_csv(full_path)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # instruction = "Given a text, Retrieve semantically similar text: "
        instruction=""
        return {
            "sentence1": instruction + self.dataset.iloc[idx]['sentence1'],
            "sentence2": instruction + self.dataset.iloc[idx]['sentence2'],
            "label": torch.tensor(self.dataset.iloc[idx]['label'], dtype=torch.float),
        }
        

def eval_pair(model, eval_loader):
    preds, labels = [], []
    device = model.device
    
    with torch.amp.autocast('cuda', dtype=torch.float16):
        with torch.no_grad():
            for batch in tqdm(eval_loader):
                input_ids1 = batch["input_ids1"].to(device)
                attn1 = batch["attention_mask1"].to(device)
                input_ids2 = batch["input_ids2"].to(device)
                attn2 = batch["attention_mask2"].to(device)
                label = batch["labels"]


                out1 = model(input_ids=input_ids1, attention_mask=attn1)
                out2 = model(input_ids=input_ids2, attention_mask=attn2)

                # Support both StellaModel (dict with 'pooled') and AutoModel (object/dict with last_hidden_state)
                if isinstance(out1, dict) and 'pooled' in out1:
                    emb1 = out1['pooled']
                    emb2 = out2['pooled']
                else:
                    emb1 = out1.last_hidden_state[:, 0, :] if hasattr(out1, 'last_hidden_state') else out1['last_hidden_state'][:, 0, :]
                    emb2 = out2.last_hidden_state[:, 0, :] if hasattr(out2, 'last_hidden_state') else out2['last_hidden_state'][:, 0, :]
        
                # cosine similarity
                sim = F.cosine_similarity(emb1, emb2)
                score = (sim + 1) / 2
        
                preds.extend(score.cpu().numpy())
                labels.extend(label.numpy())
    
    metric = get_metric_pair_classification(preds, labels)
    print(metric)

    return metric

def get_metric_pair_classification(scores, labels):
    best_acc, best_thr = 0, 0
    for thr in np.linspace(0, 1, 200):
        preds = (scores >= thr).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    preds = (scores >= best_thr).astype(int)
    return {
        "best_threshold": best_thr,
        "accuracy": best_acc,
        "f1": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds, average="macro"),
        "average_precision": average_precision_score(labels, scores)
    }


def eval_pair_task(model, path_list):
    model.eval()
    print(' eval_pair_task')
    for path in path_list:
        print(path)
        eval_dataset = PairDataset(path)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=lambda x: collate_fn(x, tokenizer)
        )
        eval_pair(model, eval_loader)
    model.train()

# Evaluation datasets - using local multi-data folder
eval_cls_tasks = [('data/multi-data/banking_train.csv', 
                   'data/multi-data/banking77_validation.csv'),
                  ('data/multi-data/emotion_train.csv', 
                   'data/multi-data/emotion_validation.csv'), 
                  ('data/multi-data/tweet_train.csv', 
                   'data/multi-data/tweet_validation.csv')]

eval_sts_tasks = ['data/multi-data/sick_validation.csv', 
                  'data/multi-data/sts12_validation.csv', 
                  'data/multi-data/stsb_validation.csv']

eval_pair_tasks = ['data/multi-data/mrpc_validation.csv', 
                   'data/multi-data/scitail_validation.csv', 
                   'data/multi-data/wic_validation.csv']

test_cls_tasks = [('data/multi-data/banking_train.csv', 
                   'data/multi-data/banking77_test.csv'),
                  ('data/multi-data/emotion_train.csv', 
                   'data/multi-data/emotion_test.csv'), 
                  ('data/multi-data/tweet_train.csv', 
                   'data/multi-data/tweet_test.csv')]

test_sts_tasks = ['data/multi-data/sick_test.csv', 
                  'data/multi-data/sts12_test.csv', 
                  'data/multi-data/stsb_test.csv']

test_pair_tasks = ['data/multi-data/mrpc_test.csv', 
                   'data/multi-data/scitail_test.csv', 
                   'data/multi-data/wic_test.csv']

