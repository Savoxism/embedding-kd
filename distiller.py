import os
import time
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.cuda.amp import autocast
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_scheduler
from collections import deque
from tqdm import tqdm
from typing import Optional, Dict, Any, Tuple
import pandas as pd
try:
    from pytorch_optimizer import SAM
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Warning: pytorch_optimizer not installed. SAM optimizer unavailable for TALAS.")
from src.data_utils import TextPairRaw, DualTokenizerCollate
from src.data_utils.dataset_cache import TextPairWithTeacher, DualTokenizerCollateWithTeacher
from src.cache_teacher import cache_teacher_embeddings, load_cached_embeddings
from src.loss import info_nce
from src.pooling import last_token_pool, mean_pooling
from src.criterions.contextual_dynamic_mapping import ContextualDynamicMapping
from src.criterions.dual_space_kd import DualSpaceKD
from src.criterions.emo_embedding_distillation import EMODistillation
from src.criterions.stella_distillation import StellaModel
from src.criterions.stella_distillation import stella_stage1_loss, stella_stage2_loss
from src.criterions.teacher_anchor_kd import TeacherAnchorKD
                
# Use evaluation_automodel for AutoModel (not evaluation_model_define which is for Stella)
from src.evaluation.evaluation_automodel import (
    eval_classification_task,
    eval_pair_task,
    eval_sts_task,
    test_cls_tasks,
    test_pair_tasks,
    test_sts_tasks
)

def is_finite(x: torch.Tensor) -> bool:
    return torch.is_tensor(x) and torch.isfinite(x).all().item()

def grads_are_finite(optim) -> bool:
    for group in optim.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                return False
    return True

class KnowledgeDistiller:
    def __init__(self, config):
        self.config = config
        self.setup_seed(config.seed)
        self.setup_devices()
        self.setup_models()
        self.setup_data()
        self.setup_training()
        
        # Initialize criterion based on method
        if config.distill_method == 'cdm':
            self.criterion = ContextualDynamicMapping(
                tok_student=self.tok_student,
                tok_teacher=self.tok_teacher,
                blending_model_special_token=config.teacher_special_token,
                base_model_special_token=config.student_special_token,
                w_task=config.w_task,
                alpha_dtw=config.alpha_dtw,
                debug_align=config.debug_align
            )
        elif config.distill_method == 'dskd':
            self.criterion = DualSpaceKD(
                student_dim=self.model_student.config.hidden_size,
                teacher_dim=self.model_teacher.config.hidden_size,
                w_task=config.w_task,
                alpha_dtw=config.alpha_dtw
            )
            # Move DSKD to device and add to optimizer
            self.criterion.to(self.device_s)
            self.optimizer.add_param_group({
                "params": self.criterion.parameters(),
                "lr": config.learning_rate
            })
            print("DSKD criterion initialized and added to optimizer")
        elif config.distill_method == 'emo':
            self.criterion = EMODistillation(
                d_teacher=self.model_teacher.config.hidden_size,
                d_student=self.model_student.config.hidden_size,
                k_layers=getattr(config, 'k_layers', 1),
                alpha_ot=getattr(config, 'alpha_ot', 0.1),
                max_iter=getattr(config, 'max_iter_ot', 100),
                teacher_special=getattr(config, 'teacher_special_token', '<s>'),
                student_special=getattr(config, 'student_special_token', '[CLS]')
            )
            # Move EMO to device and add to optimizer
            self.criterion.to(self.device_s)
            self.optimizer.add_param_group({
                "params": self.criterion.parameters(),
                "lr": config.learning_rate
            })
            print("EMO criterion initialized and added to optimizer")
        else:
            self.criterion = None
        
        # Projection layer (will be initialized in first forward)
        self.proj_s2t = None
        
        # Metrics tracking
        self.step_times = []
        self.ma_window = deque(maxlen=50)
        self.warmup_steps = 10
        
    def setup_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Done setup_seed with seed={seed}")
    
    def setup_devices(self):
        if torch.cuda.device_count() >= 2:
            self.device_s = torch.device("cuda:0")  # student
            self.device_t = torch.device("cuda:1")  # teacher
            print(f"Using 2 GPUs: Student on {self.device_s}, Teacher on {self.device_t}")
        elif torch.cuda.is_available():
            self.device_s = self.device_t = torch.device("cuda:0")
            print("[WARN] Only 1 GPU available -> both on cuda:0")
        else:
            self.device_s = self.device_t = torch.device("cpu")
            print("[WARN] No GPU -> CPU training")
        print("Done setup_devices")
    
    def setup_models(self):
        cfg = self.config
        
        print(f"Loading tokenizers...")
        self.tok_student = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.tok_teacher = AutoTokenizer.from_pretrained(cfg.teacher_model_name)
        
        if cfg.distill_method == 'stella':
            print(f"Loading Stella student model: {cfg.student_model_name}")
            self.model_student = StellaModel(
                cfg.student_model_name,
                output_dim1=getattr(cfg, 'output_dim1', 1024),
                pooling=getattr(cfg, 'pooling', 'cls'),
                output_dim2=getattr(cfg, 'output_dim2', 512),
                output_dim3=getattr(cfg, 'output_dim3', 256),
                output_dim4=getattr(cfg, 'output_dim4', 128)
            )
            self.current_stage = 1
        else:
            print(f"Loading student model: {cfg.student_model_name}")
            self.model_student = AutoModel.from_pretrained(cfg.student_model_name)
        
        print(f"Loading teacher model: {cfg.teacher_model_name}")
        teacher_kwargs = {"trust_remote_code": True}
        if cfg.teacher_dtype == "bfloat16":
            teacher_kwargs["torch_dtype"] = torch.bfloat16
        elif cfg.teacher_dtype == "float16":
            teacher_kwargs["torch_dtype"] = torch.float16
        
        # EMO method needs attentions, force eager attention implementation
        if cfg.distill_method == 'emo':
            teacher_kwargs["attn_implementation"] = "eager"
            print("Using eager attention implementation for EMO (required for output_attentions)")
        
        self.model_teacher = AutoModel.from_pretrained(
            cfg.teacher_model_name,
            **teacher_kwargs
        )
        
        self.model_student.to(self.device_s)
        self.model_teacher.to(self.device_t)
        
        self.model_teacher.eval()
        for p in self.model_teacher.parameters():
            p.requires_grad_(False)
        
        print("Models loaded successfully!")
        print("Done setup_models")
    
    def setup_data(self):
        cfg = self.config
        
        print(f"Loading training data from: {cfg.train_data_path}")
        
        df = pd.read_csv(cfg.train_data_path)
        
        if cfg.task_type == "pair_cls":
            if "premise" not in df.columns or "hypothesis" not in df.columns:
                # Create from text column
                df["premise"] = df["text"] if "text" in df.columns else df.iloc[:, 0]
                df["hypothesis"] = df["text"] if "text" in df.columns else df.iloc[:, 0]
        
        # TALAS method uses cached teacher embeddings
        if cfg.distill_method == 'talas':
            cache_path = Path(cfg.cache_path)
            
            # Check if cache exists
            if cache_path.exists():
                print(f"Loading cached teacher embeddings from: {cache_path}")
                teacher_cls_list = load_cached_embeddings(str(cache_path))
                print(f"Loaded {len(teacher_cls_list)} cached embeddings")
            else:
                print(f"Cache not found. Pre-computing teacher embeddings...")
                os.makedirs(cache_path.parent, exist_ok=True)
                
                # Create temporary dataset for caching
                temp_ds = TextPairRaw(df, cfg.task_type)
                temp_collate = DualTokenizerCollate(
                    self.tok_student,
                    self.tok_teacher,
                    cfg.task_type,
                    cfg.max_length
                )
                cache_loader = DataLoader(
                    temp_ds,
                    batch_size=cfg.batch_size,
                    shuffle=False,  # Don't shuffle for caching
                    collate_fn=temp_collate,
                    pin_memory=True,
                    num_workers=cfg.num_workers,
                    persistent_workers=cfg.num_workers > 0
                )
                
                # Cache teacher embeddings
                teacher_cls_list = cache_teacher_embeddings(
                    model_teacher=self.model_teacher,
                    dataloader=cache_loader,
                    device=self.device_t,
                    pooling_method=cfg.pooling_method,
                    normalize=cfg.normalize_cache,
                    dtype=torch.float32 if cfg.cache_dtype == "float32" else torch.float16,
                    cache_path=str(cache_path)
                )
                print(f"Cached {len(teacher_cls_list)} teacher embeddings to {cache_path}")
            
            # Free teacher model to save GPU memory (for TALAS, teacher not needed after caching)
            del self.model_teacher
            self.model_teacher = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Teacher model freed from GPU memory")
            
            self.train_ds = TextPairWithTeacher(df, cfg.task_type, teacher_cls_list)
            
            self.collate_fn = DualTokenizerCollateWithTeacher(
                self.tok_student,
                cfg.task_type,
                cfg.max_length
            )
        else:
            # Standard distillation methods
            self.train_ds = TextPairRaw(df, cfg.task_type)
            
            self.collate_fn = DualTokenizerCollate(
                self.tok_student,
                self.tok_teacher,
                cfg.task_type,
                cfg.max_length
            )
        
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=True,
            num_workers=cfg.num_workers,
            persistent_workers=cfg.num_workers > 0
        )
        
        print(f"Training samples: {len(self.train_ds)}")
        print(f"Training batches: {len(self.train_loader)}")
        print("Done setup_data")
    
    def setup_training(self):
        cfg = self.config
        
        # TALAS optimizer/scheduler will be initialized after criterion creation in train_step
        if cfg.distill_method == 'talas':
            self.optimizer = None
            self.scheduler = None
            self.scaler = GradScaler('cuda', enabled=torch.cuda.is_available())
            print("TALAS: Deferring optimizer/scheduler initialization until criterion is created")
        else:
            self.optimizer = optim.AdamW(
                self.model_student.parameters(),
                lr=cfg.learning_rate
            )
            
            self.scaler = GradScaler('cuda', enabled=torch.cuda.is_available())
            
            num_steps = len(self.train_loader)
            total_steps = num_steps * cfg.epochs
            
            min_lr_rate = cfg.min_lr / cfg.learning_rate
            self.scheduler = get_scheduler(
                name='cosine_with_min_lr',
                optimizer=self.optimizer,
                num_warmup_steps=int(total_steps * cfg.warmup_ratio),
                num_training_steps=total_steps,
                scheduler_specific_kwargs={'min_lr_rate': min_lr_rate}
            )
        
        if cfg.save_dir:
            os.makedirs(cfg.save_dir, exist_ok=True)
            print(f"Checkpoints will be saved to: {cfg.save_dir}")
        print("Done setup_training")
    
    def sync_all(self):
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)
    
    def train_step(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        cfg = self.config
        method = cfg.distill_method
        
        if method == 'talas':
            batch_s = {}
            for k, v in batch.items():
                if not torch.is_tensor(v):
                    continue
                if k.endswith("_stu") or k == "labels" or k == "teacher_cls":
                    batch_s[k] = v.to(self.device_s, non_blocking=True)
            
            # ========== FIRST PASS ==========
            with autocast(enabled=torch.cuda.is_available()):
                teacher_cls = batch_s["teacher_cls"]
                
                s_out1 = self.model_student(
                    input_ids=batch_s["input_ids1_stu"],
                    attention_mask=batch_s["attention_mask1_stu"],
                    output_hidden_states=True,
                    return_dict=True
                )
                s_out2 = self.model_student(
                    input_ids=batch_s["input_ids2_stu"],
                    attention_mask=batch_s["attention_mask2_stu"],
                    output_hidden_states=False,
                    return_dict=True
                )
                
                S_last1 = s_out1.last_hidden_state
                S_last2 = s_out2.last_hidden_state
                S_cls1 = S_last1[:, 0, :]
                S_cls2 = S_last2[:, 0, :]
                
                loss_task, _ = info_nce(S_cls1, S_cls2, temperature=cfg.temperature)
                
                # Initialize TALAS criterion if needed
                if self.criterion is None:
                    d_s = self.model_student.config.hidden_size
                    d_t = teacher_cls.shape[-1]
                    
                    # BERT-base has 13 layers: embedding + 12 transformer layers
                    num_layers = len(s_out1.hidden_states)
                    
                    self.criterion = TeacherAnchorKD(
                        student_dim=d_s,
                        teacher_dim=d_t,
                        num_layers=num_layers,
                        last_layer_idx=cfg.last_layer_idx,
                        start_rkd=cfg.start_rkd,
                        w_task=cfg.w_task,
                        w_kd=cfg.w_kd,
                        w_struct=cfg.w_struct,
                        eps_norm=cfg.eps_norm
                    ).to(self.device_s)
                    
                    # Initialize SAM optimizer with both student and criterion parameters
                    if not SAM_AVAILABLE:
                        raise RuntimeError("SAM optimizer not available. Install pytorch_optimizer.")
                    
                    base_optimizer = optim.AdamW
                    self.optimizer = SAM(
                        [
                            {"params": self.model_student.parameters(), "lr": cfg.learning_rate, "weight_decay": 0.01},
                            {"params": self.criterion.parameters(), "lr": cfg.learning_rate * 5},
                        ],
                        base_optimizer,
                        rho=getattr(cfg, 'rho', 0.05),
                        adaptive=True
                    )
                    
                    # Initialize scheduler
                    num_steps = len(self.train_loader)
                    total_steps = num_steps * cfg.epochs
                    min_lr_rate = cfg.min_lr / cfg.learning_rate
                    self.scheduler = get_scheduler(
                        name='cosine_with_min_lr',
                        optimizer=self.optimizer,
                        num_warmup_steps=int(total_steps * cfg.warmup_ratio),
                        num_training_steps=total_steps,
                        scheduler_specific_kwargs={'min_lr_rate': min_lr_rate}
                    )
                    
                    print(f"Initialized TeacherAnchorKD: {d_s} -> {d_t}, num_layers={num_layers}, last_layer_idx={cfg.last_layer_idx}, start_rkd={cfg.start_rkd}")
                    print(f"Initialized SAM optimizer with rho={getattr(cfg, 'rho', 0.05)}")
                    print(f"Initialized scheduler: {total_steps} steps, warmup={int(total_steps * cfg.warmup_ratio)}")
                
                # Now safe to call criterion with initialized projection heads
                student_outputs = {
                    'hidden_states': s_out1.hidden_states,
                    'last_hidden_state': S_last1
                }
                
                loss, metrics = self.criterion(
                    student_outputs=student_outputs,
                    teacher_cls=teacher_cls,
                    task_loss=loss_task
                )
                
                loss = loss.float()
            
            # Backward pass 1 (this will init gradients for first_step)
            self.scaler.scale(loss).backward()

            
            # Check gradients
            self.scaler.unscale_(self.optimizer)
            if not grads_are_finite(self.optimizer):
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                return loss, {**metrics, 'skip': 'grad_inf_p1'}
            
            # SAM first step
            self.optimizer.first_step(zero_grad=True)
            
            # ========== SECOND PASS ==========
            with autocast(enabled=torch.cuda.is_available()):
                s_out1_2 = self.model_student(
                    input_ids=batch_s["input_ids1_stu"],
                    attention_mask=batch_s["attention_mask1_stu"],
                    output_hidden_states=True,
                    return_dict=True
                )
                s_out2_2 = self.model_student(
                    input_ids=batch_s["input_ids2_stu"],
                    attention_mask=batch_s["attention_mask2_stu"],
                    output_hidden_states=False,
                    return_dict=True
                )
                
                S_last1_2 = s_out1_2.last_hidden_state
                S_last2_2 = s_out2_2.last_hidden_state
                S_cls1_2 = S_last1_2[:, 0, :]
                S_cls2_2 = S_last2_2[:, 0, :]
                
                loss_task_2, _ = info_nce(S_cls1_2, S_cls2_2, temperature=cfg.temperature)
                
                student_outputs_2 = {
                    'hidden_states': s_out1_2.hidden_states,
                    'last_hidden_state': S_last1_2
                }
                
                loss_2, _ = self.criterion(
                    student_outputs=student_outputs_2,
                    teacher_cls=teacher_cls,
                    task_loss=loss_task_2
                )
                
                loss_2 = loss_2.float()
            
            # Check loss_2 is finite
            if not is_finite(loss_2):
                raise RuntimeError(f"loss_2 NaN/Inf")
            
            # Check loss_2 finite before backward
            if not is_finite(loss_2):
                raise RuntimeError(f"loss_2 NaN/Inf at epoch={self.current_epoch} step={self.current_step}")
            
            # Backward pass 2 - IMPORTANT: Do NOT scale (plain backward)
            loss_2.backward()
            
            # Check gradients again
            if not grads_are_finite(self.optimizer):
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                return loss, {**metrics, 'skip': 'grad_inf_p2'}
            
            # SAM second step
            self.optimizer.second_step(zero_grad=True)
            self.scaler.update()
            self.scheduler.step()
            
            # Clean up
            del s_out1, s_out2, s_out1_2, s_out2_2
            del student_outputs, student_outputs_2
            
            return loss, metrics
        
        # Standard distillation methods with teacher inference
        batch_s, batch_t = {}, {}
        for k, v in batch.items():
            if not torch.is_tensor(v):
                continue
            if k.endswith("_stu") or k == "labels":
                batch_s[k] = v.to(self.device_s, non_blocking=True)
            if k.endswith("_tea"):
                batch_t[k] = v.to(self.device_t, non_blocking=True)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        with autocast(enabled=torch.cuda.is_available()):
            need_atts = (method == 'emo')
            with torch.inference_mode():
                t_out1 = self.model_teacher(
                    input_ids=batch_t["input_ids1_tea"],
                    attention_mask=batch_t["attention_mask1_tea"],
                    output_attentions=need_atts,
                    return_dict=True
                )
                T_last1 = t_out1.last_hidden_state
                T_cls1 = last_token_pool(T_last1, batch_t["attention_mask1_tea"])
                
                T_last1 = T_last1.to(self.device_s, non_blocking=True)
                T_cls1 = T_cls1.to(self.device_s, non_blocking=True)
                
                if need_atts:
                    T_atts = tuple(att.to(self.device_s, non_blocking=True) for att in t_out1.attentions)
            
            # Different models have different forward signatures
            if method == 'stella':
                # StellaModel doesn't accept output_attentions or return_dict
                s_out1 = self.model_student(
                    input_ids=batch_s["input_ids1_stu"],
                    attention_mask=batch_s["attention_mask1_stu"]
                )
                s_out2 = self.model_student(
                    input_ids=batch_s["input_ids2_stu"],
                    attention_mask=batch_s["attention_mask2_stu"]
                )
            elif method == 'emo':
                # EMO needs attentions
                s_out1 = self.model_student(
                    input_ids=batch_s["input_ids1_stu"],
                    attention_mask=batch_s["attention_mask1_stu"],
                    output_attentions=True,
                    return_dict=True
                )
                s_out2 = self.model_student(
                    input_ids=batch_s["input_ids2_stu"],
                    attention_mask=batch_s["attention_mask2_stu"],
                    return_dict=True
                )
            else:
                # CDM, DSKD - standard transformers models
                s_out1 = self.model_student(
                    input_ids=batch_s["input_ids1_stu"],
                    attention_mask=batch_s["attention_mask1_stu"],
                    return_dict=True
                )
                s_out2 = self.model_student(
                    input_ids=batch_s["input_ids2_stu"],
                    attention_mask=batch_s["attention_mask2_stu"],
                    return_dict=True
                )
            if method != 'stella':
                S_last1 = s_out1.last_hidden_state
                S_last2 = s_out2.last_hidden_state
                S_cls1 = S_last1[:, 0, :]
                S_cls2 = S_last2[:, 0, :]
            else:
                S_cls1 = s_out1["pooled"]
                S_cls2 = s_out2["pooled"]
            
            loss_task, _ = info_nce(S_cls1, S_cls2, temperature=cfg.temperature)
            
            # ========== Method-specific KD loss ==========
            if method == 'cdm':
                if self.proj_s2t is None:
                    d_s, d_t = S_cls1.size(-1), T_cls1.size(-1)
                    self.proj_s2t = nn.Linear(d_s, d_t, bias=False).to(self.device_s)
                    self.optimizer.add_param_group({
                        "params": self.proj_s2t.parameters(),
                        "lr": cfg.learning_rate * 2
                    })
                    print(f"Initialized projection layer: {d_s} -> {d_t}")
                
                keep_s1 = (batch_s["attention_mask1_stu"].bool() & 
                          (~batch_s["special_tokens_mask1_stu"].bool()))
                keep_t1 = (batch_t["attention_mask1_tea"].to(self.device_s).bool() &
                          (~batch_t["special_tokens_mask1_tea"].to(self.device_s).bool()))
                
                kd_dtw = self.criterion.compute_cdm_loss(
                    S_last=S_last1,
                    T_last=T_last1,
                    batch_input_ids_stu=batch["input_ids1_stu"],
                    batch_input_ids_tea=batch["input_ids1_tea"],
                    keep_mask_stu=keep_s1,
                    keep_mask_tea=keep_t1,
                    proj_s2t=self.proj_s2t,
                    device_s=self.device_s,
                    epoch=self.current_epoch,
                    step=self.current_step
                )
                
                S_proj_cls1 = self.proj_s2t(S_cls1)
                S_proj_cls1_norm = F.normalize(S_proj_cls1, p=2, dim=-1)
                T_cls1_norm = F.normalize(T_cls1, p=2, dim=-1)
                kd_cls = F.mse_loss(S_proj_cls1_norm, T_cls1_norm)
                
                loss = (cfg.w_task * loss_task + 
                       cfg.alpha_dtw * kd_dtw * 100 +
                       cfg.w_cls * kd_cls)
                
                metrics = {
                    'loss_total': loss.item(),
                    'loss_task': loss_task.item(),
                    'loss_kd_dtw': kd_dtw.item() if isinstance(kd_dtw, torch.Tensor) else kd_dtw,
                    'loss_kd_cls': kd_cls.item(),
                }
                
            elif method == 'dskd':
                mask_s1 = batch_s["attention_mask1_stu"]
                mask_t1 = batch_t["attention_mask1_tea"].to(self.device_s)
                
                spec_s1 = batch_s.get("special_tokens_mask1_stu", None)
                spec_t1 = batch_t.get("special_tokens_mask1_tea", None)
                if spec_t1 is not None:
                    spec_t1 = spec_t1.to(self.device_s)
                
                loss, metrics = self.criterion.compute_dskd_loss(
                    S_last=S_last1,
                    T_last=T_last1,
                    S_cls=S_cls1,
                    T_cls=T_cls1,
                    mask_student=mask_s1,
                    mask_teacher=mask_t1,
                    task_loss=loss_task,
                    special_tokens_mask_student=spec_s1,
                    special_tokens_mask_teacher=spec_t1,
                    device=self.device_s
                )
                
            elif method == 'emo':
                class TeacherOutput:
                    def __init__(self, last_hidden_state, attentions):
                        self.last_hidden_state = last_hidden_state
                        self.attentions = attentions
                
                class StudentOutput:
                    def __init__(self, last_hidden_state, attentions):
                        self.last_hidden_state = last_hidden_state
                        self.attentions = attentions
                
                teacher_outputs = TeacherOutput(T_last1, T_atts)
                student_outputs = StudentOutput(S_last1, s_out1.attentions)
                
                att_loss_weight = getattr(cfg, 'att_loss_weight', 0.1)
                ot_loss_weight = getattr(cfg, 'ot_loss_weight', 1.0)
                
                kd_loss, kd_metrics = self.criterion.compute_emo_loss(
                    teacher_outputs=teacher_outputs,
                    student_outputs=student_outputs,
                    input_ids_tea=batch_t["input_ids1_tea"].to(self.device_s),
                    input_ids_stu=batch_s["input_ids1_stu"],
                    attention_mask_tea=batch_t["attention_mask1_tea"].to(self.device_s),
                    attention_mask_stu=batch_s["attention_mask1_stu"],
                    tok_teacher=self.tok_teacher,
                    tok_student=self.tok_student,
                    att_loss_weight=att_loss_weight,
                    ot_loss_weight=ot_loss_weight
                )
                
                w_task = getattr(cfg, 'w_task', 0.5)
                alpha_kd = getattr(cfg, 'alpha_kd', 0.5)
                loss = w_task * loss_task + alpha_kd * kd_loss
                
                metrics = {
                    'loss_total': loss.item(),
                    'loss_task': loss_task.item(),
                    **kd_metrics
                }
            
            elif method == 'stella':
                
                if self.current_stage == 1:
                    S_emb = s_out1["fc1"]
                    loss, metrics = stella_stage1_loss(
                        S_emb, T_cls1,
                        w_cos=getattr(cfg, 'w_cos_stage1', 10.0),
                        w_sim=getattr(cfg, 'w_sim_stage1', 200.0),
                        w_tri=getattr(cfg, 'w_tri_stage1', 20.0)
                    )
                else:
                    loss, metrics = stella_stage2_loss(
                        S_cls1, S_cls2,
                        s_out1["fc1"], s_out1["fc2"], s_out1["fc3"], s_out1["fc4"],
                        T_cls1,
                        temperature=cfg.temperature,
                        w_task=cfg.w_task,
                        w_cos=getattr(cfg, 'w_cos_stage2', 10.0),
                        w_sim=getattr(cfg, 'w_sim_stage2', 200.0),
                        w_tri=getattr(cfg, 'w_tri_stage2', 20.0)
                    )
            
            else:
                raise ValueError(f"Unknown distillation method: {method}")
            
            loss = loss.float()
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        
        return loss, metrics
    
    def train_epoch(self, epoch: int):
        self.model_student.train()
        self.current_epoch = epoch
        
        total_loss = 0.0
        n_items = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
        
        for step, batch in enumerate(pbar):
            self.current_step = step
            
            self.sync_all()
            t0 = time.perf_counter()
            
            loss, metrics = self.train_step(batch)
            
            self.sync_all()
            dt = time.perf_counter() - t0
            
            bs = batch["input_ids1_stu"].size(0)
            total_loss += loss.item() * bs
            n_items += bs
            avg_loss = total_loss / max(1, n_items)
            
            mem_info = {}
            for dev_id in range(torch.cuda.device_count()):
                mem_alloc = torch.cuda.memory_allocated(dev_id) / 1024**2
                mem_reserved = torch.cuda.memory_reserved(dev_id) / 1024**2
                mem_info[f"gpu{dev_id}"] = f"{mem_alloc:.0f}/{mem_reserved:.0f}MB"
            
            if step >= self.warmup_steps:
                self.step_times.append(dt)
                self.ma_window.append(dt)
                avg_step = sum(self.step_times) / len(self.step_times)
                ma_step = sum(self.ma_window) / len(self.ma_window)
                
                postfix = {
                    "avg_loss": f"{avg_loss:.4f}",
                    "ms/step": f"{avg_step*1000:.1f}",
                    "ms/step(ma)": f"{ma_step*1000:.1f}",
                    "it/s": f"{1.0/ma_step:.2f}",
                    **mem_info
                }
                
                for k, v in metrics.items():
                    if k != 'loss_total':
                        # Format only if v is numeric (not string like 'skip': 'grad_inf_p1')
                        postfix[k] = f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
                
                pbar.set_postfix(postfix)
            else:
                pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}", **mem_info})
        
        if len(self.step_times) > 0:
            epoch_avg = sum(self.step_times) / len(self.step_times)
            print(f"[Epoch {epoch+1}] Avg step time = {epoch_avg*1000:.2f} ms "
                  f"({1.0/epoch_avg:.2f} it/s)")
        
        print(f"Done train_epoch {epoch+1}")
        return avg_loss
    
    def save_checkpoint(self, epoch: int, metrics: Optional[Dict] = None):
        cfg = self.config
        if not cfg.save_dir:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model_student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': cfg.__dict__ if hasattr(cfg, '__dict__') else cfg,
        }
        
        if self.proj_s2t is not None:
            checkpoint['proj_s2t_state_dict'] = self.proj_s2t.state_dict()
        
        if metrics:
            checkpoint['metrics'] = metrics
        
        path = os.path.join(cfg.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
        print(f"Done save_checkpoint for epoch {epoch+1}")
        
        if cfg.save_best and metrics and 'loss' in metrics:
            if not hasattr(self, 'best_loss') or metrics['loss'] < self.best_loss:
                self.best_loss = metrics['loss']
                best_path = os.path.join(cfg.save_dir, 'best_model.pt')
                torch.save(checkpoint, best_path)
                print(f"Best model saved: {best_path}")
    
    def train(self):
        cfg = self.config
        
        if cfg.distill_method == 'stella':
            print("\n" + "="*70)
            print("Starting Stella 2-Stage Training...")
            print("="*70)
            print(f"Student: {cfg.student_model_name}")
            print(f"Teacher: {cfg.teacher_model_name}")
            print(f"Stage 1 epochs: {cfg.epochs_stage1}")
            print(f"Stage 2 epochs: {cfg.epochs_stage2}")
            print(f"Batch size: {cfg.batch_size}")
            print(f"Learning rate: {cfg.learning_rate}")
            print("="*70 + "\n")
            
            print("\n" + "="*70)
            print("STAGE 1: Freeze backbone + fc2,3,4, train fc1 only")
            print("="*70)
            
            for p in self.model_student.backbone.parameters():
                p.requires_grad = False
            for head in [self.model_student.fc2, self.model_student.fc3, self.model_student.fc4]:
                for p in head.parameters():
                    p.requires_grad = False
            
            print("Frozen: backbone, fc2, fc3, fc4")
            print("Trainable: fc1")
            
            self.current_stage = 1
            for epoch in range(cfg.epochs_stage1):
                avg_loss = self.train_epoch(epoch)
                
                if (epoch + 1) % cfg.save_every == 0:
                    self.save_checkpoint(epoch, {'loss': avg_loss})
            
            print("\n" + "="*70)
            print("STAGE 1 COMPLETED!")
            print("="*70 + "\n")
            
            print("\n" + "="*70)
            print("STAGE 2: Unfreeze all, train full model")
            print("="*70)
            
            for p in self.model_student.parameters():
                p.requires_grad = True
            
            print("Unfrozen: all parameters")
            print("Trainable: backbone, fc1, fc2, fc3, fc4")
            
            self.optimizer = optim.AdamW(
                self.model_student.parameters(),
                lr=cfg.learning_rate
            )
            self.scheduler = get_scheduler(
                "cosine",
                optimizer=self.optimizer,
                num_warmup_steps=int(len(self.train_loader) * cfg.warmup_ratio),
                num_training_steps=len(self.train_loader) * cfg.epochs_stage2
            )
            
            self.step_times = []
            self.ma_window = deque(maxlen=50)
            
            self.current_stage = 2
            for epoch in range(cfg.epochs_stage2):
                avg_loss = self.train_epoch(epoch)
                
                print("\n" + "="*60)
                print(f"Evaluation after Stage2 Epoch {epoch+1}")
                print("="*60)
                
                try:
                    # from src.evaluation.evaluation_model_define import (
                    #     eval_classification_task,
                    #     eval_pair_task,
                    #     eval_sts_task,
                    #     test_cls_tasks,
                    #     test_pair_tasks,
                    #     test_sts_tasks
                    # )
                    eval_classification_task(self.model_student, test_cls_tasks)
                    eval_pair_task(self.model_student, test_pair_tasks)
                    eval_sts_task(self.model_student, test_sts_tasks)
                except Exception as e:
                    print(f"Warning: Evaluation failed with error: {e}")
                    print("Continuing training...")
                
                print("="*60 + "\n")
                
                if (epoch + 1) % cfg.save_every == 0:
                    self.save_checkpoint(epoch, {'loss': avg_loss})
            
            print("\n" + "="*70)
            print("STAGE 2 COMPLETED!")
            print("="*70)
            
            self.save_checkpoint(cfg.epochs_stage2 - 1, {'loss': avg_loss})
            
            print("\n" + "="*70)
            print("Training completed successfully!")
            print("="*70)
            
        else:
            print("\n" + "="*60)
            print("Starting training...")
            print("="*60)
            print(f"Method: {cfg.distill_method}")
            print(f"Student: {cfg.student_model_name}")
            print(f"Teacher: {cfg.teacher_model_name}")
            print(f"Epochs: {cfg.epochs}")
            print(f"Batch size: {cfg.batch_size}")
            print(f"Learning rate: {cfg.learning_rate}")
            print("="*60 + "\n")
            
            for epoch in range(cfg.epochs):
                avg_loss = self.train_epoch(epoch)
                
                print("\n" + "="*60)
                print(f"Evaluation after Epoch {epoch+1}")
                print("="*60)
                
                try:
                    eval_classification_task(self.model_student, test_cls_tasks)
                    eval_pair_task(self.model_student, test_pair_tasks)
                    eval_sts_task(self.model_student, test_sts_tasks)
                except Exception as e:
                    print(f"Warning: Evaluation failed with error: {e}")
                    print("Continuing training...")
                
                print("="*60 + "\n")
                
                if (epoch + 1) % cfg.save_every == 0:
                    try:
                        self.save_checkpoint(epoch, {'loss': avg_loss})
                    except Exception as e:
                        print(f"Warning: Saving checkpoint failed with error: {e}")
                        print("Continuing training...")
            
            print("\n" + "="*60)
            print("Training completed!")
            print("="*60)
            print("Done train()")
            
            self.save_checkpoint(cfg.epochs - 1, {'loss': avg_loss})
