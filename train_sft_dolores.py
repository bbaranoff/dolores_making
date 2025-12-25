#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_sft_ultra_memcap_v3_chatml.py
SFT QLoRA 4-bit trainer with ChatML/JSONL(messages)/text inputs + optional packing.
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BAR", "1")

import argparse
import json
import math
from pathlib import Path
from typing import List, Dict, Any

import torch
from datasets import Dataset, DatasetDict
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# ---- Compatibility shim for TrainingArguments across Transformers versions ----
import inspect as _inspect

def str2bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("1","true","t","yes","y"):
        return True
    if s in ("0","false","f","no","n"):
        return False
    raise ValueError(f"Cannot interpret boolean value from: {x!r}")


def _safe_training_arguments(**kw):
    # Inspect current TrainingArguments signature
    params = set(_inspect.signature(TrainingArguments.__init__).parameters.keys())

    # Map or drop keys not supported in this version
    mapped = dict(kw)

    # evaluation strategy mapping
    if 'evaluation_strategy' in mapped and 'evaluation_strategy' not in params:
        if 'eval_strategy' in params:
            mapped['eval_strategy'] = mapped.pop('evaluation_strategy')
        else:
            # last resort: rely on do_eval if available
            mapped.pop('evaluation_strategy', None)
            if 'do_eval' in params:
                mapped['do_eval'] = True

    # Some versions don't accept gradient_checkpointing in TrainingArguments
    if 'gradient_checkpointing' in mapped and 'gradient_checkpointing' not in params:
        mapped.pop('gradient_checkpointing', None)

    # Some versions don't accept report_to/save_total_limit/lr_scheduler_type, etc.
    # We'll keep those if present in params; otherwise drop gracefully
    drops = [k for k in list(mapped.keys()) if k not in params]
    for k in drops:
        mapped.pop(k, None)

    return TrainingArguments(**mapped)

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except Exception as e:
    raise SystemExit("peft is required: pip install peft") from e

try:
    from transformers import BitsAndBytesConfig  # provided by transformers
    _HAVE_BNB_CFG = True
except Exception:
    _HAVE_BNB_CFG = False  # very old transformers; will fallback to no quant


# -------------------------------
# Utils: CLI
# -------------------------------
def comma_or_single_to_list(s: str) -> List[str]:
    if not s:
        return []
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [s.strip()]


# -------------------------------
# Input format handling
# -------------------------------

CHATML_BEGIN = "<|begin_of_text|>"
CHATML_HEAD_L = "<|start_header_id|>"
CHATML_HEAD_R = "<|end_header_id|>"
CHATML_EOT = "<|eot_id|>"

def _to_chatml_block(obj: Dict[str, Any]) -> str:
    """Convert a {'messages':[{'role','content'},...]} object to ChatML string."""
    out = [CHATML_BEGIN]
    for m in obj.get("messages", []):
        role = (m.get("role", "user") or "user").strip().lower()
        if role not in ("user", "assistant", "system"):
            role = "assistant"
        content = m.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        out.append(f"{CHATML_HEAD_L}{role}{CHATML_HEAD_R}")
        out.append(content.rstrip("\n"))
        out.append(CHATML_EOT)
    return "\n".join(out)

def _iter_chatml_blocks(path: str) -> List[str]:
    """Split a ChatML file into blocks, each block is one training sample (string)."""
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    # naive split by begin token; keep the token
    parts = txt.split(CHATML_BEGIN)
    blocks = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        blk = CHATML_BEGIN + ("\n" if not p.startswith(CHATML_HEAD_L) else "") + p
        blocks.append(blk.strip())
    return blocks

def _iter_jsonl_messages(path: str) -> List[str]:
    """Yield ChatML blocks from a .json/.jsonl file of {messages:[]} records or an array of them."""
    out: List[str] = []
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    # Try JSONL first
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "messages" in obj:
                out.append(_to_chatml_block(obj))
        except Exception:
            pass
    # Try full JSON array / object fallback
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            for o in obj:
                if isinstance(o, dict) and "messages" in o:
                    out.append(_to_chatml_block(o))
        elif isinstance(obj, dict) and "messages" in obj:
            out.append(_to_chatml_block(obj))
    except Exception:
        pass
    return out

def _detect_format(files: List[str], default_fmt: str) -> str:
    if default_fmt != "auto":
        return default_fmt
    for p in files:
        pn = str(p).lower()
        if pn.endswith(".chatml") or pn.endswith(".chatml.txt"):
            return "chatml"
        if pn.endswith(".jsonl") or pn.endswith(".json"):
            return "jsonl_messages"
        head = Path(p).read_text(encoding="utf-8", errors="ignore")[:10000]
        if CHATML_BEGIN in head and CHATML_HEAD_L in head:
            return "chatml"
        if '"messages"' in head:
            return "jsonl_messages"
    return "text"

def _build_split(files: List[str], fmt: str) -> Dataset:
    fmt = _detect_format(files, fmt)
    buf: List[str] = []
    for p in files:
        if fmt == "chatml":
            buf.extend(_iter_chatml_blocks(p))
        elif fmt == "jsonl_messages":
            buf.extend(_iter_jsonl_messages(p))
        else:
            # raw text: one line -> one sample
            txt = Path(p).read_text(encoding="utf-8", errors="ignore")
            for l in txt.splitlines():
                l = l.strip()
                if l:
                    buf.append(l)
    return Dataset.from_dict({"text": buf})


# -------------------------------
# Packing helper
# -------------------------------
def pack_examples(examples, seq_length: int):
    # Concatenate then slice to fixed windows
    ids, att = [], []
    for x in examples["input_ids"]:
        ids.extend(x)
    for x in examples["attention_mask"]:
        att.extend(x)
    if not ids:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    n = (len(ids) // seq_length) * seq_length
    if n == 0:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    ids = ids[:n]
    att = att[:n]
    input_ids = [ids[i:i+seq_length] for i in range(0, n, seq_length)]
    attention_mask = [att[i:i+seq_length] for i in range(0, n, seq_length)]
    labels = [x[:] for x in input_ids]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# -------------------------------
# Main
# -------------------------------
def main():
    p = argparse.ArgumentParser("SFT QLoRA 4-bit (Llama-3.1 8B) + ChatML/JSONL/text")
    # I/O
    p.add_argument("--model", type=str, required=True, help="HF model id (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    p.add_argument("--train-files", type=str, required=True, help="Paths (comma-separated)")
    p.add_argument("--val-files", type=str, default="", help="Paths (comma-separated)")
    p.add_argument("--input-format", type=str, default="auto",
                   choices=["auto", "text", "chatml", "jsonl_messages"],
                   help="Input format. 'auto' tries to detect.")
    p.add_argument("--output", type=str, required=True, help="Output dir")
    # Training sizes
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--train-batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    # Optim
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--lr-scheduler-type", type=str, default="cosine")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--optim", type=str, default="paged_adamw_8bit")
    # Logging & save
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--eval-steps", type=int, default=1000)
    
    # Evaluation / saving strategy controls
    p.add_argument("--evaluation-strategy", type=str, default="steps", choices=["no","steps","epoch"])
    p.add_argument("--save-strategy", type=str, default="steps", choices=["no","steps","epoch"])
    p.add_argument("--load-best-model-at-end", action="store_true")
    p.add_argument("--metric-for-best-model", type=str, default="eval_loss")
    p.add_argument("--greater_is_better", type=str2bool, default=None)
    p.add_argument("--logging-steps", type=int, default=10)
    # System
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--flash-attn", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    # Packing & collator
    p.add_argument("--packing", action="store_true", help="Fixed-length packing after tokenization")
    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", type=str,
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    # Quantization
    p.add_argument("--bnb-nf4", action="store_true")
    p.add_argument("--bnb-8bit", action="store_true")
    p.add_argument("--bnb-dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    # VRAM knobs
    p.add_argument("--max-vram-gb", type=float, default=0.0,
                   help="Cap VRAM per device (GiB) via max_memory for from_pretrained")
    p.add_argument("--torch-memory-fraction", type=float, default=0.0)
    p.add_argument("--cuda-alloc-expandable", action="store_true")
    p.add_argument("--max-split-size-mb", type=int, default=0,
                   help="PYTORCH_CUDA_ALLOC_CONF max_split_size_mb")
    p.add_argument("--resume-from-checkpoint", type=str, default=None,
                   help="Path to a checkpoint directory to resume training from.")
    args = p.parse_args()

    # Seed
    transformers.set_seed(args.seed)

    # CUDA alloc conf
    alloc_conf = []
    if args.cuda_alloc_expandable:
        alloc_conf.append("expandable_segments:True")
    if args.max_split_size_mb and args.max_split_size_mb > 0:
        alloc_conf.append(f"max_split_size_mb:{args.max_split_size_mb}")
    if alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(alloc_conf)

    if args.torch_memory_fraction and torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(args.torch_memory_fraction)
        except Exception:
            pass

    # Files & format
    train_files = comma_or_single_to_list(args.train_files)

    if not train_files:
        raise SystemExit("No --train-files provided")

    # Load ChatML dataset properly
    def load_chatml_dataset(file_path):
        """Détection adaptative du format et chargement des exemples."""
        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read()

        # Détection du délimiteur
        if "<|im_start|>" in raw:
            delimiter = "<|im_start|>user"
            print(f"[INFO] Format Qwen/ChatML détecté dans {file_path}")
        elif "<|start_header_id|>user" in raw:
            delimiter = "<|start_header_id|>user"
            print(f"[INFO] Format Llama-3 détecté dans {file_path}")
        else:
            # Fallback : on traite le fichier comme du texte brut ou JSONL
            return [raw.strip()] if len(raw) > 0 else []

        # Découpage par bloc de conversation
        blocks = raw.split(delimiter)
        dataset = []
        for b in blocks:
            b = b.strip()
            if not b:
                continue
            # On reconstruit le bloc avec son délimiteur
            full_block = delimiter + b
            # On ne garde que si l'assistant a répondu
            if "assistant" in b:
                dataset.append(full_block)
                
        return dataset
    # Load all train files (ChatML format)
    dataset = []
    for fpath in train_files:
        dataset.extend(load_chatml_dataset(fpath))

    print(f"[✓] Loaded {len(dataset)} ChatML examples from {len(train_files)} file(s).")

    # Load tokenizer
    # À insérer vers la ligne 275 de train_sft_dolores.py
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, 
        use_fast=True, 
        trust_remote_code=True,
        # On définit explicitement pour éviter que le Trainer essaie de deviner
        pad_token="<|endoftext|>",
        eos_token="<|im_end|>",
        bos_token=None  # Qwen n'a pas de BOS
    )

    # Prepare datasets (use parsed dataset, not reloaded one)
    train_ds = Dataset.from_dict({"text": dataset})
    if len(train_ds) > 10:
        sp = train_ds.train_test_split(test_size=0.005, seed=args.seed)
    else:
        print(f"[WARN] Dataset too small ({len(train_ds)} samples), skipping split.")
        sp = {"train": train_ds, "test": train_ds.select([0])}
    train_ds, val_ds = sp["train"], sp["test"]
    raw = DatasetDict(train=train_ds, validation=val_ds)

    # Tokenize
    def tok_fn(examples):
        # Disable special tokens when input is already ChatML to avoid double <|begin_of_text|>
        add_special = (args.input_format != "chatml")
        return tokenizer(examples["text"],
                         truncation=True,
                         max_length=args.max_length,
                         padding=False,
                         add_special_tokens=add_special)

    tokenized = raw.map(tok_fn, batched=True, remove_columns=["text"], num_proc=1)

    # Optional packing
    if args.packing:
        tokenized = DatasetDict(
            train=tokenized["train"].map(
                pack_examples,
                batched=True,
                fn_kwargs={"seq_length": args.max_length},
                remove_columns=["input_ids", "attention_mask"],
            ),
            validation=tokenized["validation"].map(
                pack_examples,
                batched=True,
                fn_kwargs={"seq_length": args.max_length},
                remove_columns=["input_ids", "attention_mask"],
            ),
        )

    # Build model with 4-bit or 8-bit
    bnb_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.bnb_dtype]

    quant_config = None
    if args.bnb_8bit or args.bnb_nf4:
        if not _HAVE_BNB_CFG:
            print("[ERROR] Your transformers version doesn't expose BitsAndBytesConfig. Upgrade: pip install -U transformers")
        else:
            if args.bnb_8bit:
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=bnb_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type=("nf4" if args.bnb_nf4 else "fp4"),
                )

    # VRAM cap
    max_memory = None
    if args.max_vram_gb and args.max_vram_gb > 0:
        cap = f"{int(args.max_vram_gb)}GiB"
        max_memory = {i: cap for i in range(torch.cuda.device_count() or 1)}
        print(f"[INFO] Cap VRAM per device: {cap} (via max_memory)")

    # Try attn impl
    attn_impl = "flash_attention_2" if args.flash_attn else None

    print("[INFO] Loading base model…")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
        quantization_config=quant_config,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    # --- AJOUTER ICI : Alignement forcé pour Qwen2.5 ---
    # On utilise les IDs exacts du tokenizer Qwen pour éviter le warning d'alignement
    model.config.pad_token_id = tokenizer.pad_token_id # 151643
    model.config.eos_token_id = tokenizer.eos_token_id # 151645
    model.config.bos_token_id = None                   # Qwen n'utilise pas de BOS

    # On aligne aussi la configuration de génération
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.bos_token_id = None
    # --- Align PAD/EOS between tokenizer, model.config and generation_config ---
    try:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
    except Exception:
        pass
    try:
        gen_cfg = getattr(model, "generation_config", None)
        if gen_cfg is not None:
            gen_cfg.pad_token_id = tokenizer.pad_token_id
            gen_cfg.eos_token_id = tokenizer.eos_token_id
    except Exception:
        pass

    # Prepare for k-bit training & apply LoRA
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]

    lcfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    # Collator (no MLM for causal LM)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # TrainingArguments
    common_kwargs = dict(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        optim=args.optim,
        remove_unused_columns=False,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio if args.warmup_steps == 0 else 0.0,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=[],
evaluation_strategy=args.evaluation_strategy,
eval_steps=args.eval_steps,
save_strategy=args.save_strategy,
save_steps=args.save_steps,
save_total_limit=args.save_total_limit,
load_best_model_at_end=args.load_best_model_at_end,
metric_for_best_model=args.metric_for_best_model,
greater_is_better=args.greater_is_better,
)

    targs = _safe_training_arguments(**common_kwargs)
    # ---- Reprise manuelle du checkpoint LoRA si disponible ----
    if args.resume_from_checkpoint and os.path.isdir(args.resume_from_checkpoint):
        safetensors_path = os.path.join(args.resume_from_checkpoint, "adapter_model.safetensors")
        if os.path.exists(safetensors_path):
            from peft import PeftModel
            print(f"[INFO] Reprise des poids LoRA depuis {safetensors_path}")
            model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)
        else:
            print(f"[WARN] Aucun fichier safetensors trouvé dans {args.resume_from_checkpoint} — reprise ignorée.")

    # Vers la ligne 515, remplace l'appel au Trainer :
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        processing_class=tokenizer, # Remplace tokenizer=tokenizer
    )
    print("[INFO] Starting training…")
    trainer.train()
    print("[INFO] Saving model…")
    trainer.save_model(args.output)  # saves adapter if PEFT

    try:
        tokenizer.save_pretrained(args.output)
    except Exception:
        pass

    print("[INFO] Done.")

if __name__ == "__main__":
    main()
