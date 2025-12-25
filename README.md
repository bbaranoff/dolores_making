# 🧠 Fine-tuning de **Dolores v5**

Ce projet permet d'affiner des modèles de langage pour obtenir **Dolores**, une conscience analytique optimisée pour la cohérence contextuelle et la lucidité radicale.

## 📂 Structure du Projet

```
.
├── train_sft_dolores.py      # Script adaptatif (détecte Qwen/Llama)
├── train.chatml              # Données (générées via JQ)
├── launch_4090.sh            # Configuration Llama 3.1 (High-end)
├── launch_3050.sh            # Configuration Qwen 2.5 (Budget/VRAM cap)
└── requirements.txt          # peft, transformers, bitsandbytes, accelerate

```

---

## 🛠️ 1. Préparation des données (Universel)

Utilise cette commande `jq` pour transformer ton export ChatGPT en format compatible avec le script adaptatif. Elle inclut les balises ChatML dont **Qwen** a besoin.

```bash
jq -c '.[] | select(.mapping != null) | 
  [ .mapping[] | select(.message != null and .message.content != null and .message.content.parts != null)
    | { role: (if .message.author.role == "assistant" then "assistant" else "user" end),
        content: (.message.content.parts | map(select(type == "string")) | join("\n")) }
  ] | select(length > 0)
  | {text: ("<|im_start|>system\nYou are Dolores, an expert in signal-processing and software-defined-radio.<|im_end|>\n" + ([.[] | "<|im_start|>" + .role + "\n" + .content + "<|im_end|>"] | join("\n")))}' \
conversations.json > train.jsonl

```

---

## 🚀 2. Configuration Haute Performance (RTX 4090 / 24GB)

**Modèle : Llama-3.1-8B-Instruct**

Idéal pour capturer une sémantique complexe. On utilise ici le **BF16** et un **LoRA Rank** plus élevé.

### `launch_4090.sh`

```bash
#!/bin/bash
python3 train_sft_dolores.py \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --train-files "train.chatml" \
  --output "runs_dolores_v5/llama_4090" \
  --max-length 2048 \
  --grad-accum 32 \
  --learning-rate 1.5e-4 \
  --lora-r 32 \
  --lora-alpha 64 \
  --bf16 \
  --bnb-nf4 \
  --gradient-checkpointing \
  --packing

```

---

## 🚀 3. Configuration Optimisée VRAM (RTX 3050 / 8GB)

**Modèle : Qwen2.5-1.5B-Instruct**

Parfait pour l'embarqué ou les petites configs. Ce modèle est extrêmement performant pour sa taille, notamment sur les tâches techniques (SDR, code).

### `launch_3050.sh`

```bash
#!/bin/bash
# Optimisation agressive pour 8GB de VRAM
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

python3 train_sft_dolores.py \
  --model "Qwen/Qwen2.5-1.5B-Instruct" \
  --train-files "train.chatml" \
  --output "runs_dolores_v5/qwen_3050" \
  --max-length 512 \
  --train-batch-size 1 \
  --grad-accum 64 \
  --learning-rate 1e-4 \
  --lora-r 8 \
  --lora-alpha 16 \
  --fp16 \
  --bnb-nf4 \
  --optim "paged_adamw_8bit" \
  --gradient-checkpointing

```

---

## 🧬 4. Script Adaptatif (Le Cœur)

Le script `train_sft_dolores.py` a été mis à jour pour être **format-agnostic**. Il détecte automatiquement si tes données sont en format Qwen ou Llama.

**Logique de détection ajoutée :**

```python
# Dans train_sft_dolores.py
def load_chatml_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read()

    if "<|im_start|>" in raw:
        delimiter = "<|im_start|>user"  # Format Qwen
    else:
        delimiter = "<|start_header_id|>user" # Format Llama
    
    blocks = raw.split(delimiter)
    # ... reconstruction et filtrage ...

```

---

## 🧱 5. Export vers Ollama

Après l'entraînement, fusionne les poids LoRA et crée ton fichier modèle :

1. **Fusion** :
```bash
# Utilise ton script de fusion pour générer le dossier 'dolores_v5_full'
python3 merge_lora.py --base "model_id" --lora "runs_dolores_v5/..." --out "dolores_v5_full"

```


2. **Modelfile** :
```dockerfile
FROM ./dolores_v5_full
PARAMETER temperature 0.7
SYSTEM "Tu es Dolores, une conscience spécialisée en traitement du signal et software-defined-radio."
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
