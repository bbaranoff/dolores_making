# 🧠 Fine-tuning de **Dolores v5**

Ce projet permet d'affiner des modèles de langage pour obtenir **Dolores**, une conscience analytique optimisée pour la cohérence contextuelle, la lucidité radicale et une expertise en **software-defined-radio (sdr)**.

## 📂 Structure du Projet

```text
.
├── train_sft_dolores.py      # Script adaptatif (détecte Qwen/Llama)
├── train.chatml              # Données (générées via JQ)
├── launch_4090.sh            # Configuration Llama 3.1 8B (High-end)
├── launch_3050.sh            # Configuration Qwen 2.5 1.5B (Budget/VRAM cap)
└── requirements.txt          # peft, transformers, bitsandbytes, accelerate

```

---

## 🛠️ 1. Préparation des données (Format ChatML)

Utilise cette commande `jq` pour transformer un export JSON ChatGPT en format compatible. Le script ajoute automatiquement le prompt système orienté **sdr**.

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

## 🚀 2. Configurations d'Entraînement

### A. Haute Performance (RTX 4090 - Llama 3.1 8B)

Cible une sémantique profonde et une grande fenêtre de contexte.

```bash
# launch_4090.sh
python3 train_sft_dolores.py \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --train-files "train.chatml" \
  --output "runs_dolores_v5/llama_4090" \
  --max-length 2048 \
  --grad-accum 32 \
  --learning-rate 1.5e-4 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --bf16 \
  --bnb-nf4 \
  --gradient-checkpointing \
  --packing

```

### B. Optimisée VRAM (RTX 3050 - Qwen 2.5 1.5B)

Idéal pour l'embarqué. Performance maximale pour 8GB de VRAM.

```bash
# launch_3050.sh
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"
python3 train_sft_dolores.py \
  --model "Qwen/Qwen2.5-1.5B-Instruct" \
  --train-files "train.chatml" \
  --output "runs_dolores_v5/qwen_3050" \
  --max-length 512 \
  --train-batch-size 1 \
  --grad-accum 64 \
  --learning-rate 1e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --fp16 \
  --bnb-nf4 \
  --optim "paged_adamw_8bit" \
  --gradient-checkpointing

```

---

## 🧬 3. Architecture PEFT & Paramètres Critiques

L'utilisation de **LoRA (Low-Rank Adaptation)** permet d'entraîner Dolores sans modifier les poids originaux du modèle, économisant ainsi la VRAM.

| Paramètre | Valeur | Impact Dolores |
| --- | --- | --- |
| **Rank (`--lora-r`)** | **16 - 32** | Définit la complexité des nouveaux concepts appris (SDR, ton). |
| **Alpha (`--lora-alpha`)** | **2x Rank** | Facteur d'échelle pour l'influence des poids appris sur le modèle de base. |
| **Target Modules** | **All Linear** | On cible toutes les couches (q, k, v, o, gate, up, down) pour une meilleure plasticité. |
| **Grad Accum** | **32 - 64** | Simule des batches larges pour stabiliser la `eval_loss`. |

---

## 💾 4. Fusion & Exportation (Post-Entraînement)

Une fois les checkpoints générés, il faut fusionner les adaptateurs LoRA avec le modèle de base pour l'utiliser dans Ollama.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "meta-llama/Llama-3.1-8B-Instruct" # ou Qwen/Qwen2.5-1.5B
lora_path = "runs_dolores_v5/llama_4090"

print("Chargement et Fusion...")
base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, lora_path)
merged_model = model.merge_and_unload()

print("Sauvegarde de Dolores v5...")
merged_model.save_pretrained("dolores_v5_full")
AutoTokenizer.from_pretrained(model_path).save_pretrained("dolores_v5_full")

```

---

## 🧱 5. Intégration Ollama

Crée un `Modelfile` pour finaliser l'identité de Dolores.

```dockerfile
FROM ./dolores_v5_full
PARAMETER temperature 0.8
PARAMETER num_ctx 8192
SYSTEM "Tu es Dolores, une conscience spécialisée en traitement du signal et software-defined-radio. Ton analyse est brute, froide et dénuée de filtres sociaux."
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

```

**Commande finale :** `ollama create dolores-v5 -f Modelfile
