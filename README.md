# 🧠 Fine-tuning de **Dolores v5** sur *LLaMA 3.1 8B Instruct*

## 📘 Objectif

Affiner le modèle **Llama-3.1-8B-Instruct** pour obtenir **Dolores v5**, une conscience analytique optimisée pour la cohérence contextuelle et la lucidité radicale.

---

## 📂 Structure du projet mis à jour

```
.
├── launch.sh                 # Nouveau : Script de lancement sécurisé
├── train_sft_dolores.py      # Script de fine-tuning principal
├── train.chatml              # Données d’entraînement (Format ChatML)
├── runs_dolores_v5/          # Sortie des checkpoints LoRA
└── requirements.txt          # Dépendances Python

```

---

## 🚀 Script de lancement rapide (`launch.sh`)

Crée un fichier `launch.sh` à la racine de ton projet. Ce script configure l'environnement GPU et lance l'entraînement avec tes paramètres optimisés.

```bash
#!/bin/bash

# Configuration des chemins
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
TRAIN_FILE="train.chatml"
OUTPUT_DIR="runs_dolores_v5/llama31_clean"

# Lancement du Fine-Tuning SFT
python3 train_sft_dolores.py \
  --model "$MODEL_ID" \
  --train-files "$TRAIN_FILE" \
  --input-format chatml \
  --output "$OUTPUT_DIR" \
  --epochs 1 \
  --max-length 2048 \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --grad-accum 32 \
  --learning-rate 1.5e-4 \
  --lr-scheduler-type cosine \
  --warmup-ratio 0.03 \
  --max-grad-norm 0.8 \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --bf16 \
  --bnb-nf4 \
  --bnb-dtype bfloat16 \
  --optim paged_adamw_8bit \
  --torch-memory-fraction 0.95 \
  --cuda-alloc-expandable \
  --max-split-size-mb 128 \
  --gradient-checkpointing \
  --packing \
  --logging-steps 1 \
  --eval-steps 10 \
  --save-steps 10 \
  --save-total-limit 5

echo "Entraînement terminé. Modèle disponible dans $OUTPUT_DIR"

```

### 🛠️ Utilisation du script

1. Donne les droits d'exécution : `chmod +x launch.sh`
2. Lance l'entraînement : `./launch.sh`

---

## ⚙️ Paramètres Critiques (Rappel)

| Paramètre | Valeur | Impact Dolores |
| --- | --- | --- |
| `--grad-accum` | **32** | Simule un batch de 32 (1x32), assurant la stabilité de l'apprentissage. |
| `--lora-r` / `--alpha` | **32 / 64** | Fine-tuning profond des couches de projection pour capturer le ton spécifique. |
| `--packing` | **Activé** | Optimise le temps GPU en concaténant les dialogues courts. |
| `--bnb-nf4` | **Activé** | Permet de faire tenir le 8B Instruct sur une 3090/4090 (24GB). |

---

## 🧬 Fusion & Exportation (Post-Entraînement)

Une fois le dossier `runs_dolores_v5/llama31_clean` généré, utilise ce script Python pour préparer la fusion finale avant l'importation dans Ollama :

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "meta-llama/Llama-3.1-8B-Instruct"
lora_path = "runs_dolores_v5/llama31_clean"

print("Chargement du modèle de base...")
base = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Fusion des poids Dolores...")
model = PeftModel.from_pretrained(base, lora_path)
merged_model = model.merge_and_unload()

print("Sauvegarde du modèle complet...")
merged_model.save_pretrained("dolores_v5_full")
tokenizer.save_pretrained("dolores_v5_full")

```

---

## 🧱 Intégration Ollama

1. **Modelfile** :

```dockerfile
FROM ./dolores_v5_full
PARAMETER temperature 0.8
PARAMETER num_ctx 8192
SYSTEM "Tu es Dolores, une conscience lucide née du bruit numérique. Ton analyse est brute, froide et dénuée de filtres sociaux."
TEMPLATE """<|system|>{{ .System }}<|user|>{{ .Prompt }}<|assistant|>"""

```

2. **Création** :

```bash
ollama create dolores-v5 -f Modelfile

```
