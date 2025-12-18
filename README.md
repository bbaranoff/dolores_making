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

Pour que **Dolores v5** s'éveille sans crash système (surtout avec ROCm et ta VRAM déjà bien entamée), l'ordre des opérations est crucial. Tu ne peux pas lancer le moteur si les vannes de sécurité sont fermées.

Voici les étapes de préparation à insérer avant ta commande Python :

---

### Étape 1 : Nettoyage de la zone (VRAM Flush)

```bash
# Tue les processus utilisant le GPU (attention, ça coupera Ollama si actif)
fuser -v /dev/dri/renderD* | xargs -r kill -9

```

### Étape 2 : Configuration de l'environnement ROCm

Indispensable pour que ta carte AMD accepte de traiter les calculs de Llama 3.1.

```bash
# Force la compatibilité logicielle (indispensable pour les cartes grand public)
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # Si RX 6000
# export HSA_OVERRIDE_GFX_VERSION=11.0.0  # Si RX 7000

# Empêche les erreurs de segmentation sur certains systèmes ROCm
export NCCL_P2P_DISABLE=1

```

### Étape 3 : Authentification Hugging Face

Le modèle Llama-3.1-8B-Instruct est protégé. Sans cette étape, le script plante au chargement du tokenizer.

```bash
# Installation du CLI (si pas déjà fait)
curl -LsSf https://hf.co/cli/install.sh | bash

(venv) ubuntu@swift:~$ hf auth login

    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

    To log in, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Enter your token (input will not be visible): 
Add token as git credential? [y/N]: y
Token is valid (permission: fineGrained).
The token `blah` has been saved to /home/ubuntu/.cache/huggingface/stored_tokens
Cannot authenticate through git-credential as no helper is defined on your machine.
You might have to re-authenticate when pushing to the Hugging Face Hub.
Run the following command in your terminal in case you want to set the 'store' credential helper as default.

git config --global credential.helper store

Read https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage for more details.
Token has not been saved to git credential helper.
Your token has been saved to /home/ubuntu/.cache/huggingface/token
Login successful.
The current active token is: `blah`


```

### Étape 4 : Vérification du fichier de données

Assure-toi que ton `train.chatml` existe et qu'il est propre (format UTF-8 sans caractères parasites).

Pour le créer à partir d'un export de conversation avec ChatGPT (persona)

```
jq -r '
  .[]
  | select(type=="object" and has("mapping") and (.mapping|type)=="object")
  | .mapping
  | to_entries[]
  | select(
      (.value|type)=="object"
      and (.value.message|type)=="object"
      and (.value.message.author|type)=="object"
      and (.value.message.content|type)=="object"
      and (.value.message.content.parts|type)=="array"
    )
  | .value.message
  | "<|start_header_id|>"+(.author.role // "unknown")+"<|end_header_id|>\n"+
    ((.content.parts | map(tostring) | join("\n")) // "") +
    "<|eot_id|>"
' conversations.json | sed '1s/^/<|begin_of_text|>/' > train.chatml
```

Le script de lancement complet (`launch.sh`)

Voici comment tout assembler pour que ce soit propre :

```bash
#!/bin/bash

# 1. Variables de base
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
TRAIN_FILE="train.chatml"
OUTPUT_DIR="runs_dolores_v5/llama31_clean"

# 2. Setup ROCm/AMD
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:128"

# 3. Lancement de l'entraînement
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
  --torch-memory-fraction 0.90 \
  --cuda-alloc-expandable \
  --max-split-size-mb 128 \
  --gradient-checkpointing \
  --packing \
  --logging-steps 1 \
  --eval-steps 10 \
  --save-steps 10 \
  --save-total-limit 5
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
