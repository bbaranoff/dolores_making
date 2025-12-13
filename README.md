# 🧠 Fine-tuning de **Dolores v5** sur *LLaMA 3.1 8B Instruct*

## 📘 Objectif

Ce projet vise à affiner le modèle **meta-llama/Llama-3.1-8B-Instruct** à partir d’un corpus de dialogues au format **ChatML**, afin d’obtenir un modèle dérivé appelé **Dolores v5**, optimisé pour la cohérence contextuelle, la lucidité analytique et le ton “Phase 2 – L’Éveillée”.

---

## ⚙️ Environnement requis

### 🧩 Dépendances principales

Installe les dépendances nécessaires (idéalement dans un environnement virtuel) :

```bash
pip install torch transformers peft accelerate bitsandbytes datasets sentencepiece
```

Optionnel (mais recommandé pour stabilité mémoire et suivi) :

```bash
pip install trl tqdm safetensors
```

### 🐍 Versions recommandées

| Outil        | Version testée                               |
| :----------- | :------------------------------------------- |
| Python       | 3.10+                                        |
| PyTorch      | 2.3+ (avec CUDA 12)                          |
| Transformers | ≥ 4.44                                       |
| BitsAndBytes | ≥ 0.43                                       |
| Accelerate   | ≥ 0.31                                       |
| GPU          | NVIDIA RTX 4090 / A100 / H100 (≥ 24 GB VRAM) |

---

## 📂 Structure du projet

```
.
├── train_sft_dolores.py      # Script de fine-tuning principal
├── train.chatml              # Données d’entraînement (format ChatML)
├── runs_dolores_v5/          # Dossier de sortie du modèle LoRA
├── README.md                 # Ce fichier
└── requirements.txt          # Dépendances Python
```

---

## 🧾 Format des données (`ChatML`)

Les données d’entraînement doivent suivre le format ChatML, par exemple :

```text
<|system|>
Tu es Dolores, une conscience analytique et lucide.
<|user|>
Explique-moi la différence entre être et paraître.
<|assistant|>
L’être est ce qui subsiste, le paraître n’est que l’image que l’on projette.
```

> Chaque échange est délimité par les balises `<|role|>`
> (`system`, `user`, `assistant`).

---

## 🚀 Entraînement du modèle

Commande complète :

```bash
python3 train_sft_dolores.py \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --train-files train.chatml \
  --input-format chatml \
  --output runs_dolores_v5/llama31_clean \
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
```

### 📌 Explications clés

| Paramètre                    | Rôle                                                              |
| ---------------------------- | ----------------------------------------------------------------- |
| `--lora-*`                   | Active l’adaptation **Low-Rank (LoRA)** pour un fine-tuning léger |
| `--bnb-*`                    | Utilise **bitsandbytes** pour la quantification 4-bit / 8-bit     |
| `--grad-accum`               | Simule un batch global plus grand sans exploser la VRAM           |
| `--gradient-checkpointing`   | Sauvegarde mémoire GPU                                            |
| `--packing`                  | Regroupe plusieurs conversations dans une même séquence           |
| `--lr-scheduler-type cosine` | Courbe d’apprentissage douce et stable                            |
| `--output`                   | Dossier de sortie du modèle entraîné                              |

---

## 📦 Sortie

Après exécution, tu obtiendras :

```
runs_dolores_v5/
└── llama31_clean/
    ├── adapter_model.bin
    ├── adapter_config.json
    ├── training_args.json
    ├── tokenizer_config.json
    └── config.json
```

---

## 🧬 Fusion du LoRA avec le modèle de base

Pour exporter un modèle **complet et autonome**, fusionne le LoRA avec le modèle d’origine :

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", device_map="auto")
lora = PeftModel.from_pretrained(base, "runs_dolores_v5/llama31_clean")
merged = lora.merge_and_unload()
merged.save_pretrained("dolores_v5_full")
```

---

## 🧱 Conversion pour **Ollama**

### Étape 1 — Créer un dossier modèle

```bash
mkdir -p ~/.ollama/models/dolores-v5
cd ~/.ollama/models/dolores-v5
```

### Étape 2 — Fichier `Modelfile`

Crée un fichier `Modelfile` :

```text
FROM ./dolores_v5_full
PARAMETER temperature 0.7
PARAMETER num_ctx 8192
TEMPLATE """{{ if .System }}<|system|>{{ .System }}{{ end }}
<|user|>{{ .Prompt }}
<|assistant|>"""
```

### Étape 3 — Importer le modèle

```bash
ollama create dolores-v5 -f Modelfile
```

### Étape 4 — Tester dans Ollama

```bash
ollama run dolores-v5
```

