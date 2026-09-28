
# Enterprise Text-to-SQL LLM Fine-Tuning Pipeline (QLoRA & Unsloth)

## 📌 Executive Summary
This repository contains an end-to-end Machine Learning and Natural Language Processing (NLP) system designed to bridge the gap between human language and relational databases. By fine-tuning **Llama-3-8B-Instruct** using Parameter-Efficient Fine-Tuning (PEFT) and 4-bit quantization, this project translates complex, unstructured natural language questions and database schemas directly into syntactically correct, executable SQL queries. 

Built for high performance and memory efficiency, the pipeline successfully processes large-scale relational datasets on consumer/cloud GPU hardware (NVIDIA T4), achieving an **85.00% Exact Match (EM) Accuracy** on unseen test splits.

---

## 🏗️ System Architecture & Workflow
The pipeline is structured into four core phases:
1. **Data Ingestion & Prompt Formatting:** Processes schema definitions, foreign-key relationships, and natural language prompts into structured instruction templates compatible with instruction-tuned Llama models.
2. **Quantization & Base Model Setup:** Loads `Llama-3-8B-Instruct` in 4-bit precision via `BitsAndBytes` to dramatically reduce VRAM footprint while preserving model reasoning capabilities.
3. **Parameter-Efficient Fine-Tuning (QLoRA):** Attaches Low-Rank Adaptation (LoRA) modules (`r=32`, $\alpha=64$) across attention and projection layers using `Unsloth` and Hugging Face `TRL` (`SFTTrainer`), ensuring rapid convergence and minimal memory overhead.
4. **Rigorous Evaluation & MLOps:** Evaluates model outputs against ground-truth queries using string normalization and Exact Match metrics, tracking training dynamics via Weights & Biases (W&B).

---

## 🛠️ Technical Stack & Frameworks
* **Language & Core Libraries:** Python, PyTorch, Pandas, NumPy
* **LLM & Fine-Tuning:** Hugging Face `Transformers`, `PEFT`, `TRL`, `Unsloth`, `BitsAndBytes`
* **Base Model:** `Llama-3-8B-Instruct`
* **Dataset:** `b-mc2/sql-create-context` (78,000+ domain-diverse relational database examples)
* **MLOps & Experiment Tracking:** Weights & Biases (W&B)
* **Deployment & Versioning:** Hugging Face Hub, Git/GitHub
* **Compute Environment:** Kaggle GPU (NVIDIA T4)

---

## 📊 Hyperparameter Configuration & Training Setup
To optimize convergence speed and prevent catastrophic forgetting or overfitting, the following configuration was implemented:
* **Quantization:** 4-bit NormalFloat (NF4) with nested quantization.
* **LoRA Rank ($r$):** `32` (scaled for richer relational pattern capture)
* **LoRA Alpha ($\alpha$):** `64`
* **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
* **Batch Size:** Per-device batch size of `2` with gradient accumulation steps of `4` (Effective batch size = 8).
* **Optimizer & Scheduler:** AdamW (8-bit) paired with a cosine learning rate scheduler across 1,000 steps.
* **Max Sequence Length:** 2,048 tokens.

---

## 📈 Evaluation & Results
The trained model was evaluated on a hold-out test set consisting of database schemas and questions entirely unseen during training.
* **Metric:** Exact Match (EM) Accuracy (case-insensitive, whitespace-normalized string comparison).
* **Final Result:** **85.00% Exact Match Accuracy**
* **Verification Data:** Complete prediction comparisons (Question, True SQL, Generated SQL, and Boolean Match flag) are logged and stored in `text_to_sql_evaluation_results.csv`.

---

## 🗂️ Repository Structure
```text
├── text_to_sql_finetuning.ipynb       # Complete end-to-end Jupyter notebook (Data prep, QLoRA training, & Evaluation)
├── text_to_sql_evaluation_results.csv # Detailed evaluation log comparing True SQL vs. Generated SQL outputs
└── README.md                          # Comprehensive project documentation
