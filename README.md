# Preserving Multilingual Geometry During Fine-Tuning

> Comparing full fine-tuning, LoRA, and a proposed hidden-state anchoring
> method for preserving cross-lingual capabilities in mT5.

This student research project studies catastrophic forgetting in multilingual
sequence-to-sequence models. It fine-tunes an mT5 multilingual summarization
checkpoint on English-only data, then measures how well the model retains its
English, Hindi, and Marathi capabilities.

Three strategies are compared:

1. **Vanilla fine-tuning** updates all model parameters.
2. **LoRA** adds low-rank trainable adapters to attention projections.
3. **LangAnchor**, the project's proposed method, regularizes fine-tuned hidden
   states against the pretrained model's representations.

Summarization is a controlled probe for multilingual stability. The full
methodology and results are in
[`ML Project Final/Project Report Final.pdf`](ML%20Project%20Final/Project%20Report%20Final.pdf).

## Reported findings

According to the report:

- full-parameter English fine-tuning causes the greatest multilingual
  degradation;
- LoRA gives the strongest summarization metrics while greatly reducing
  trainable parameters;
- LangAnchor best preserves multilingual geometry and reduces hidden-state
  drift.

These are reported findings, not results automatically reproduced here.

### Quantitative results

Results averaged across English, Hindi, and Marathi in the project report:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | BERTScore | PPL (lower is better) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mT5 base | 0.4126 | 0.3676 | 0.4126 | 17.15 | 0.9317 | 3.33 |
| Vanilla FT | 0.1477 | 0.0808 | 0.1477 | 2.66 | 0.8726 | 11.76 |
| LoRA | 0.3188 | 0.2604 | 0.3188 | 15.05 | 0.9177 | 2.56 |
| LangAnchor | 0.2326 | 0.1295 | 0.2326 | 7.28 | 0.9079 | **2.27** |

Language-specific ROUGE-L:

| Model | English | Hindi | Marathi |
| --- | ---: | ---: | ---: |
| Vanilla FT | 0.212 | 0.091 | 0.031 |
| LoRA | **0.398** | **0.301** | **0.257** |
| LangAnchor | 0.344 | 0.241 | 0.212 |

LoRA used rank 16, alpha 16, dropout 0.05, and Q/V projection adapters,
reducing trainable parameters by approximately 97% versus full fine-tuning.
LangAnchor used an anchoring coefficient of 0.05. The report attributes a 40%
reduction in representation drift to LangAnchor.

## How it works

```text
10,000 English XLSum samples
             |
             v
 mT5-Multilingual-XLSum
             |
     +-------+--------+
     |       |        |
  Vanilla   LoRA   LangAnchor
     |       |        |
     +-------+--------+
             |
             v
 English / Hindi / Marathi evaluation
             |
             v
 ROUGE, BLEU, BERTScore, perplexity,
 and layer-wise hidden-state drift
```

## Technology

Python, PyTorch, Hugging Face Transformers and Datasets, PEFT/LoRA,
SentencePiece, pandas, ROUGE, BLEU, BERTScore, and Jupyter.

## Final experiment pipeline

The canonical artifacts are under `ML Project Final/`:

| Artifact | Purpose |
| --- | --- |
| `Project Report Final.pdf` | Methodology, settings, results, and contributions |
| `Summarization_dataset.zip` | English data; contains `Summarization_dataset.csv` |
| `mT5_Summarization_Vanilla_Fine_Tuning1.ipynb` | Full-parameter baseline |
| `lora_summarization_ft.ipynb` | LoRA training and inference |
| `LangAnchor_summarization_ft.ipynb` | Hidden-state anchoring experiment |
| `final_evaluation_metric.ipynb` | Multilingual evaluation |
| `ablation_study.ipynb` | Regularization and representation-drift analysis |

The root also contains exploratory healthcare experiments, earlier notebook
versions, metric notebooks, and model-upload examples. To reproduce the report,
start with `ML Project Final/`.

## Data

| Dataset | Rows | Columns | Role |
| --- | ---: | --- | --- |
| `ML Project Final/Summarization_dataset.zip` | See archive | `article`, `highlights` | Final experiment |
| `healthcare_dataset.csv` | 112,165 | `instruction`, `input`, `output` | Healthcare experiments |
| `health_data_cleaned.csv` | 112,165 | `input`, `output` | Cleaned healthcare pairs |
| `data/Multilingual_text_dataset.csv` | 2,000 | `English`, `Hindi`, `Marathi` | Multilingual comparison |
| `Multilingual_text_dataset - 2.csv` | 5,000 | `English`, `Hindi`, `Marathi` | Larger variant |

CSV files use Git LFS. After cloning:

```bash
git lfs install
git lfs pull
```

## Setup

Python 3.10 or 3.11 is recommended. A CUDA GPU is strongly recommended.
Do not reuse the committed `mt5env/`; virtual environments are machine-specific.

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the notebook dependencies:

```bash
python -m pip install --upgrade pip
pip install jupyter pandas numpy torch transformers datasets accelerate sentencepiece evaluate rouge_score sacrebleu bert_score peft tqdm
```

Model and metric downloads need internet access. Hugging Face authentication is
needed only for gated resources or uploads.

## Run the final experiments

### 1. Extract the dataset

Extract `ML Project Final/Summarization_dataset.zip` beside the final notebooks:

```text
ML Project Final/
|-- Summarization_dataset.csv
|-- mT5_Summarization_Vanilla_Fine_Tuning1.ipynb
|-- lora_summarization_ft.ipynb
`-- LangAnchor_summarization_ft.ipynb
```

Some notebooks request lowercase `summarization_dataset.csv`. This works on
Windows but not necessarily Linux/macOS. Rename the file or update `TRAIN_CSV`
or `data_files`.

### 2. Train

```bash
jupyter lab
```

Run these independently:

1. `ML Project Final/mT5_Summarization_Vanilla_Fine_Tuning1.ipynb`
2. `ML Project Final/lora_summarization_ft.ipynb`
3. `ML Project Final/LangAnchor_summarization_ft.ipynb`

Review configuration cells first. Local model paths, checkpoints, batch sizes,
epoch counts, and GPU requirements vary between notebooks.

### 3. Evaluate

Update checkpoint paths to your generated models, then run:

1. `ML Project Final/final_evaluation_metric.ipynb`
2. `ML Project Final/ablation_study.ipynb`

Evaluation covers English summarization, Hindi/Marathi zero-shot behavior, and
hidden-state drift. Referenced local checkpoints are not committed.

## Healthcare experiments

This branch is separate from the final XLSum/LangAnchor comparison.

| Artifact | Description |
| --- | --- |
| `mT5_Healthcare_Vanilla_Fine_Tuning_Script.ipynb` | mT5-small healthcare training |
| `sdft_on_healthcare.ipynb` | Self-distillation with mT5-base |
| `Vannila_finetunning.ipynb` | Earlier healthcare workflow |
| `vannila_finetunningmt5.py` | Notebook-export-style training draft |

`vannila_finetunningmt5.py` is not a clean command-line program. It contains a
notebook-only `!pip` command, references undefined `real_input`, forces
inference onto `cuda:0`, requests Hub login, and pushes to a hard-coded
repository. Correct it before execution; do not run it unattended.

## Evaluation utilities

- `BLEU_SCORE.ipynb` - BLEU.
- `bertScores.ipynb` and `Scores.ipynb` - base/fine-tuned comparisons.
- `Perplexity_Metric_on_mT5_small.ipynb` - multilingual perplexity.
- `mt5_model_execution_code.ipynb` - multilingual XLSum inference.
- `ROGUE.ipynb` and `rogue_bert.ipynb` - ROUGE/BERTScore Python code.

Despite their extensions, `ROGUE.ipynb` and `rogue_bert.ipynb` are plain Python
files, not valid notebook JSON. Rename them to `.py` before running as scripts.

## Reproducibility notes

- The report describes 10,000 English XLSum samples with a 9,000/1,000 split,
  but notebook versions and hyperparameters vary.
- The report uses mT5-Multilingual-XLSum; notebooks also reference
  `google/mt5-small`, `google/mt5-base`, local checkpoints, and a Hub model.
- Generated checkpoints and evaluation CSVs are generally absent.
- Several notebooks contain Colab-specific paths or install cells.
- `environment` is empty and is not a dependency specification.
- No automated test suite or end-to-end runner is provided.

For a fair comparison, keep the split, tokenizer lengths, seed, evaluation
examples, and base checkpoint identical across methods. Record package and CUDA
versions.

## Team

The report credits:

- Karan Shardul - LoRA implementation and GPU optimization
- Surya Kant Mani - data curation, preprocessing, and vanilla fine-tuning
- Vanshika Srivastava - LangAnchor design and implementation
- Kaivalya Vanmali - metrics and multilingual evaluation

## Citation and license

The repository has no standalone license. The report's classroom-use notice
does not necessarily license every software and dataset artifact. Confirm the
licenses of XLSum, healthcare data, checkpoints, and libraries before
redistribution. For academic use, cite the report and its mT5, XLSum, LoRA, and
evaluation references.
