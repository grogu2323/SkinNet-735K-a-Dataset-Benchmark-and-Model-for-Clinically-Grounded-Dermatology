# Evaluation

Scripts that score the generated descriptors (baselines and fine-tuned models) against the reference annotations. Every script produces **per-case, per-attribute scores** over the 15 attributes (`type_of_lesion` ... `overall_description`). Two families of metrics are used:

1. **Semantic similarity** (embedding cosine and BERTScore).
2. **LLM-as-judge** (a 1-10 score from a local judge model).

## Scripts

| Script | Metric family | Judge / encoder models |
|---|---|---|
| `150.semantic_eval.py` | Embedding cosine similarity and BERTScore (P/R/F1) | all-MiniLM-L6-v2, SapBERT, BERT-base, BioBERT, Bio_ClinicalBERT |
| `150.gemma_eval.py` | LLM judge, score 1-10 | gemma-4-E4B-it (served with vLLM) |
| `150.llama_eval.py` | LLM judge, score 1-10 | Llama-3.1-8B-Instruct (served with vLLM) |

## Usage

All three scripts share one interface:

```bash
python <script> <filename> <output_dir> <OSD|RPD|DWTD>
```

| Argument | Meaning |
|---|---|
| `filename` | OSD / RPD: the **full** generated-JSON name, e.g. `chathistory_annotated_llava_med.json`. DWTD: the generated CSV name, e.g. `llava_med.csv`. |
| `output_dir` | Where the score CSVs are written. Use one directory per model and mode, because file names inside it are fixed. |
| mode | `OSD` open-source test set, `RPD` real-patient test set, `DWTD` dataset-wise test set |

```bash
# semantic metrics
python 150.semantic_eval.py chathistory_annotated_llava_med.json results/llava_med/OSD OSD
python 150.semantic_eval.py llava_med.csv results/llava_med/DWTD DWTD

# LLM judges (vLLM server must be running, see below)
python 150.gemma_eval.py chathistory_annotated_llava_med.json results/llava_med/OSD OSD
python 150.llama_eval.py chathistory_annotated_llava_med.json results/llava_med/OSD OSD
```

The generated files come from the scripts in the baselines folder. A model's JSON may use `label` instead of `differential_diagnosis`. Both are mapped to `differential_diagnosis`.

## Reference data

| Mode | Generated descriptors | Reference |
|---|---|---|
| OSD | `GENERATED_DESCRIPTORS/TEST_OPEN_SOURCE_GEN_DATA/<pid>/<filename>` | `TEST_OPEN_SOURCE_FULL_DATA/<pid>/chathistory_annotated.json` |
| RPD | `GENERATED_DESCRIPTORS/TEST_REAL_PATIENT_GEN_DATA/<pid>/<cid>/<vid>/<filename>` | `TEST_REAL_PATIENT_FULL_DATA/<pid>/<cid>/<vid>/chathistory_annotated.json` |
| DWTD | `DATASET_WISE_TEST_DATA/generated_descriptors/<filename>` | a reference CSV (`GOLD_DWTD_PATH`) keyed by `dataset_name` and `image_name` |

Set all paths in the constants at the top of each script.

## Semantic metrics (`150.semantic_eval.py`)

| Metric | Model | Notes | Output files |
|---|---|---|---|
| Cosine similarity | `sentence-transformers/all-MiniLM-L6-v2` | cosine of sentence embeddings | `allminilm.csv` |
| Cosine similarity | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | loaded through sentence-transformers (mean pooling) | `sapbert.csv` |
| BERTScore | `bert-base-uncased` | library-default layer | `p_bert.csv`, `r_bert.csv`, `f_bert.csv` |
| BERTScore | `dmis-lab/biobert-base-cased-v1.2` (BioBERT) | layer 12 | `p_biomedbert.csv`, `r_biomedbert.csv`, `f_biomedbert.csv` |
| BERTScore | `emilyalsentzer/Bio_ClinicalBERT` | layer 12 | `p_bioclinbert.csv`, `r_bioclinbert.csv`, `f_bioclinbert.csv` |

BERTScore is computed without IDF weighting or baseline rescaling.

## LLM-as-judge (`150.gemma_eval.py`, `150.llama_eval.py`)

For every attribute, the judge sees the field name, the reference answer and the generated answer, and returns one integer from 1 to 10:

```text
You are an expert dermatologist evaluating AI-generated medical descriptions.

Field: {field}
Reference answer: {reference}
Generated answer: {generated}

Score the generated answer from 1 to 10 based on:
- Clinical accuracy
- Completeness compared to reference
- Relevance to the field

Respond with ONLY a single integer from 1 to 10. No explanation.
```

- Decoding: `temperature=0`, `max_tokens=128`, single user turn
- Each cell is the majority vote of 3 queries; if all three differ, the median is used
- The first integer in 1-10 found in the reply is taken as the score; replies without one count as failed
- Queries are spread round-robin over the configured ports, and a failed port is skipped

### Serving the judges

`MODEL_NAME` in each script must match `--served-model-name`, and the ports must match `LLAMA_PORTS`.

| Script | `--served-model-name` | Ports |
|---|---|---|
| `150.gemma_eval.py` | `gemma_4b` | 8010 |
| `150.llama_eval.py` | `llama_8b` | 8011, 8012 |

```bash
# Gemma judge
VLLM_ATTENTION_BACKEND=TORCH_SDPA CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model <path-to-gemma-4b> \
    --served-model-name gemma_4b \
    --port 8010 \
    --gpu-memory-utilization 0.55 \
    --max-model-len 512 \
    --tensor-parallel-size 1 \
    --trust-remote-code

# Llama judge (one server per port / GPU)
CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.1-8B-Instruct --served-model-name llama_8b --port 8011
CUDA_VISIBLE_DEVICES=1 vllm serve meta-llama/Llama-3.1-8B-Instruct --served-model-name llama_8b --port 8012
```

## Outputs

One CSV per metric, written to `output_dir`. Rows are cases and columns are the 15 attributes.

| Mode | ID columns |
|---|---|
| OSD | `pid` |
| RPD | `pid`, `cid`, `vid` |
| DWTD | `dataset_name`, `image_name` |

| Script | Files |
|---|---|
| `150.semantic_eval.py` | 11 CSVs (table above), values are similarity / BERTScore in [0, 1] range |
| `150.gemma_eval.py` | `gemma_4b.csv`, integer scores 1-10 |
| `150.llama_eval.py` | `llama_8b.csv`, integer scores 1-10 |

A cell is `NaN` when the reference or the generated answer is empty or invalid (empty, `nan`, or text containing "error", which covers the `API_ERROR:` / `ERROR:` markers from generation), or when every judge query failed. `NaN` cells are skipped, not scored as zero.

## Resuming

All scripts reload existing output CSVs and only process cases that still have at least one unscored attribute. The judge scripts save every 20 cases. Cases that legitimately contain `NaN` cells are re-processed on each run, which is harmless but slower.

## Aggregation

These scripts write per-case scores only. Report means with `pandas` (`DataFrame.mean(skipna=True)`) per attribute and overall.

## Requirements

`pandas`, `requests`, `torch`, `transformers`, `sentence-transformers`, `bert-score`; `vllm` for the judge servers.
