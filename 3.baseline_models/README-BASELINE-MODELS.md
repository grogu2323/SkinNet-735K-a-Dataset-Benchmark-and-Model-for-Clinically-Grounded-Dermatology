# Baseline Models

Scripts used to generate the baseline descriptors for the ten comparison models in the paper. Each script asks the same 15 dermatology questions and writes answers in the same schema, so all outputs can be scored with the same evaluation code.

## Scripts

| Script | Model | Backend |
|---|---|---|
| `101.lingshu_7b.py` | Lingshu-7B | vLLM |
| `102.lingshu_32b.py` | Lingshu-32B | vLLM |
| `103.med_r1_dermo_2b.py` | MED-R1-DERMO (2B, Qwen2-VL) | Transformers |
| `104.llava_med_7b.py` | LLaVA-Med v1.5 (Mistral-7B) | LLaVA codebase |
| `105.med_gemma_4b.py` | MedGemma-4B | vLLM |
| `106.med_gemma_27b.py` | MedGemma-27B | vLLM |
| `107.dermato_llama_11b.py` | DermatoLlama-0.1 (Llama-3.2-11B-Vision) | Transformers |
| `108.skinvl_7b.py` | SkinVL-PubMM (7B) | LLaVA codebase |
| `109.gemini.py` | Gemini 2.5 Flash | Google API |
| `110.qwen_generation.py` | Qwen2-VL-72B | vLLM |
| `111.ours_4b.py` | **Finetuned-Med-Gemma-4B** | vLLM |
| `112.ours_27b.py` | **Finetuned-Med-Gemma-27B** | vLLM |

## Protocol

- **Test data:** three modes, selected with `--mode`
  - `open_source`: one folder per case containing `chathistory_annotated.json` and images
  - `real_patient`: `patient/case/visit/` folders with the same contents
  - `dataset_wise`: single images from `yuhos`, `scin`, `mm_skin_qa`, `fitzpatrick_17k` and `biomedica`
- **Questions:** the 15 prompts of Appendix C (`You are an Expert Dermatologist. ... Answer in N words maximum.`), plus the system message *"You are an Expert Dermatologist analyzing medical images. Give me precise answers only. Do not give explanations."*
- **Single-turn:** every question is a separate request with the images re-sent. The exception is Gemini (see below).
- **Images:** the first *N* images of a case in sorted filename order, where *N* is the model's `MAX_IMAGES`.
- **Question order:** the differential diagnosis is asked 13th here.

## Per-model settings

Max new tokens are listed as *12 short attributes / differential diagnosis / additional notes / overall description*.

| Model | Image size | Max images | Max new tokens | Decoding | Serving |
|---|---|---|---|---|---|
| Lingshu-7B | 256×256 | 6 | 50 / 250 / 512 / 512 | T=0 | vLLM `:8010`, 32 concurrent |
| Lingshu-32B | 256×256 | 6 | 50 / 250 / 512 / 512 | T=0 | vLLM `:8009`, 32 concurrent |
| MED-R1-DERMO | 256×256 | 5 | 100 / 350 / 750 / 750 | greedy | fp16, `--batch_size` (default 50) |
| LLaVA-Med | 256×256 | 6 | 100 / 350 / 750 / 750 | greedy | fp16, sequential |
| MedGemma-4B | 256×256 | 6 | 100 / 350 / 750 / 750 | T=0 | vLLM `:8008`, 32 concurrent |
| MedGemma-27B | 256×256 | 6 | 100 / 350 / 750 / 750 | T=0 | vLLM `:8007`, 32 concurrent |
| DermatoLlama | 224×224 | 5 | 50 / 250 / 350 / 350 | greedy | fp16, sequential |
| SkinVL-PubMM | original (model's own processor) | 10 | 200 / 350 / 750 / 750 | greedy | fp16, sequential |
| Gemini 2.5 Flash | downscaled to 512×512 if larger | 10 | API default | API default | sequential, 0.5 s between calls |
| Qwen2-VL-72B | 128×128 | 2 | 50 / 256 / 384 / 384 | T=0.1, top-p 0.9, frequency penalty 0.5, repetition penalty 1.15 | 3 vLLM servers `:8001-8003`, 4 concurrent each, round-robin |
| Finetuned-Med-Gemma-4B | 896×896 | 6 | 30 / 300 / 600 / 600 | T=variable | vLLM `:8011`, 32 concurrent |
| Finetuned-Med-Gemma-27B | 896×896 | 6 | 30 / 300 / 600 / 600 | T=variable | vLLM `:8012`, 32 concurrent |

**Gemini** does not use the 15 separate questions. It gets one request per case with a system instruction asking for a single JSON object containing all 15 attributes (no word limits).

## Requirements

- All scripts: `pandas`, `pillow`
- vLLM scripts (101, 102, 105, 106, 110, 111, 112): `vllm`, `openai`
- Transformers scripts (103, 107): `torch`, `transformers`, `accelerate`
- LLaVA scripts (104, 108): the [LLaVA](https://github.com/haotian-liu/LLaVA) / LLaVA-Med codebase
- Gemini (109): `google-genai` and `export GEMINI_API_KEY=...`

## Configuration

Edit the constants at the top of each script before running:

- `OPEN_SOURCE_DATA`, `REAL_PATIENT_DATA`, `DATASET_FOLDERS`: input data
- `OUTPUT_OPEN_SOURCE`, `OUTPUT_REAL_PATIENT`, `OUTPUT_CSV`: output locations
- `MODEL_PATH` (Transformers/LLaVA scripts) or `VLLM_BASE_URL` / `VLLM_SERVERS` (vLLM scripts)

## Hosting with vLLM

`MODEL_NAME` in each script must match `--served-model-name`, and the port must match the script.

```bash
# MedGemma-4B example (adjust flags to your vLLM version)
vllm serve <model-id-or-path> \
  --served-model-name med_gemma_4b \
  --port 8008 \
  --limit-mm-per-prompt '{"image": 6}'
```

| Script | `--served-model-name` | Port |
|---|---|---|
| 101 | `lingshu_7b` | 8010 |
| 102 | `lingshu_32b` | 8009 |
| 105 | `med_gemma_4b` | 8008 |
| 106 | `med_gemma_27b` | 8007 |
| 110 | `qwen_72b` | 8001, 8002, 8003 (one server per GPU group) |
| 111 | `finetuned-med_gemma_4b` | 8011 |
| 112 | `finetuned-med_gemma_27b` | 8012 |

## Running

```bash
python 105.med_gemma_4b.py --mode open_source
python 105.med_gemma_4b.py --mode real_patient
python 105.med_gemma_4b.py --mode dataset_wise
python 105.med_gemma_4b.py                 # all three
python 103.med_r1_dermo_2b.py --batch_size 8   # reduce on OOM
```

## Outputs

- **Open-source / real-patient:** one JSON per case, `chathistory_annotated_<model>.json`, written next to the case ID under the output folder. Keys are the 15 attributes: `type_of_lesion`, `site`, `count`, `arrangement`, `size`, `color_pattern`, `border`, `surface_changes`, `presence_of_exudate_or_discharge`, `surrounding_skin_changes`, `secondary_changes`, `pattern_or_shape`, `differential_diagnosis`, `additional_notes`, `overall_description`.
- **Dataset-wise:** one CSV per model with columns `dataset_name`, `image_name` and the 15 attributes.
- **Failures** are stored in the cell as text starting with `API_ERROR:`, `IMAGE_ERROR:`, `BATCH_ERROR:` or `ERROR:`. Filter these before scoring.

| Script | JSON suffix | CSV |
|---|---|---|
| 101 | `lingshu_7b` | `lingshu_7b.csv` |
| 102 | `lingshu_32b` | `lingshu_32b.csv` |
| 103 | `medr1_dermo` | `med_r1_dermo.csv` |
| 104 | `llava_med` | `llava_med.csv` |
| 105 | `med_gemma` | `med_gemma_4b.csv` |
| 106 | `med_gemma` | `med_gemma_27b.csv` |
| 107 | `dermatollama` | `dermatollama.csv` |
| 108 | `skinvl` | `skinvl_pubmm.csv` |
| 109 | `gemini` | `gemini.csv` |
| 110 | `qwen2vl_72b` | `qwen2vl_72b.csv` |
| 111 | `finetuned_med_gemma_4b` | `ours_4b.py.csv` |
| 112 | `finetuned_med_gemma_27b` | `ours_27b.py.csv` |

## Resuming and overwriting

- 104 and 108 skip cases whose output is already complete and re-run any case with an empty answer or one containing "error".
- 101 and 102 resume the dataset-wise CSV only. All other modes overwrite existing outputs.
- The remaining scripts regenerate everything on each run.
