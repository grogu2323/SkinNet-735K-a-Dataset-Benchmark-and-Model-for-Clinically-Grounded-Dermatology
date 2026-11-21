# MedGemma Fine-Tuning

![MedGemma fine-tuning and inference pipeline](Finetuning_inferencing.png)
*(A) Fine-tuning on 15-turn conversations, where each prediction is conditioned on the image and
all prior responses. (B) Inference with the same sequential prompting structure at scale.*

Fine-tuning code for MedGemma-4B and MedGemma-27B, used to generate structured clinical descriptors for SkinNet-735K. 
Each case is trained as a single **15-turn conversation**: 
  - one question per clinical attribute,
  - with the image(s) at turn 1 and every earlier question and answer kept in the history for later turns.

## Files

| File | Purpose |
|------|---------|
| `finetuning_4b.py` | LoRA fine-tuning of MedGemma-4B |
| `finetuning_27b.py` | LoRA fine-tuning of MedGemma-27B |

## Train Method

- **Conversation format.** One system message, then 15 user/assistant turns in a fixed order. The image tokens are attached to the first user turn. The full dialogue history is passed at every later turn, so later attributes (for example the differential diagnosis at turn 15) are conditioned on all earlier answers.
- **System message.** *"You are an Expert Dermatologist analyzing medical images. Give me precise answers only. Do not give explanations. I donot want responses outside of the asked questions."*
- **Images.** Each image is converted to RGB and resized to 896 x 896.
- **Quantization.** The base model is loaded in 4-bit NF4 with double quantization. The compute dtype is bfloat16 on GPUs with compute capability 8 or higher, otherwise float16.
- **LoRA.** Adapters are trained with dropout 0.05 and `alpha = 2 x r`. `lm_head` is trained in full alongside the adapters (`modules_to_save`).
- **Loss masking.** Padding tokens and image tokens are masked with `-100` in `collate_fn`.
- **Optimization.** 8-bit paged AdamW, learning rate 2e-5, cosine schedule with 10% warmup, gradient clipping at 1.0, gradient checkpointing, 10 epochs, per-device batch size 1.
- **Validation.** 15% of the samples, split with seed 64, evaluated at the end of every epoch.

### The 15 attributes (turn order)

| Turn | Attribute | Annotation key |
|-----:|-----------|----------------|
| 1 | Type of lesion | `type_of_lesion` |
| 2 | Site | `site` |
| 3 | Count | `count` |
| 4 | Arrangement | `arrangement` |
| 5 | Size | `size` |
| 6 | Color pattern | `color_pattern` |
| 7 | Border | `border` |
| 8 | Surface changes | `surface_changes` |
| 9 | Exudate or discharge | `presence_of_exudate_or_discharge` |
| 10 | Surrounding skin changes | `surrounding_skin_changes` |
| 11 | Secondary changes | `secondary_changes` |
| 12 | Pattern or shape | `pattern_or_shape` |
| 13 | Additional notes | `additional_notes` |
| 14 | Overall description | `overall_description` |
| 15 | Differential diagnosis | `label` (falls back to `differential_diagnosis`) |

Every question starts with "You are an Expert Dermatologist." and ends with a word limit. The full prompts are in the `QUESTIONS` dictionary of each script.

## Differences between the 4B and 27B runs

| Setting | MedGemma-4B | MedGemma-27B |
|---------|-------------|--------------|
| LoRA rank `r` | 16 | 64 |
| LoRA alpha | 32 | 128 |
| LoRA target modules | all linear layers | MLP projections (`gate_proj`, `up_proj`, `down_proj`) |
| Max images per sample | 6 | 8 |
| Gradient accumulation | 4 steps | 2 steps |
| Checkpoint saving | every epoch | every 2000 steps |
| Total samples | 10,071 (train 8,560 / val 1,511) | 9,859 (train 8,380 / val 1,479) |
| Training time (reported) | about 88 h | about 45 h |

Shared settings: 4-bit NF4, learning rate 2e-5, 10 epochs, per-device batch size 1, 85/15 split with seed 64, 896 x 896 images, dropout 0.05.

With two GPUs (DDP), the effective batch size per optimizer step is `1 x gradient accumulation x 2`, so 8 for 4B and 4 for 27B.

## Training data format

The scripts walk `OPEN_DIR` (open-source cases) and `REAL_DIR` (real-patient cases) and treat every folder that contains a `chathistory_annotated.json` file as one case:

```
case_001/
  image_1.jpg
  image_2.png
  chathistory_annotated.json
```

- Images (`.png`, `.jpg`, `.jpeg`) are sorted by file name. A case with more images than the limit (6 for 4B, 8 for 27B) is split into several samples of at most that many images. This is why the 4B and 27B runs have different sample counts.
- The JSON file holds one entry per attribute, using the annotation keys in the table above. A missing attribute is replaced with the text "Not specified". A list-style label such as `"['a', 'b']"` is joined into `a, b`.
- Cases are skipped, and listed with a reason in `OUTPUT_DIR/skipped.json`, if the JSON is unreadable, the folder has no images, or the case has no diagnosis label.

## Requirements

- Python 3.9+
- `torch`, `transformers`, `peft`, `trl`, `datasets`, `bitsandbytes`, `scikit-learn`, `pillow`,
  `tensorboard`
- Library versions used: `<fill in>`
- Hardware used for the reported runs: one node with two 80 GB GPUs (DDP), 128 CPU cores, 128 GB RAM

```bash
pip install torch transformers peft trl datasets bitsandbytes scikit-learn pillow tensorboard
```

## Configuration

Edit the paths at the top of each script before running:

```python
MODEL_PATH = "path/to/medgemma-base-model"   # local copy of the base model
OPEN_DIR   = "path/to/train_open_source"     # open-source training cases
REAL_DIR   = "path/to/train_real_patient"    # real-patient training cases
OUTPUT_DIR = "path/to/output"                # checkpoints, logs, final adapter
```

The base models are the MedGemma-4B and MedGemma-27B instruction-tuned checkpoints from Hugging
Face (accepting the Hugging Face license is required): `<fill in exact model ids>`. The scripts
set `HF_HUB_OFFLINE=1`, so the base model must already be downloaded locally. They also set
`CUDA_VISIBLE_DEVICES="0,1"`; change this to match your machine.

## Running

The model is placed on the GPU given by `LOCAL_RANK`, so launch with `torchrun` for DDP:

```bash
torchrun --nproc_per_node=2 finetuning_4b.py
torchrun --nproc_per_node=2 finetuning_27b.py
```

Training resumes automatically from the latest checkpoint in `OUTPUT_DIR`, if one exists.

### Outputs

| Output | Location |
|--------|----------|
| Checkpoints | `OUTPUT_DIR/checkpoint-*` |
| Final LoRA adapter | `OUTPUT_DIR` (via `save_model`) |
| Trainer state | `OUTPUT_DIR` (via `save_state`) |
| TensorBoard logs | `OUTPUT_DIR/logs` |
| Skipped cases | `OUTPUT_DIR/skipped.json` |

View the curves with `tensorboard --logdir OUTPUT_DIR/logs`.

## Results

Per-epoch validation metrics. The best MedGemma-27B checkpoint is epoch 3 (lowest validation
loss); MedGemma-4B plateaus from epoch 7.

| Epoch | 4B loss | 4B token acc. | 4B entropy | 27B loss | 27B token acc. | 27B entropy |
|------:|--------:|--------------:|-----------:|---------:|---------------:|------------:|
| 1 | 0.3132 | 0.9180 | 0.3181 | 0.2062 | 0.9383 | 0.2104 |
| 2 | 0.2617 | 0.9276 | 0.2631 | 0.1902 | 0.9418 | 0.1894 |
| 3 | 0.2471 | 0.9304 | 0.2371 | **0.1865** | 0.9427 | 0.1749 |
| 4 | 0.2401 | 0.9319 | 0.2284 | 0.1877 | 0.9430 | 0.1594 |
| 5 | 0.2361 | 0.9329 | 0.2251 | 0.1936 | 0.9427 | 0.1487 |
| 6 | 0.2342 | 0.9331 | 0.2277 | 0.2052 | 0.9420 | 0.1344 |
| 7 | **0.2336** | 0.9334 | 0.2229 | 0.2174 | 0.9412 | 0.1261 |
| 8 | 0.2336 | 0.9335 | 0.2196 | 0.2270 | 0.9406 | 0.1208 |
| 9 | 0.2335 | 0.9335 | 0.2201 | 0.2303 | 0.9404 | 0.1187 |
| 10 | 0.2335 | 0.9335 | 0.2201 | 0.2304 | 0.9404 | 0.1186 |

Checkpoint used for inference: `<fill in>`

## Inference (used for SkinNet-735K)

After fine-tuning, the LoRA adapters are merged back into the base model and served with vLLM.
Inference uses the same 15-turn conversation format as training:

- Each case is a single 15-turn conversation; the images and the full dialogue history are kept at every turn.
- Up to six images per open-source or real-patient case are encoded once and attached to every user message, so one case sends up to 6 x 15 = 90 image payloads. Benchmark datasets use one image per call.
- Greedy decoding (temperature 0.0) at a fixed 896 x 896 resolution. Top-p values 1.0, 0.9, 0.8, 0.7, and 0.6 were also run as reproducibility checks; under greedy decoding they have no effect.
- Per-attribute token budgets range from 20 tokens (short descriptors) to 600 tokens (additional notes and overall description). The budget for the differential diagnosis is 300 tokens.

| Group | Attributes | Token budget |
|-------|------------|-------------:|
| Primary morphology | Type of lesion, count, arrangement, size | 20 each |
| Primary morphology | Site | 40 |
| Appearance | Border, surface changes | 20 each |
| Appearance | Color pattern | 60 |
| Secondary findings | Exudate or discharge | 20 |
| Secondary findings | Surrounding skin changes, secondary changes, pattern or shape | 100 each |
| Synthesis | Additional notes, overall description | 600 each |
| Synthesis | Differential diagnosis | 300 |

The merge and inference code is not part of these scripts: `<link or folder>`.

## Notes

- Hardcoded local paths in the scripts must be replaced with your own.
- `load_best_model_at_end` is off, so the final saved model is from the last epoch. Use the checkpoint noted under Results for inference.
- Images and annotations are not part of this repository.
