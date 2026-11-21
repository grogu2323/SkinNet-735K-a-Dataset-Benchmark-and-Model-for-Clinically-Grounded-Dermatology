# Prescreening Pipeline

Four-stage image prescreening used to build SkinNet-735K. Each stage sends every image to an instruction-tuned vision-language model (`medgemma-4b-it`, served with vLLM) and records a label, a confidence score, and a short reason. An image is retained only if it receives the pass label at every stage.

## Stages

| # | Stage | Script | Pass | Reject | Resize |
|---|-------|--------|------|--------|--------|
| 1 | Human dermatology check | `prescreen_human_dermatology.py` | `YES` | `NO` | 512 px |
| 2 | Image quality filter | `prescreen_image_quality.py` | `GOOD` | `BAD` | 896 px |
| 3 | Modality filter | `prescreen_microscopic_dermoscopic.py` | `ACCEPT` | `REJECT` | 896 px |
| 4 | Surgery detection | `prescreen_surgery.py` | `NON_SURGERY` | `SURGERY` | 896 px |

### What each stage accepts and rejects

**1. Human dermatology check**
- Accepts: human skin with any lesion, rash, or abnormality; dermoscopic images; mucosal or genital dermatology images; nail, scalp, or hair disorders; close-up clinical skin photos.
- Rejects: non-medical images, histology/microscopy only, animal images, radiology scans, drawings or illustrations, prescriptions.

**2. Image quality filter**
- Accepts: sharp and visible lesions, proper lighting, sufficient resolution, clear clinical details.
- Rejects: extremely blurry images, very dark or overexposed images, severe artifacts or corruption, lesion not visible at all.

**3. Modality filter**
- Accepts: clinical dermatology photos (direct patient skin photographs).
- Rejects: histopathology slides, dermoscopic images (polarised close-up), microscopy / cellular images, laboratory microscopic views.

**4. Surgery detection**
- Accepts (`NON_SURGERY`): normal dermatology lesions, healed surgical scars, post-surgical skin (no active procedure), wounds without instruments present.
- Rejects (`SURGERY`): ongoing surgery with instruments, suturing or stitching in progress, active operative procedures, scalpels or clamps cutting tissue.

The full prompt for each stage is in the `QUESTION` constant of its script.

## Inference settings

All four stages use the same model and decoding settings.

| Setting | Value |
|---------|-------|
| Model | `medgemma-4b-it` served via vLLM |
| Decoding | Greedy (`temperature = 0.0`), so predictions are deterministic and reproducible. Top-p and top-k are left at their defaults and do not affect greedy output. |
| Max new tokens | 128 (enough for the label, confidence, and short rationale) |
| Query format | Single turn: a system message plus a user message containing the question and the image |
| Image encoding | JPEG (quality 90), base64, longest side 512 px (stage 1) or 896 px (stages 2-4) |
| Request timeout | 180 s |
| Retries | Up to 3 attempts with exponential back-off before the image is flagged as failed (HTTP 400 is not retried) |
| Concurrency | 100 requests per server |

## Output fields

Each stage writes a CSV with the image path and three structured fields:

| Field | Type | Description |
|-------|------|-------------|
| `image_path` | String | Absolute path of the image |
| `LABEL` | Categorical | Task-specific pass/reject label (see the stage table) |
| `CONFIDENCE` | Integer (0-100) | Model self-reported confidence score |
| `REASON` | String (max 50 words) | Free-text justification for the label |

The label alone decides pass or fail; no confidence threshold is applied. Rejected images are not deleted. They stay in the CSV with their label, confidence, and reason, so they can be reviewed or re-queried. Images that fail (unreadable file, HTTP error, unparsable reply) are written with an empty `LABEL` and a `REASON` that starts with `ERROR:`.

## Requirements

- Python 3.8+
- `aiohttp`, `Pillow`
- `vllm`, and one or more GPUs that can serve MedGemma-4B-it

```bash
pip install aiohttp pillow vllm
```

Run settings used for SkinNet-735K (fill in before release):

- vLLM version: `<fill in>`
- GPUs: `<fill in>`

## Step 1: host the model with vLLM

The scripts talk to OpenAI-compatible vLLM servers on `localhost`, ports **8000-8003**. Start one server per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve google/medgemma-4b-it --port 8000
CUDA_VISIBLE_DEVICES=1 vllm serve google/medgemma-4b-it --port 8001
CUDA_VISIBLE_DEVICES=2 vllm serve google/medgemma-4b-it --port 8002
CUDA_VISIBLE_DEVICES=3 vllm serve google/medgemma-4b-it --port 8003
```

Add any extra flags your vLLM version needs. The scripts do not need all four servers: at startup they check `/v1/models` on each port, print `[UP]` or `[DOWN]`, and use whichever servers respond. If none respond, they exit. If the servers report different model names, a warning is printed, so use the same model on every port.

## Step 2: run a stage

Each script takes an image folder and an output CSV path:

```bash
python prescreen_human_dermatology.py        IMAGE_FOLDER  stage1_human.csv
python prescreen_image_quality.py            IMAGE_FOLDER  stage2_quality.csv
python prescreen_microscopic_dermoscopic.py  IMAGE_FOLDER  stage3_modality.csv
python prescreen_surgery.py                  IMAGE_FOLDER  stage4_surgery.csv
```

Each script scans `IMAGE_FOLDER` recursively for `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`, and `.gif` files and classifies every image it finds. The usage message each script prints names a slightly different filename; it only affects the help text.

### Resuming and retrying

The scripts are safe to interrupt. On start they read the existing output CSV and skip every image that already has a complete row (label, confidence, and reason present, and no `ERROR:`). Rerunning the same command resumes where it stopped and retries failed rows. At the end the CSV is compacted to one row per image and the script prints the row count, the pass-label count, and how many rows are still incomplete. Rerun until that number is zero.

## Combining the stages

An image passes the pipeline only if it has the pass label in all four CSVs. Treat rows with a missing label or an `ERROR:` as unresolved, not as accepted. For example:

```python
import pandas as pd

keep = {
    "stage1_human.csv":    "YES",
    "stage2_quality.csv":  "GOOD",
    "stage3_modality.csv": "ACCEPT",
    "stage4_surgery.csv":  "NON_SURGERY",
}

passed = None
for csv_path, label in keep.items():
    df = pd.read_csv(csv_path)
    ok = set(df.loc[df["LABEL"] == label, "image_path"])
    passed = ok if passed is None else passed & ok

pd.Series(sorted(passed), name="image_path").to_csv("retained_images.csv", index=False)
```

An image can be flagged by more than one stage, so the per-stage rejection counts add up to more than the number of unique rejected images. Count unique rejections from the combined result.

## Results on SkinNet-735K

Prescreening rejected 65,244 of 803,201 open-source images (8.1%) and retained 737,957 (91.9%). Filter columns give the number of images flagged by each filter (an image can be flagged by more than one); "Unique rej." counts each rejected image once. Percentages are of the source total.

| Source | Total | Non-derm. | Low qual. | Dermosc. | Surgical | Unique rej. | Retained |
|--------|------:|----------:|----------:|---------:|---------:|------------:|---------:|
| Derm1M | 423,016 | 22,229 (5.3) | 18,143 (4.3) | 7,788 (1.8) | 8,532 (2.0) | 38,766 (9.2) | 384,250 (90.8) |
| DermaVQA | 2,945 | 10 (0.3) | 21 (0.7) | 0 (0.0) | 12 (0.4) | 37 (1.3) | 2,908 (98.7) |
| MM-Skin | 11,002 | 513 (4.7) | 44 (0.4) | 950 (8.6) | 164 (1.5) | 1,219 (11.1) | 9,783 (88.9) |
| PAD-UFES-20 | 2,298 | 2 (0.1) | 0 (0.0) | 0 (0.0) | 7 (0.3) | 9 (0.4) | 2,289 (99.6) |
| SCIN | 10,379 | 92 (0.9) | 249 (2.4) | 1 (0.0) | 20 (0.2) | 293 (2.8) | 10,086 (97.2) |
| Biomedica | 69,827 | 3,585 (5.1) | 704 (1.0) | 680 (1.0) | 3,940 (5.6) | 7,731 (11.1) | 62,096 (88.9) |
| Fitzpatrick17k | 16,514 | 65 (0.4) | 60 (0.4) | 35 (0.2) | 284 (1.7) | 388 (2.3) | 16,126 (97.7) |
| DermNet | 262,874 | 8,105 (3.1) | 9,612 (3.7) | 4,074 (1.5) | 5,358 (2.0) | 16,705 (6.4) | 246,169 (93.6) |
| YUHOS16-SkinCaRe | 4,346 | 18 (0.4) | 21 (0.5) | 6 (0.1) | 64 (1.5) | 96 (2.2) | 4,250 (97.8) |
| **Total** | **803,201** | **34,619 (4.3)** | **28,854 (3.6)** | **13,534 (1.7)** | **18,381 (2.3)** | **65,244 (8.1)** | **737,957 (91.9)** |

## Notes and limitations

- Prescreening labels come from a model, not from human review, so they can be wrong. Use `CONFIDENCE` and `REASON` to review borderline images.
- The stage 3 prompt also asks for a `TYPE` line (`CLINICAL`, `DERMOSCOPIC`, or `MICROSCOPIC`), but the script does not parse or save it. Only `LABEL`, `CONFIDENCE`, and `REASON` are written.
- Output CSVs contain absolute local paths. Remove or rewrite them before sharing results.
- Images, including the hospital data, are not part of this repository.
