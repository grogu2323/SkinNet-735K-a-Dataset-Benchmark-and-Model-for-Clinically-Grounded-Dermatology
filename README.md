# SkinNet-735K

SkinNet-735K is a large-scale clinical dermatology dataset built to address four gaps in existing resources: bias toward lighter skin tones, reliance on dermoscopic rather than macroscopic images, inconsistent labeling, and a lack of structured clinical supervision for vision-language models (VLMs).

It consolidates images from nine open-source datasets and pairs each image with:

- **15 expert-informed clinical descriptors**
  - Lesion type
  - Anatomical site
  - Lesion count
  - Arrangement
  - Size
  - Color pattern
  - Border characteristics
  - Surface changes
  - Exudate
  - Surrounding skin findings
  - Secondary changes
  - Pattern / shape
  - Additional notes
  - Overall description
  - Differential diagnosis
- **Fitzpatrick skin tone annotations** (Types I–VI)
- **SNOMED CT identifiers**

To scale annotation beyond manual curation, we fine-tuned MedGemma-4B and 27B on an expert-labeled seed set and used them to generate descriptors for the remaining images. We also provide a held-out, expert-scored test set of 2,862 cases for benchmarking.

This repository contains the dataset, annotations, benchmark, and code to reproduce our results.

---

## Repository Contents

The repository includes:

- Data-construction and preprocessing pipelines
- Caption generation 
- SNOMED CT grounding framework
- Differential diagnosis generation pipeline
- MedGemma fine-tuning code
- Evaluation scripts and benchmark utilities

---

## Dataset Access

The released dataset and models are hosted on the HuggingFace account:

**HuggingFace Models:**  
- **Finetuned-Med-Gemma-27B**: Pratyush2323/SkinNet_735K_Finetuned_Med_Gemma_27b
- **Finetuned-Med-Gemma-4B**: Pratyush2323/SkinNet_735K_Finetuned_Med_Gemma_4b

## Repository Contents

The repository is organized in five steps, in the order of the pipeline. Each folder has its own README with details.

| Folder | What it contains | README |
|---|---|---|
| [`1.prescreening_codes`](1.prescreening_codes) | Four-stage image prescreening (human dermatology, image quality, modality, surgery) using MedGemma-4B served with vLLM | [README](1.prescreening_codes/README-PRESCREENING.md) |
| [`2.finetuning_codes`](2.finetuning_codes) | LoRA fine-tuning of MedGemma-4B and 27B on the expert-labeled seed set, plus the fine-tuning and inference pipeline figure and training curves | [README](2.finetuning_codes/README-FINETUNING-INFERENCING.md) |
| [`3.baseline_models`](3.baseline_models) | Descriptor-generation scripts for the baseline models (Lingshu, MED-R1-DERMO, LLaVA-Med, MedGemma, DermatoLlama, SkinVL, Gemini, Qwen) and for our fine-tuned 4B and 27B models | [README](3.baseline_models/README-BASELINE-MODELS.md) |
| [`4.evaluation_llm_semantic`](4.evaluation_llm_semantic) | Evaluation scripts: semantic similarity (embedding cosine, BERTScore) and LLM-as-judge (Gemma, Llama) | [README](4.evaluation_llm_semantic/README-EVALUATION.md) |
| [`5.results`](5.results) | Notebooks for dataset statistics (source composition, resolution, prescreening yield, train/test splits), score aggregation, and skin-tone-stratified results | [Dataset statistics](5.results/README-DATASET-STATISTICS.md), [Results](5.results/README-RESULTS.md) |
**HuggingFace Dataset:**
- **Dataset-Repositry**: Pratyush2323/SkinNet_735K_Dataset

**License:**  
CC-BY-NC-SA-4.0

