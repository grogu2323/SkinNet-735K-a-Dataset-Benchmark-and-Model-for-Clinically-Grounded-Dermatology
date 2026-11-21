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

**HuggingFace Dataset:**
- **Dataset Repositry**: Pratyush2323/SkinNet_735K_Dataset

**License:**  
CC-BY-NC-SA-4.0

