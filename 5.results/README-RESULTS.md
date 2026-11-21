# Dataset Statistics

Notebooks that produce the dataset numbers reported in the paper: 
- source composition,
- image resolution,
- prescreening yield,
- train/test split sizes.

## Notebooks

| Notebook | Input | What it reports |
|---|---|---|
| `201.image_resolution_stats_.ipynb` | `FINAL_DATASET/<source>/` (searched recursively) | Per-source image count; min / median / max of width, height and megapixels; number of images below 64×64, 128×128 and 256×256 (width **or** height under the threshold) |
| `202.prescreening_stats.ipynb` | `PRESCREENING/<STAGE>/<subset>.csv` | Images flagged by each of the four prescreening stages, unique images rejected by any stage, and images remaining, per source |
| `203.train_test_data_stats_.ipynb` | `{TRAIN,TEST}_OPEN_SOURCE_FULL_DATA`, `{TRAIN,TEST}_REAL_PATIENT_FULL_DATA` | Patients, complaints, visits and images; images per case; resolution and small-image counts |

## Expected layout

```text
FINAL_DATASET/<source>/...                         # 9 sources, any depth
PRESCREENING/{HUMAN_DERMATOLOGY,IMAGE_QUALITY,
              MICROSCOPIC_DERMOSCOPIC,SURGERY}/<subset>.csv   # columns: image_path, LABEL
<SPLIT>_OPEN_SOURCE_FULL_DATA/<patient>/*.png|jpg
<SPLIT>_REAL_PATIENT_FULL_DATA/<patient>/<complaint>/<visit>/*.png|jpg
```

Set the paths in the constants at the top of each notebook. Requirements: `pandas`, `pillow`.

## 201: source composition and resolution

| Source | Images | Share | Median W × H (px) | Below 256 px |
|---|---:|---:|---|---:|
| DERM_1M | 423,016 | 52.7% | 459 × 391 | 25.6% |
| DERM_VQA | 2,945 | 0.4% | 1024 × 845 | 3.4% |
| MM_SKIN_QA | 11,002 | 1.4% | 426 × 394 | 16.7% |
| PAD_UFES_20 | 2,298 | 0.3% | 780 × 780 | 2.0% |
| SCIN | 10,379 | 1.3% | 810 × 1080 | 0.5% |
| BIOMEDICA_PUBMED | 69,827 | 8.7% | 708 × 508 | 7.4% |
| FITZPATRICK_17K | 16,593 | 2.1% | 458 × 385 | 8.2% |
| MUHOMMAD_DATA | 262,874 | 32.7% | 500 × 472 | 31.4% |
| YUHOS16_SKINCARE | 4,346 | 0.5% | 466 × 394 | 8.4% |
| **All** | **803,280** | | **504 × 450** (0.239 MP median) | **24.8%** |

Across all sources, 0.2% of images are below 64×64 px and 2.5% below 128×128 px. Medians are computed separately for width and height.

## 202: prescreening yield

Labels that count as a rejection: human dermatology `NO`, image quality `BAD`, modality `REJECT`, surgery `REJECT`. An image is retained only if no stage rejects it.

| | Images | Share |
|---|---:|---:|
| Total prescreened | 803,201 | |
| Human dermatology = NO | 34,619 | 4.3% |
| Image quality = BAD | 28,854 | 3.6% |
| Dermoscopic / microscopic = REJECT | 13,534 | 1.7% |
| Surgery = REJECT | 18,381 | 2.3% |
| **Unique images rejected (any stage)** | **65,244** | **8.1%** |
| **Retained** | **737,957** | **91.9%** |

Per-stage counts overlap, so they sum to more than the unique rejections. Per-source tables are printed by the notebook; see also the prescreening folder README.

## 203: train / test splits

| Split | Subset | Patients | Complaints | Cases (patients / visits) | Images |
|---|---|---:|---:|---:|---:|
| Train | Open source | 2,624 | n/a | 2,624 | 4,110 |
| Train | Real patient | 5,963 | 7,055 | 7,117 | 17,293 |
| Test | Open source | 250 | n/a | 250 | 352 |
| Test | Real patient | 1,735 | 2,093 | 2,110 | 5,847 |

Images per case:

| Images | 1 | 2 | 3 | 4 | 5 | 6 | 7+ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train OSD | 1,809 | 144 | 671 | - | - | - | - |
| Train RPD | 3,089 | 1,686 | 948 | 577 | 304 | 189 | 324 |
| Test OSD | 192 | 14 | 44 | - | - | - | - |
| Test RPD | 800 | 484 | 290 | 196 | 107 | 100 | 133 |

- **Resolution:** all real-patient images are 512×512. Open-source training images have median 739 × 810 px (3.8% below 256 px); open-source test images have median 670 × 656 px (2.6% below 256 px).
- **Complaints:** most patients have one complaint (5,180 of 5,963 in train, 1,493 of 1,735 in test); 7,001 of 7,055 train complaints have a single visit.
- **Training cases:** 2,624 + 7,117 = 9,741. Cases with more images than the fine-tuning image limit are split into several samples.
