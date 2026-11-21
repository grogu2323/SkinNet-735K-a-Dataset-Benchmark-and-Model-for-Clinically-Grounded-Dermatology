# Score Aggregation and Skin-Tone Analysis

Notebooks that turn the per-case evaluation CSVs (from the evaluation scripts) into the tables reported in the paper.

## Notebooks

| Notebook | Purpose |
|---|---|
| `204.score_final_.ipynb` | Completeness check, per-metric / per-split aggregation tables, per-dataset breakdown for the dataset-wise set |
| `205.skin_tone_wise_scores.ipynb` | BioClinBERT precision / recall / F1 stratified by Fitzpatrick skin tone (FP 1-6) |

## Expected input layout

```text
SCORES/
  OSD/ | RPD/ | DWTD/
    <model>/
      allminilm.csv  sapbert.csv
      p_bert.csv  r_bert.csv  f_bert.csv
      p_biomedbert.csv  r_biomedbert.csv  f_biomedbert.csv
      p_bioclinbert.csv  r_bioclinbert.csv  f_bioclinbert.csv
      gemma_4b.csv  llama_8b.csv
```

Each CSV has one row per case and one column per attribute (15 attributes), with ID columns `pid` (OSD), `pid, cid, vid` (RPD) or `dataset_name, image_name` (DWTD). Set `SCORES_ROOT` and the other paths in the first lines of each cell. Requirements: `pandas`, `numpy`.

## 204: aggregation

| Step | What it does |
|---|---|
| Completeness check | For each split and model, prints the row count of every expected metric CSV, or `MISSING` / `ERR` |
| Per-field means | Mean of each attribute for one metric (SapBERT) per split and model |
| RPD tables | Per-model means for SapBERT and the Gemma judge, plus `average_score`; the Gemma table also reports `count_10` (number of cells scored 10) |
| Main aggregation | For every metric and split, writes `scores_3d/<metric>_<split>.csv` with `row_count`, the 15 attribute means and `average_score`; checks `row_count` against the expected split size (OSD 250, DWTD 530, RPD 2118) |
| Per-dataset breakdown (DWTD) | Splits the dataset-wise scores by `dataset_name` and writes one CSV per dataset to `scores_gemma_4b/`, with one row per model |

### Definitions

- **Attribute mean:** average over cases with a valid score; `NaN` cells (empty or failed outputs) are skipped.
- **`average_score`:** mean of the 15 attribute means. An attribute with no valid score at all counts as 0.
- **`row_count`:** number of rows in the metric CSV, used as a sanity check against the expected test-set size.

## 205: skin-tone analysis

1. Load the Fitzpatrick annotation files: `FITZPATRICK_ANNOTATIONS_OPEN_SOURCE.csv` (key `patient_id`) and `FITZPATRICK_ANNOTATIONS_REAL_PATIENT.csv` (key `patient_id, complaint_id, visit_id`).
2. Keep only rows whose `STATUS` is `agreed` or `adjudicated` and whose `FINAL` value is an integer from 1 to 6.
3. Match annotated cases to the OSD and RPD score rows and pool them.
4. For each tone (FP 1-6) and each of BioClinBERT P / R / F1, average the attribute scores per model, add `avg_score` (mean over attributes) and sort by it.

| FP | Description |
|---|---|
| 1 | Pale white |
| 2 | White |
| 3 | Light brown / olive |
| 4 | Moderate brown |
| 5 | Dark brown |
| 6 | Deeply pigmented |

Output: 18 CSVs named `fp{tone}_{p|r|f}_bioclinbert.csv` in `skin_tone_tables/`. Cases without a valid annotation are excluded.
