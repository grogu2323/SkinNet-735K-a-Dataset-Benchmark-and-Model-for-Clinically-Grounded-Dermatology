"""
Semantic evaluation — no lexical metrics.
Usage: python code_semantic.py <json_or_csv_name> <output_dir> <OSD|RPD|DWTD>

json_or_csv_name : e.g. "llava_med.json" or "llava_med.csv"
                   For OSD/RPD: walks subfolders looking for chathistory_annotated_<stem>.json
                   For DWTD:    looks for <name> directly in generated_descriptors/
output_dir       : directory where metric CSVs are written
mode             : OSD | RPD | DWTD

Output CSVs:
  allminilm.csv
  sapbert.csv
  p_bert.csv  r_bert.csv  f_bert.csv
  p_biomedbert.csv  r_biomedbert.csv  f_biomedbert.csv
  p_bioclinbert.csv r_bioclinbert.csv f_bioclinbert.csv
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
import json
import math
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score_fn
import bert_score.utils as _bsu

# Monkey-patch: bert_score passes max_length=None to the fast tokenizer when
# num_layers is supplied, causing OverflowError. Force it to 512.
_orig_sent_encode = _bsu.sent_encode
def _safe_sent_encode(tokenizer, sent):
    if getattr(tokenizer, 'model_max_length', None) is None or tokenizer.model_max_length > 1_000_000:
        tokenizer.model_max_length = 512
    return _orig_sent_encode(tokenizer, sent)
_bsu.sent_encode = _safe_sent_encode

# ── data roots ────────────────────────────────────────────────────────────────
OSD_ROOT  = "/home/pratyushprasad/ACL/DATA/GENERATED_DESCRIPTORS/TEST_OPEN_SOURCE_GEN_DATA"
RPD_ROOT  = "/home/pratyushprasad/ACL/DATA/GENERATED_DESCRIPTORS/TEST_REAL_PATIENT_GEN_DATA"
DWTD_ROOT = "/home/pratyushprasad/ACL/DATA/DATASET_WISE_TEST_DATA/generated_descriptors"

GOLD_OSD_ROOT = "/home/pratyushprasad/ACL/DATA/TRAIN_TEST_DATA/TEST_OPEN_SOURCE_FULL_DATA"
GOLD_RPD_ROOT = "/home/pratyushprasad/ACL/DATA/TRAIN_TEST_DATA/TEST_REAL_PATIENT_FULL_DATA"
GOLD_DWTD_PATH = "/home/pratyushprasad/ACL/DATA/DATASET_WISE_TEST_DATA/generated_descriptors/gemini.csv"
GOLD_JSON     = "chathistory_annotated.json"

# ── models ────────────────────────────────────────────────────────────────────
MINILM_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
SAPBERT_MODEL     = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
BERT_MODEL        = "bert-base-uncased"
BIOMEDBERT_MODEL  = "dmis-lab/biobert-base-cased-v1.2"
BIOCLINBERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"

# ── canonical fields ──────────────────────────────────────────────────────────
ALL_COLS = [
    "type_of_lesion", "site", "count", "arrangement", "size",
    "color_pattern", "border", "surface_changes",
    "presence_of_exudate_or_discharge", "surrounding_skin_changes",
    "secondary_changes", "pattern_or_shape", "differential_diagnosis",
    "additional_notes", "overall_description",
]
COL_ALIASES = {"label": "differential_diagnosis"}


# ── helpers ───────────────────────────────────────────────────────────────────
_gold_dwtd_df = None
def load_gold_dwtd(dname, iname):
    global _gold_dwtd_df
    if _gold_dwtd_df is None:
        _gold_dwtd_df = pd.read_csv(GOLD_DWTD_PATH, dtype=str)
    df = _gold_dwtd_df
    row = df[(df["dataset_name"].str.strip() == dname) & (df["image_name"].str.strip() == iname)]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {normalize_col(c): r[c] for c in df.columns if normalize_col(c) in ALL_COLS}

def is_bad(v) -> bool:
    if v is None:
        return True
    try:
        if math.isnan(float(v)):
            return True
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return s == "" or s == "nan" or "error" in s.lower()


def normalize_col(c: str) -> str:
    return COL_ALIASES.get(c.strip(), c.strip())


def safe_str(v) -> str:
    return "" if is_bad(v) else str(v).strip()


def cosine_sim(model, ref: str, hyp: str) -> float:
    if not ref or not hyp:
        return float("nan")
    e1 = model.encode(ref, convert_to_tensor=True)
    e2 = model.encode(hyp, convert_to_tensor=True)
    return float(util.cos_sim(e1, e2).item())


def bertscore_batch(refs, hyps, model_type, num_layers=None):
    valid_idx = [i for i, (r, h) in enumerate(zip(refs, hyps)) if r and h]
    P = [float("nan")] * len(refs)
    R = [float("nan")] * len(refs)
    F = [float("nan")] * len(refs)
    if not valid_idx:
        return P, R, F
    kwargs = dict(model_type=model_type, lang="en", verbose=False)
    if num_layers is not None:
        kwargs["num_layers"] = num_layers
    p, r, f = bert_score_fn(
        [hyps[i] for i in valid_idx],
        [refs[i] for i in valid_idx],
        **kwargs,
    )
    for j, i in enumerate(valid_idx):
        P[i] = float(p[j])
        R[i] = float(r[j])
        F[i] = float(f[j])
    return P, R, F


def list_to_dict(lst):
    return {ALL_COLS[i]: lst[i] for i in range(len(ALL_COLS))}


# ── gold loaders ──────────────────────────────────────────────────────────────
def load_gold_osd(pid):
    path = os.path.join(GOLD_OSD_ROOT, pid, GOLD_JSON)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {normalize_col(k): v for k, v in json.load(f).items()}


def load_gold_rpd(pid, cid, vid):
    path = os.path.join(GOLD_RPD_ROOT, pid, cid, vid, GOLD_JSON)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {normalize_col(k): v for k, v in json.load(f).items()}


# ── input collectors ──────────────────────────────────────────────────────────
def collect_osd(json_filename):
    """Walk OSD_ROOT/<pid>/<json_filename>, return [(pid, fields)]"""
    entries = []
    for pid in sorted(os.listdir(OSD_ROOT)):
        path = os.path.join(OSD_ROOT, pid, json_filename)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            raw = json.load(f)
        entries.append((pid, {normalize_col(k): v for k, v in raw.items()}))
    return entries


def collect_rpd(json_filename):
    """Walk RPD_ROOT/<pid>/<cid>/<vid>/<json_filename>, return [((pid,cid,vid), fields)]"""
    entries = []
    for pid in sorted(os.listdir(RPD_ROOT)):
        ppath = os.path.join(RPD_ROOT, pid)
        if not os.path.isdir(ppath):
            continue
        for cid in sorted(os.listdir(ppath)):
            cpath = os.path.join(ppath, cid)
            if not os.path.isdir(cpath):
                continue
            for vid in sorted(os.listdir(cpath)):
                path = os.path.join(cpath, vid, json_filename)
                if not os.path.exists(path):
                    continue
                with open(path) as f:
                    raw = json.load(f)
                entries.append(((pid, cid, vid), {normalize_col(k): v for k, v in raw.items()}))
    return entries


def collect_dwtd(csv_filename):
    """Read DWTD_ROOT/<csv_filename>, return [((dname,iname), fields)]"""
    path = os.path.join(DWTD_ROOT, csv_filename)
    df   = pd.read_csv(path, dtype=str)
    entries = []
    for _, row in df.iterrows():
        dname  = str(row.get("dataset_name", "")).strip()
        iname  = str(row.get("image_name", "")).strip()
        fields = {normalize_col(c): row.get(c, "") for c in df.columns if normalize_col(c) in ALL_COLS}
        entries.append(((dname, iname), fields))
    return entries


# ── CSV resume helpers ────────────────────────────────────────────────────────
def load_existing(csv_path, id_cols):
    existing = {}
    if not os.path.exists(csv_path):
        return existing
    try:
        df = pd.read_csv(csv_path, dtype=str)
        for _, row in df.iterrows():
            key = tuple(str(row[c]).strip() for c in id_cols)
            scores = {}
            for col in ALL_COLS:
                try:
                    scores[col] = float(row[col])
                except (KeyError, TypeError, ValueError):
                    scores[col] = float("nan")
            existing[key] = scores
    except Exception:
        pass
    return existing


def all_scored(scores):
    return all(not math.isnan(scores.get(c, float("nan"))) for c in ALL_COLS)


def save_scores(csv_path, id_cols, all_rows):
    records = []
    for key_tuple, scores in all_rows.items():
        rec = {id_cols[i]: key_tuple[i] for i in range(len(id_cols))}
        for col in ALL_COLS:
            rec[col] = scores.get(col, float("nan"))
        records.append(rec)
    pd.DataFrame(records, columns=id_cols + ALL_COLS).to_csv(csv_path, index=False)


# ── main ──────────────────────────────────────────────────────────────────────
def run_eval(filename, output_dir, mode):
    os.makedirs(output_dir, exist_ok=True)

    if mode == "OSD":
        id_cols     = ["pid"]
        entries     = collect_osd(filename)
        get_gold    = lambda key: load_gold_osd(key)
        key_tuple   = lambda k: (k,)
    elif mode == "RPD":
        id_cols     = ["pid", "cid", "vid"]
        entries     = collect_rpd(filename)
        get_gold    = lambda key: load_gold_rpd(*key)
        key_tuple   = lambda k: k
    elif mode == "DWTD":
        id_cols     = ["dataset_name", "image_name"]
        entries     = collect_dwtd(filename)
        get_gold    = lambda key: load_gold_dwtd(key[0], key[1])
        key_tuple   = lambda k: k
    else:
        raise ValueError(f"Unknown mode: {mode}")

    metric_files = {
        "allminilm":     os.path.join(output_dir, "allminilm.csv"),
        "sapbert":       os.path.join(output_dir, "sapbert.csv"),
        "p_bert":        os.path.join(output_dir, "p_bert.csv"),
        "r_bert":        os.path.join(output_dir, "r_bert.csv"),
        "f_bert":        os.path.join(output_dir, "f_bert.csv"),
        "p_biomedbert":  os.path.join(output_dir, "p_biomedbert.csv"),
        "r_biomedbert":  os.path.join(output_dir, "r_biomedbert.csv"),
        "f_biomedbert":  os.path.join(output_dir, "f_biomedbert.csv"),
        "p_bioclinbert": os.path.join(output_dir, "p_bioclinbert.csv"),
        "r_bioclinbert": os.path.join(output_dir, "r_bioclinbert.csv"),
        "f_bioclinbert": os.path.join(output_dir, "f_bioclinbert.csv"),
    }

    existing = {m: load_existing(p, id_cols) for m, p in metric_files.items()}
    all_rows = {m: dict(existing[m]) for m in metric_files}

    pending = [(key, gen) for key, gen in entries
               if any(not all_scored(existing[m].get(key_tuple(key), {})) for m in metric_files)]
    skipped = len(entries) - len(pending)
    print(f"  {mode}: {len(entries)} total | {skipped} skipped | {len(pending)} pending")

    if not pending:
        print("  All done.")
        return

    print("  Loading embedding models...")
    minilm  = SentenceTransformer(MINILM_MODEL)
    sapbert = SentenceTransformer(SAPBERT_MODEL)
    print("  Models ready.")

    # Build flat lists across ALL pending patients × ALL cols
    # Shape: all_refs[patient_idx * N_COLS + col_idx]
    N = len(ALL_COLS)
    keys_ordered = []
    all_refs_flat = []
    all_hyps_flat = []

    for key, gen in pending:
        gold = get_gold(key)
        keys_ordered.append(key_tuple(key))
        for c in ALL_COLS:
            all_refs_flat.append(safe_str(gold.get(c, "")))
            all_hyps_flat.append(safe_str(gen.get(c, "")))

    # ── cosine sims (fast, no reloading) ─────────────────────────────────────
    print(f"  Computing cosine sims over {len(pending)} patients...")
    for idx, kt in enumerate(keys_ordered):
        refs = all_refs_flat[idx*N:(idx+1)*N]
        hyps = all_hyps_flat[idx*N:(idx+1)*N]
        all_rows["allminilm"][kt] = {c: cosine_sim(minilm,  refs[i], hyps[i]) for i, c in enumerate(ALL_COLS)}
        all_rows["sapbert"][kt]   = {c: cosine_sim(sapbert, refs[i], hyps[i]) for i, c in enumerate(ALL_COLS)}

    # ── BERTScore — one pass per model over ALL text ──────────────────────────
    for model_type, num_layers, p_key, r_key, f_key in [
        (BERT_MODEL,        None, "p_bert",        "r_bert",        "f_bert"),
        (BIOMEDBERT_MODEL,  12,   "p_biomedbert",  "r_biomedbert",  "f_biomedbert"),
        (BIOCLINBERT_MODEL, 12,   "p_bioclinbert", "r_bioclinbert", "f_bioclinbert"),
    ]:
        print(f"  BERTScore: {model_type} ...")
        P_flat, R_flat, F_flat = bertscore_batch(all_refs_flat, all_hyps_flat, model_type, num_layers)
        for idx, kt in enumerate(keys_ordered):
            all_rows[p_key][kt] = list_to_dict(P_flat[idx*N:(idx+1)*N])
            all_rows[r_key][kt] = list_to_dict(R_flat[idx*N:(idx+1)*N])
            all_rows[f_key][kt] = list_to_dict(F_flat[idx*N:(idx+1)*N])

    for idx, kt in enumerate(keys_ordered):
        print(f"  [{idx+1}/{len(pending)}] {kt}")

    for m, p in metric_files.items():
        save_scores(p, id_cols, all_rows[m])
        print(f"  ✓ {p}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python code_semantic.py <filename> <output_dir> <OSD|RPD|DWTD>")
        sys.exit(1)
    run_eval(sys.argv[1], sys.argv[2], sys.argv[3].upper())