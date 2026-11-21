"""
LLM-judge evaluation using google/gemma-4-E4B-it via vLLM.
Usage: python code_llm_gemma.py <filename> <output_dir> <OSD|RPD|DWTD>

filename   : e.g. "llava_med.json" or "llava_med.csv"
output_dir : directory where gemma_4b.csv is written
mode       : OSD | RPD | DWTD

export CUDA_HOME=/usr/local/cuda-13.0;
export PATH=$CUDA_HOME/bin:$PATH;
export CPATH=$CUDA_HOME/targets/x86_64-linux/include;
export C_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include;
export CPLUS_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include;

VLLM_ATTENTION_BACKEND=TORCH_SDPA CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model /home/pratyushprasad/ACL/MODELS/gemm_4b \
    --served-model-name gemma_4b \
    --port 8010 \
    --gpu-memory-utilization 0.55 \
    --max-model-len 512 \
    --tensor-parallel-size 1 \
    --trust-remote-code

Each cell = integer score 1-10 (majority vote of 3 queries)
"""

import os
import sys
import json
import math
import time
import requests
from collections import Counter

import pandas as pd

# ── SET THESE FOR YOUR VLLM SETUP ────────────────────────────────────────────
LLAMA_PORTS = list(range(8010, 8011))   # 8000, 8001, 8002
MODEL_NAME  = "gemma_4b"               # must match --served-model-name
OUTPUT_FILE = "gemma_4b.csv"
# ─────────────────────────────────────────────────────────────────────────────

REQUEST_TIMEOUT = 120

OSD_ROOT  = "/home/pratyushprasad/ACL/DATA/GENERATED_DESCRIPTORS/TEST_OPEN_SOURCE_GEN_DATA"
RPD_ROOT  = "/home/pratyushprasad/ACL/DATA/GENERATED_DESCRIPTORS/TEST_REAL_PATIENT_GEN_DATA"
DWTD_ROOT = "/home/pratyushprasad/ACL/DATA/DATASET_WISE_TEST_DATA/generated_descriptors"

GOLD_OSD_ROOT  = "/home/pratyushprasad/ACL/DATA/TRAIN_TEST_DATA/TEST_OPEN_SOURCE_FULL_DATA"
GOLD_RPD_ROOT  = "/home/pratyushprasad/ACL/DATA/TRAIN_TEST_DATA/TEST_REAL_PATIENT_FULL_DATA"
GOLD_JSON      = "chathistory_annotated.json"
GOLD_DWTD_PATH = "/home/pratyushprasad/ACL/DATA/TRAIN_TEST_DATA/annotated_dwtd.csv"

ALL_COLS = [
    "type_of_lesion", "site", "count", "arrangement", "size",
    "color_pattern", "border", "surface_changes",
    "presence_of_exudate_or_discharge", "surrounding_skin_changes",
    "secondary_changes", "pattern_or_shape", "differential_diagnosis",
    "additional_notes", "overall_description",
]
COL_ALIASES = {"label": "differential_diagnosis"}

JUDGE_PROMPT = """\
You are an expert dermatologist evaluating AI-generated medical descriptions.

Field: {field}
Reference answer: {reference}
Generated answer: {generated}

Score the generated answer from 1 to 10 based on:
- Clinical accuracy
- Completeness compared to reference
- Relevance to the field

Respond with ONLY a single integer from 1 to 10. No explanation."""


# ── helpers ───────────────────────────────────────────────────────────────────
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


def normalize_col(c):
    return COL_ALIASES.get(c.strip(), c.strip())


def safe_str(v):
    return "" if is_bad(v) else str(v).strip()


def get_active_ports(ports):
    active = []
    for port in ports:
        try:
            r = requests.get(f"http://localhost:{port}/v1/models", timeout=3)
            if r.status_code == 200:
                active.append(port)
        except Exception:
            pass
    return active


def query_llm(port, prompt):
    """Query vLLM served gemma_4b — no Qwen-specific chat_template_kwargs."""
    r = requests.post(
        f"http://localhost:{port}/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128,
            "temperature": 0.0,
        },
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def parse_score(text):
    if "\n\n" in text:
        text = text.split("\n\n")[-1]
    for tok in text.strip().split():
        tok = tok.strip(".,")
        try:
            v = int(tok)
            if 1 <= v <= 10:
                return float(v)
        except ValueError:
            pass
    return float("nan")


def majority_score(scores_3):
    """Return majority vote of 3 scores; fall back to median if all different."""
    valid = [s for s in scores_3 if not math.isnan(s)]
    if not valid:
        return float("nan")
    counts = Counter(valid)
    top_count = counts.most_common(1)[0][1]
    if top_count >= 2:
        # majority exists — return the most common value
        return float(counts.most_common(1)[0][0])
    # all different — return median (middle value of sorted 3)
    return float(sorted(valid)[len(valid) // 2])


def query_with_majority(active_ports, port_idx, prompt):
    """Query 3 times and return (majority_score, updated_port_idx)."""
    scores_3 = []
    for _ in range(3):
        for attempt in range(len(active_ports)):
            port = active_ports[(port_idx + attempt) % len(active_ports)]
            try:
                s = parse_score(query_llm(port, prompt))
                scores_3.append(s)
                port_idx = (port_idx + 1) % len(active_ports)
                break
            except Exception as e:
                print(f"  [WARN] port {port}: {e}")
        else:
            scores_3.append(float("nan"))

    return majority_score(scores_3), port_idx


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


# ── input collectors ──────────────────────────────────────────────────────────
def collect_osd(json_filename):
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
    path = os.path.join(DWTD_ROOT, csv_filename)
    df   = pd.read_csv(path, dtype=str)
    entries = []
    for _, row in df.iterrows():
        dname  = str(row.get("dataset_name", "")).strip()
        iname  = str(row.get("image_name", "")).strip()
        fields = {normalize_col(c): row.get(c, "") for c in df.columns if normalize_col(c) in ALL_COLS}
        entries.append(((dname, iname), fields))
    return entries


# ── CSV resume ────────────────────────────────────────────────────────────────
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


def save_csv(csv_path, id_cols, all_rows):
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
    csv_path = os.path.join(output_dir, OUTPUT_FILE)

    if mode == "OSD":
        id_cols   = ["pid"]
        entries   = collect_osd(filename)
        get_gold  = lambda key: load_gold_osd(key)
        key_tuple = lambda k: (k,)
    elif mode == "RPD":
        id_cols   = ["pid", "cid", "vid"]
        entries   = collect_rpd(filename)
        get_gold  = lambda key: load_gold_rpd(*key)
        key_tuple = lambda k: k
    elif mode == "DWTD":
        id_cols   = ["dataset_name", "image_name"]
        entries   = collect_dwtd(filename)
        get_gold  = lambda key: load_gold_dwtd(key[0], key[1])
        key_tuple = lambda k: k
    else:
        raise ValueError(f"Unknown mode: {mode}")

    existing = load_existing(csv_path, id_cols)
    all_rows = dict(existing)

    pending = [(key, gen) for key, gen in entries
               if not all_scored(existing.get(key_tuple(key), {}))]
    skipped = len(entries) - len(pending)
    print(f"  {mode}: {len(entries)} total | {skipped} skipped | {len(pending)} pending")

    if not pending:
        print("  All done.")
        return

    active_ports = get_active_ports(LLAMA_PORTS)
    if not active_ports:
        print(f"  ERROR: No active vLLM ports in {LLAMA_PORTS}")
        sys.exit(1)
    print(f"  Active ports: {active_ports}")

    port_idx = 0
    start    = time.time()

    for idx, (key, gen) in enumerate(pending, 1):
        kt     = key_tuple(key)
        gold   = get_gold(key)
        scores = {}

        for col in ALL_COLS:
            ref = safe_str(gold.get(col, ""))
            hyp = safe_str(gen.get(col, ""))

            if not ref or not hyp:
                scores[col] = float("nan")
                continue

            prompt = JUDGE_PROMPT.format(field=col, reference=ref, generated=hyp)
            scores[col], port_idx = query_with_majority(active_ports, port_idx, prompt)

        all_rows[kt] = scores
        rate = idx / (time.time() - start)
        print(f"  [{idx}/{len(pending)}] ({rate:.2f}/s) {kt}")
        if idx <= 10:
            print("  ── feature scores ──")
            for col in ALL_COLS:
                v = scores.get(col, float("nan"))
                marker = "  (skipped)" if math.isnan(v) else ""
                print(f"    {col:<45} {v}{marker}")
            print()

        if idx % 20 == 0:
            save_csv(csv_path, id_cols, all_rows)

    save_csv(csv_path, id_cols, all_rows)
    print(f"  ✓ {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python code_llm_gemma.py <filename> <output_dir> <OSD|RPD|DWTD>")
        sys.exit(1)
    run_eval(sys.argv[1], sys.argv[2], sys.argv[3].upper())