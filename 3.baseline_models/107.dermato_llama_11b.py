import os
import json
import torch
import argparse
import time
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Dict, Any

import pandas as pd
from transformers import AutoProcessor, MllamaForConditionalGeneration

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH          = "/home/pratyushprasad/ACL/MODELS/DermatoLlama-0.1"
BASE_MODEL          = "meta-llama/Llama-3.2-11B-Vision-Instruct"
OPEN_SOURCE_DATA    = "/home/pratyushprasad/ACL/DATA/TRAIN_TEST_DATA/TEST_OPEN_SOURCE_FULL_DATA"
REAL_PATIENT_DATA   = "/home/pratyushprasad/ACL/DATA/TRAIN_TEST_DATA/TEST_REAL_PATIENT_FULL_DATA"
OUTPUT_OPEN_SOURCE  = "/home/pratyushprasad/ACL/DATA/GENERATED_DESCRIPTORS/TEST_OPEN_SOURCE_GEN_DATA"
OUTPUT_REAL_PATIENT = "/home/pratyushprasad/ACL/DATA/GENERATED_DESCRIPTORS/TEST_REAL_PATIENT_GEN_DATA"

DATASET_FOLDERS = {
    "yuhos":           "/home/pratyushprasad/ACL/DATA/DATASET_WISE_TEST_DATA/yuhos",
    "scin":            "/home/pratyushprasad/ACL/DATA/DATASET_WISE_TEST_DATA/scin",
    "mm_skin_qa":      "/home/pratyushprasad/ACL/DATA/DATASET_WISE_TEST_DATA/mm_skin_qa",
    "fitzpatrick_17k": "/home/pratyushprasad/ACL/DATA/DATASET_WISE_TEST_DATA/fitzpatrick_17k",
    "biomedica":       "/home/pratyushprasad/ACL/DATA/DATASET_WISE_TEST_DATA/biomedica",
}
OUTPUT_CSV = "/home/pratyushprasad/ACL/DATA/DATASET_WISE_TEST_DATA/generated_descriptors/dermatollama.csv"

MAX_IMAGES   = 5
IMAGE_PREFIX = ""
IMAGE_SIZE   = 224      # Llama Vision uses 224×224

QUESTIONS = [
    "You are an Expert Dermatologist. What type of lesion is shown in this dermatological image? Answer in 16 words maximum.",
    "You are an Expert Dermatologist. Where is this lesion located on the body? Answer in 16 words maximum.",
    "You are an Expert Dermatologist. How many lesions are present in this image? Answer in 16 words maximum.",
    "You are an Expert Dermatologist. How are the lesions arranged in this image? Answer in 16 words maximum.",
    "You are an Expert Dermatologist. What is the size of the lesion(s) in this image? Answer in 16 words maximum.",
    "You are an Expert Dermatologist. What color pattern does this lesion exhibit? Answer in 16 words maximum.",
    "You are an Expert Dermatologist. How would you describe the border of this lesion? Answer in 16 words maximum.",
    "You are an Expert Dermatologist. What surface changes are visible in this lesion? Answer in 16 words maximum.",
    "You are an Expert Dermatologist. Is there any exudate or discharge present? Answer in 16 words maximum.",
    "You are an Expert Dermatologist. What changes are visible in the surrounding skin? Answer in 16 words maximum.",
    "You are an Expert Dermatologist. What secondary changes can you identify? Answer in 16 words maximum.",
    "You are an Expert Dermatologist. What is the pattern or shape of this lesion? Answer in 16 words maximum.",
    "You are an Expert Dermatologist. What is the possible differential diagnosis? I need 5 differential diagnoses maximum. Answer in 128 words maximum.",
    "You are an Expert Dermatologist. Based on the clinical features and diagnosis, what else do you see in the image? Answer in 256 words maximum.",
    "You are an Expert Dermatologist. Describe the overall clinical appearance of the lesion(s) in detail and relevant diagnostic context. Answer in 256 words maximum.",
]

MAX_TOKENS = [50] * 12 + [250, 350, 350]

SYSTEM_MESSAGE = (
    "You are an Expert Dermatologist analyzing medical images. "
    "Give me precise answers only. Do not give explanations."
)

ANSWER_TO_COL = {
    0:  "type_of_lesion",
    1:  "site",
    2:  "count",
    3:  "arrangement",
    4:  "size",
    5:  "color_pattern",
    6:  "border",
    7:  "surface_changes",
    8:  "presence_of_exudate_or_discharge",
    9:  "surrounding_skin_changes",
    10: "secondary_changes",
    11: "pattern_or_shape",
    12: "differential_diagnosis",
    13: "additional_notes",
    14: "overall_description",
}

IMAGE_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
NUM_QUESTIONS = len(QUESTIONS)

class DermatoLlamaModel:
    def __init__(self):
        print(f"  Loading DermatoLlama-0.1 from {MODEL_PATH} ...")
        print(f"  Processor from base: {BASE_MODEL}")
        self.processor = AutoProcessor.from_pretrained(BASE_MODEL)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto",
        )
        self.model.eval()
        print(f"  Model ready on {DEVICE}")

    @staticmethod
    def load_image(path: str) -> Image.Image:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im = im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            return im.copy()

    def infer_one(
        self,
        images: List[Image.Image],
        q_idx: int,
    ) -> str:
        question   = QUESTIONS[q_idx]
        max_tokens = MAX_TOKENS[q_idx]

        try:
            messages = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": [{"type": "image"} for _ in images]
                    + [{"type": "text", "text": question}],
                },
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text], images=images, return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    use_cache=True,
                    num_beams=1,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )

            gen    = outputs[0][inputs["input_ids"].shape[1]:]
            answer = self.processor.decode(gen, skip_special_tokens=True)
            return answer.strip()

        except Exception as e:
            return f"ERROR: {e}"

def run_all_questions(
    model: DermatoLlamaModel,
    entries: List[Tuple[List[str], Any]],   # ([img_paths], label)
    desc: str = "",
) -> Dict[Any, dict]:
    total   = len(entries)
    results: Dict[Any, List[str]] = {lbl: [""] * NUM_QUESTIONS for _, lbl in entries}

    print(f"  [{desc}] Pre-loading images for {total} item(s)...")
    loaded:       Dict[Any, List[Image.Image]] = {}
    error_labels: set = set()

    for paths, lbl in entries:
        try:
            loaded[lbl] = [model.load_image(p) for p in paths]
        except Exception as e:
            for q in range(NUM_QUESTIONS):
                results[lbl][q] = f"IMAGE_ERROR: {e}"
            error_labels.add(lbl)

    valid = [(lbl,) for _, lbl in entries if lbl not in error_labels]

    start       = time.time()
    total_calls = NUM_QUESTIONS * len(valid)
    done_calls  = 0

    for q_idx in range(NUM_QUESTIONS):
        for (lbl,) in valid:
            results[lbl][q_idx] = model.infer_one(loaded[lbl], q_idx)
            done_calls += 1

        torch.cuda.empty_cache()

        elapsed = time.time() - start
        rate    = done_calls / elapsed if elapsed > 0 else 0.0
        pct     = 100 * (q_idx + 1) / NUM_QUESTIONS
        print(
            f"  [{desc}] Q{q_idx+1:02d}/{NUM_QUESTIONS} done | "
            f"{done_calls}/{total_calls} calls | "
            f"{rate:.1f} calls/s | {pct:.0f}%"
        )

    return {
        lbl: {ANSWER_TO_COL[i]: answers[i] for i in range(NUM_QUESTIONS)}
        for lbl, answers in results.items()
    }

def _pick_images(folder: str) -> List[str]:
    imgs = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if Path(f).suffix.lower() in IMAGE_EXTS and f.startswith(IMAGE_PREFIX)
    ])
    return imgs[:MAX_IMAGES]


def collect_open_source() -> List[Tuple[List[str], str]]:
    entries = []
    for pid in sorted(os.listdir(OPEN_SOURCE_DATA)):
        ppath = os.path.join(OPEN_SOURCE_DATA, pid)
        if not os.path.isdir(ppath):
            continue
        if not os.path.exists(os.path.join(ppath, "chathistory_annotated.json")):
            continue
        imgs = _pick_images(ppath)
        if imgs:
            entries.append((imgs, pid))
    return entries

def collect_real_patient() -> List[Tuple[List[str], Tuple]]:
    entries = []
    for pid in sorted(os.listdir(REAL_PATIENT_DATA)):
        ppath = os.path.join(REAL_PATIENT_DATA, pid)
        if not os.path.isdir(ppath):
            continue
        for cid in sorted(os.listdir(ppath)):
            cpath = os.path.join(ppath, cid)
            if not os.path.isdir(cpath):
                continue
            for vid in sorted(os.listdir(cpath)):
                vpath = os.path.join(cpath, vid)
                if not os.path.isdir(vpath):
                    continue
                if not os.path.exists(os.path.join(vpath, "chathistory_annotated.json")):
                    continue
                imgs = _pick_images(vpath)
                if imgs:
                    entries.append((imgs, (pid, cid, vid)))
    return entries

def collect_dataset_wise() -> List[Tuple[List[str], Tuple]]:
    entries = []
    for dname, folder in DATASET_FOLDERS.items():
        if not os.path.isdir(folder):
            print(f"  [WARN] {dname} not found: {folder}")
            continue
        imgs = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if Path(f).suffix.lower() in IMAGE_EXTS
        ])
        print(f"  {dname}: {len(imgs)} images")
        for p in imgs:
            entries.append(([p], (dname, Path(p).name)))
    return entries

def run_open_source(model: DermatoLlamaModel):
    all_entries = collect_open_source()
    print(f"  Open source patients : {len(all_entries)} — regenerating ALL (overwrite mode)")
    if not all_entries:
        print("  [SKIP] No data found.")
        return

    os.makedirs(OUTPUT_OPEN_SOURCE, exist_ok=True)
    t0      = time.time()
    results = run_all_questions(model, all_entries, desc="open_source")
    print(f"  Inference done in {time.time()-t0:.1f}s — writing JSON...")

    for imgs, pid in all_entries:
        out_dir = os.path.join(OUTPUT_OPEN_SOURCE, pid)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "chathistory_annotated_dermatollama.json"), "w") as f:
            json.dump(results[pid], f, indent=2)
        dd = (results[pid].get("differential_diagnosis") or "")[:50].replace("\n", " ")
        print(f"  ✓ {pid} | {dd}")

def run_real_patient(model: DermatoLlamaModel):
    all_entries = collect_real_patient()
    print(f"  Real patient visits  : {len(all_entries)} — regenerating ALL (overwrite mode)")
    if not all_entries:
        print("  [SKIP] No data found.")
        return

    os.makedirs(OUTPUT_REAL_PATIENT, exist_ok=True)
    t0      = time.time()
    results = run_all_questions(model, all_entries, desc="real_patient")
    print(f"  Inference done in {time.time()-t0:.1f}s — writing JSON...")

    for imgs, (pid, cid, vid) in all_entries:
        out_dir = os.path.join(OUTPUT_REAL_PATIENT, pid, cid, vid)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "chathistory_annotated_dermatollama.json"), "w") as f:
            json.dump(results[(pid, cid, vid)], f, indent=2)
        dd = (results[(pid, cid, vid)].get("differential_diagnosis") or "")[:50].replace("\n", " ")
        print(f"  ✓ {pid}/{cid}/{vid} | {dd}")

def run_dataset_wise(model: DermatoLlamaModel):
    print("  Scanning dataset folders...")
    all_entries = collect_dataset_wise()
    print(f"  Total images : {len(all_entries)} — regenerating ALL (overwrite mode)")
    if not all_entries:
        print("  [SKIP] No images found.")
        return

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    t0      = time.time()
    results = run_all_questions(model, all_entries, desc="dataset_wise")
    print(f"  Inference done in {time.time()-t0:.1f}s — writing CSV...")

    rows = []
    for imgs, (dname, iname) in all_entries:
        row = {"dataset_name": dname, "image_name": iname, **results[(dname, iname)]}
        rows.append(row)

    # Full overwrite
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, mode="w", header=True, index=False)
    print(f"  ✓ {len(rows)} rows written to {OUTPUT_CSV}")

MODES = {
    "open_source":  ("OPEN SOURCE GENERATION",  run_open_source),
    "real_patient": ("REAL PATIENT GENERATION",  run_real_patient),
    "dataset_wise": ("DATASET-WISE GENERATION",  run_dataset_wise),
}

def main(modes: List[str]):
    print("\n" + "=" * 70)
    print("  DERMATOLLAMA-0.1 DESCRIPTOR GENERATION")
    print("=" * 70)
    print(f"  Model      : {MODEL_PATH}")
    print(f"  Base model : {BASE_MODEL}")
    print(f"  Device     : {DEVICE}  (CUDA_VISIBLE_DEVICES=0)")
    print(f"  Modes      : {', '.join(modes)}")
    print("=" * 70)

    model = DermatoLlamaModel()

    for mode in modes:
        label, runner = MODES[mode]
        print(f"\n{'─' * 70}")
        print(f"  [{label}]")
        print(f"{'─' * 70}")
        t0 = time.time()
        runner(model)
        print(f"  Done in {time.time()-t0:.1f}s")

    print("\n" + "=" * 70)
    print("  ALL GENERATION COMPLETE")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DermatoLlama-0.1 descriptor generation")
    parser.add_argument(
        "--mode",
        choices=list(MODES.keys()),
        default=None,
        help="Which mode to run. Omit to run all three.",
    )
    args = parser.parse_args()

    modes_to_run = [args.mode] if args.mode else list(MODES.keys())
    main(modes_to_run)