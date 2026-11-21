import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import json
import math
import torch
from pathlib import Path
from PIL import Image
import time
import argparse
import pandas as pd
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images as llava_process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

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
OUTPUT_CSV = "/home/pratyushprasad/ACL/DATA/DATASET_WISE_TEST_DATA/generated_descriptors/llava_med.csv"

MODEL_PATH = "/home/pratyushprasad/ACL/MODELS/llava-med-v1.5-mistral-7b"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MAX_IMAGES = 6

# ─────────────────────────────────────────────────────────────────────────────
# QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────

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

MAX_TOKENS = [100] * 12 + [350, 750, 750]

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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def first3(s: str) -> str:
    return " ".join(str(s).strip().split()[:3])

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

def os_already_done(pid: str, json_filename: str) -> bool:
    fpath = os.path.join(OUTPUT_OPEN_SOURCE, pid, json_filename)
    if not os.path.exists(fpath):
        return False
    try:
        with open(fpath) as f:
            data = json.load(f)
        return not any(is_bad(v) for v in data.values())
    except Exception:
        return False

def rp_already_done(pid: str, cid: str, vid: str, json_filename: str) -> bool:
    fpath = os.path.join(OUTPUT_REAL_PATIENT, pid, cid, vid, json_filename)
    if not os.path.exists(fpath):
        return False
    try:
        with open(fpath) as f:
            data = json.load(f)
        return not any(is_bad(v) for v in data.values())
    except Exception:
        return False

def dw_already_done_set(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path)
        done = set()
        for _, row in df.iterrows():
            if not any(is_bad(v) for v in row.values):
                done.add((str(row["dataset_name"]), str(row["image_name"])))
        return done
    except Exception:
        return set()

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class ModelInference:
    def __init__(self):
        print("[INFO] Loading LLaVA-Med v1.5 Mistral-7B...")
        model_name = get_model_name_from_path(MODEL_PATH)
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(MODEL_PATH, None, model_name, device_map="auto")
        )
        self.model.eval()

        model_name_lower = model_name.lower()
        if "mistral" in model_name_lower:
            self.conv_mode = "mistral_instruct"
        elif "v1.6" in model_name_lower or "34b" in model_name_lower:
            self.conv_mode = "chatml_direct"
        elif "v1" in model_name_lower:
            self.conv_mode = "llava_v1"
        else:
            self.conv_mode = "llava_v0"
        print(f"[INFO] Model loaded on {DEVICE}, conv_mode={self.conv_mode}")

    def process_image(self, image_path: str):
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            im = im.resize((256, 256), Image.LANCZOS)
        return im

    def generate_answer(self, images: list, question: str, max_tokens: int) -> str:
        try:
            image_tokens  = DEFAULT_IMAGE_TOKEN * len(images)
            full_question = SYSTEM_MESSAGE + "\n\n" + image_tokens + "\n" + question

            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], full_question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(self.model.device)

            image_tensors = llava_process_images(images, self.image_processor, self.model.config)
            if isinstance(image_tensors, list):
                image_tensors = [t.to(self.model.device, dtype=torch.float16) for t in image_tensors]
            else:
                image_tensors = image_tensors.to(self.model.device, dtype=torch.float16)

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    images=image_tensors,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Try full decode first — preserves spaces correctly
            try:
                full_output = self.tokenizer.decode(
                    output_ids[0], skip_special_tokens=True
                ).strip()
            except Exception:
                # Fallback: token by token — avoids piece id errors
                parts = []
                for t in output_ids[0].tolist():
                    try:
                        parts.append(self.tokenizer.decode([t], skip_special_tokens=True))
                    except Exception:
                        pass
                full_output = "".join(parts).strip()

            # Extract answer after role separator
            for sep in ["ASSISTANT:", "Assistant:", "[/INST]", "###Assistant:"]:
                if sep in full_output:
                    full_output = full_output.split(sep)[-1].strip()
                    break

            torch.cuda.empty_cache()
            return full_output if full_output else "ERROR: empty output"

        except Exception as e:
            return f"ERROR: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_images(inference: ModelInference, image_paths: list) -> dict:
    try:
        images = [inference.process_image(p) for p in image_paths]
    except Exception as e:
        return {col: f"ERROR: {e}" for col in ANSWER_TO_COL.values()}

    result = {}
    for q_idx, question in enumerate(QUESTIONS):
        col    = ANSWER_TO_COL[q_idx]
        answer = inference.generate_answer(images, question, MAX_TOKENS[q_idx])
        result[col] = answer
        print(f"    {col}: {first3(answer)}")

    return result

# ─────────────────────────────────────────────────────────────────────────────
# COLLECTORS
# ─────────────────────────────────────────────────────────────────────────────

def _pick_images(folder: str) -> list:
    imgs = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if Path(f).suffix.lower() in IMAGE_EXTS
    ])
    return imgs[:MAX_IMAGES]

def collect_open_source() -> list:
    entries = []
    for pid in sorted(os.listdir(OPEN_SOURCE_DATA)):
        ppath = os.path.join(OPEN_SOURCE_DATA, pid)
        if not os.path.isdir(ppath):
            continue
        if not os.path.exists(os.path.join(ppath, "chathistory_annotated.json")):
            continue
        imgs = _pick_images(ppath)
        if imgs:
            entries.append((pid, imgs))
    return entries

def collect_real_patient() -> list:
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
                    entries.append((pid, cid, vid, imgs))
    return entries

def collect_dataset_images() -> list:
    all_images = []
    for dataset_name, folder_path in DATASET_FOLDERS.items():
        if not os.path.isdir(folder_path):
            print(f"[WARN] Dataset folder not found: {folder_path}")
            continue
        images = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if Path(f).suffix.lower() in IMAGE_EXTS
        ])
        print(f"  {dataset_name}: {len(images)} image(s)")
        for img_path in images:
            all_images.append((dataset_name, Path(img_path).name, img_path))
    return all_images

# ─────────────────────────────────────────────────────────────────────────────
# RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def run_open_source(inference: ModelInference):
    all_entries = collect_open_source()
    pending     = [(pid, imgs) for pid, imgs in all_entries
                   if not os_already_done(pid, "chathistory_annotated_llava_med.json")]
    skipped     = len(all_entries) - len(pending)
    total       = len(pending)
    print(f"  Open source : {len(all_entries)} total | {skipped} skipped | {total} pending")
    if not pending:
        print("  [SKIP] All clean.")
        return

    os.makedirs(OUTPUT_OPEN_SOURCE, exist_ok=True)
    completed, start_time = 0, time.time()

    for pid, imgs in pending:
        result  = process_images(inference, imgs)
        out_dir = os.path.join(OUTPUT_OPEN_SOURCE, pid)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "chathistory_annotated_llava_med.json"), "w") as f:
            json.dump(result, f, indent=2)
        completed += 1
        rate = completed / (time.time() - start_time)
        dd   = first3(result.get("differential_diagnosis") or "")
        print(f"  [{completed}/{total}] ({rate:.2f}/s) ✓ {pid} | {dd}")


def run_real_patient(inference: ModelInference):
    all_entries = collect_real_patient()
    pending     = [(pid, cid, vid, imgs) for pid, cid, vid, imgs in all_entries
                   if not rp_already_done(pid, cid, vid, "chathistory_annotated_llava_med.json")]
    skipped     = len(all_entries) - len(pending)
    total       = len(pending)
    print(f"  Real patient : {len(all_entries)} total | {skipped} skipped | {total} pending")
    if not pending:
        print("  [SKIP] All clean.")
        return

    os.makedirs(OUTPUT_REAL_PATIENT, exist_ok=True)
    completed, start_time = 0, time.time()

    for pid, cid, vid, imgs in pending:
        result  = process_images(inference, imgs)
        out_dir = os.path.join(OUTPUT_REAL_PATIENT, pid, cid, vid)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "chathistory_annotated_llava_med.json"), "w") as f:
            json.dump(result, f, indent=2)
        completed += 1
        rate = completed / (time.time() - start_time)
        dd   = first3(result.get("differential_diagnosis") or "")
        print(f"  [{completed}/{total}] ({rate:.2f}/s) ✓ {pid}/{cid}/{vid} | {dd}")


def run_dataset_wise(inference: ModelInference):
    print("  Scanning dataset folders...")
    all_images  = collect_dataset_images()
    done        = dw_already_done_set(OUTPUT_CSV)
    pending     = [(d, n, p) for d, n, p in all_images if (d, n) not in done]
    skipped     = len(all_images) - len(pending)
    total       = len(pending)
    print(f"  Dataset-wise : {len(all_images)} total | {skipped} skipped | {total} pending")
    if not pending:
        print("  [SKIP] All clean.")
        return

    file_exists   = os.path.exists(OUTPUT_CSV)
    completed, start_time = 0, time.time()

    for dataset_name, img_name, img_path in pending:
        try:
            result = process_images(inference, [img_path])
        except Exception as e:
            print(f"  [ERROR] {dataset_name}/{img_name}: {e}")
            result = {col: f"ERROR: {e}" for col in ANSWER_TO_COL.values()}

        row = {"dataset_name": dataset_name, "image_name": img_name, **result}
        pd.DataFrame([row]).to_csv(OUTPUT_CSV, mode="a", header=not file_exists, index=False)
        file_exists = True

        completed += 1
        rate = completed / (time.time() - start_time)
        dd   = first3(result.get("differential_diagnosis") or "")
        print(f"  [{completed}/{total}] ({rate:.2f}/s) ✓ {dataset_name}/{img_name} | {dd}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

MODES = {
    "open_source":  ("OPEN SOURCE GENERATION",  run_open_source),
    "real_patient": ("REAL PATIENT GENERATION",  run_real_patient),
    "dataset_wise": ("DATASET-WISE GENERATION",  run_dataset_wise),
}

def main(modes: list):
    print("\n" + "=" * 70)
    print("  LLAVA-MED v1.5 MISTRAL-7B DESCRIPTOR GENERATION")
    print("=" * 70)
    print(f"  Model      : {MODEL_PATH}")
    print(f"  Device     : {DEVICE}")
    print(f"  Modes      : {', '.join(modes)}")
    print("=" * 70)

    inference = ModelInference()

    for mode in modes:
        label, runner = MODES[mode]
        print(f"\n{'─' * 70}")
        print(f"  [{label}]")
        print(f"{'─' * 70}")
        t0 = time.time()
        runner(inference)
        print(f"  Done in {time.time()-t0:.1f}s")

    print("\n" + "=" * 70)
    print("  ALL GENERATION COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA-Med descriptor generation")
    parser.add_argument("--mode", choices=list(MODES.keys()), default=None)
    args = parser.parse_args()
    modes_to_run = [args.mode] if args.mode else list(MODES.keys())
    main(modes_to_run)