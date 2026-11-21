import os
import json
import torch
import argparse
import time
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Dict, Any
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda:0")

MODEL_PATH          = "/home/pratyushprasad/ACL/MODELS/SkinVL-PubMM"
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
OUTPUT_CSV = "/home/pratyushprasad/ACL/DATA/DATASET_WISE_TEST_DATA/generated_descriptors/skinvl_pubmm.csv"

MAX_IMAGES   = 10
IMAGE_PREFIX = ""

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

MAX_TOKENS = [200] * 12 + [350, 750, 750]

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

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def first3(s: str) -> str:
    return " ".join(str(s).strip().split()[:3])

def is_bad(v) -> bool:
    s = str(v).strip()
    return s == "" or "error" in s.lower()

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

class SkinVLModel:
    def __init__(self):
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        print(f"  Loading SkinVL-PubMM from {MODEL_PATH} ...")
        model_name = get_model_name_from_path(MODEL_PATH)
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(MODEL_PATH, None, model_name)
        )

        vision_tower = self.model.get_model().get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()

        self.model = self.model.to(DEVICE).half()
        self.model.eval()

        if self.image_processor is None:
            from transformers import CLIPImageProcessor
            self.image_processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14-336"
            )

        # conv_mode detection OUTSIDE the if block — always runs
        ml = model_name.lower()
        if "mistral" in ml:
            self.conv_mode = "mistral_instruct"
        elif "pubmm" in ml or "skinvl" in ml or "v1" in ml:
            self.conv_mode = "llava_v1"
        else:
            self.conv_mode = "llava_v0"

        # tokenizer length — includes added special tokens like <image>
        self.tokenizer_len = len(self.tokenizer)

        print(f"  Model ready on {DEVICE} | conv_mode={self.conv_mode} | tokenizer_len={self.tokenizer_len}")

    @staticmethod
    def load_image(path: str) -> Image.Image:
        with Image.open(path) as im:
            return im.convert("RGB").copy()

    def prepare_tensors(self, pil_images: List[Image.Image]):
        from llava.mm_utils import process_images as llava_process_images
        tensors = llava_process_images(pil_images, self.image_processor, self.model.config)
        if isinstance(tensors, list):
            tensors = [t.to(DEVICE, dtype=torch.float16) for t in tensors]
        else:
            tensors = tensors.to(DEVICE, dtype=torch.float16)
        return tensors

    def infer_one(self, image_tensors, n_images: int, q_idx: int) -> str:
        from llava.mm_utils import tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates

        question   = QUESTIONS[q_idx]
        max_tokens = MAX_TOKENS[q_idx]

        try:
            image_tokens  = DEFAULT_IMAGE_TOKEN * n_images
            full_question = SYSTEM_MESSAGE + "\n\n" + image_tokens + "\n" + question

            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], full_question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(DEVICE)

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensors,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Use len(tokenizer) not vocab_size — includes <image> token at 32000
            safe_output = [t for t in output_ids[0].tolist() if 0 <= t < self.tokenizer_len]

            def safe_decode(token_ids: list) -> str:
                try:
                    return self.tokenizer.decode(
                        torch.tensor(token_ids), skip_special_tokens=True
                    ).strip()
                except Exception:
                    parts = []
                    for t in token_ids:
                        try:
                            parts.append(self.tokenizer.decode([t], skip_special_tokens=True))
                        except Exception:
                            pass
                    return "".join(parts).strip()

            full_output = safe_decode(safe_output)

            # Extract answer after role separator
            for sep in ["ASSISTANT:", "Assistant:", "###Assistant:", "assistant\n"]:
                if sep in full_output:
                    full_output = full_output.split(sep)[-1].strip()
                    break

            return full_output if full_output else "ERROR: empty output"

        except Exception as e:
            return f"ERROR: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_all_questions(
    model: SkinVLModel,
    entries: List[Tuple[List[str], Any]],
    desc: str = "",
) -> Dict[Any, dict]:
    total   = len(entries)
    results: Dict[Any, List[str]] = {lbl: [""] * NUM_QUESTIONS for _, lbl in entries}

    print(f"  [{desc}] Pre-loading images for {total} item(s)...")
    loaded_tensors: Dict[Any, Any] = {}
    loaded_n:       Dict[Any, int] = {}
    error_labels:   set = set()

    for paths, lbl in entries:
        try:
            pil_imgs = [model.load_image(p) for p in paths]
            loaded_tensors[lbl] = model.prepare_tensors(pil_imgs)
            loaded_n[lbl]       = len(pil_imgs)
        except Exception as e:
            for q in range(NUM_QUESTIONS):
                results[lbl][q] = f"IMAGE_ERROR: {e}"
            error_labels.add(lbl)

    valid = [(lbl,) for _, lbl in entries if lbl not in error_labels]

    start       = time.time()
    total_calls = NUM_QUESTIONS * len(valid)
    done_calls  = 0

    for q_idx in range(NUM_QUESTIONS):
        col = ANSWER_TO_COL[q_idx]
        for (lbl,) in valid:
            answer = model.infer_one(loaded_tensors[lbl], loaded_n[lbl], q_idx)
            results[lbl][q_idx] = answer
            done_calls += 1
            print(f"    {col}: {first3(answer)}")

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

# ─────────────────────────────────────────────────────────────────────────────
# COLLECTORS
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def run_open_source(model: SkinVLModel):
    all_entries = collect_open_source()
    pending = [(imgs, pid) for imgs, pid in all_entries
               if not os_already_done(pid, "chathistory_annotated_skinvl.json")]
    skipped = len(all_entries) - len(pending)
    print(f"  Open source : {len(all_entries)} total | {skipped} skipped | {len(pending)} pending")
    if not pending:
        print("  [SKIP] All clean.")
        return

    os.makedirs(OUTPUT_OPEN_SOURCE, exist_ok=True)
    t0      = time.time()
    results = run_all_questions(model, pending, desc="open_source")
    print(f"  Inference done in {time.time()-t0:.1f}s — writing JSON...")

    for imgs, pid in pending:
        out_dir = os.path.join(OUTPUT_OPEN_SOURCE, pid)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "chathistory_annotated_skinvl.json"), "w") as f:
            json.dump(results[pid], f, indent=2)
        dd = first3(results[pid].get("differential_diagnosis") or "")
        print(f"  ✓ {pid} | {dd}")


def run_real_patient(model: SkinVLModel):
    all_entries = collect_real_patient()
    pending = [(imgs, (pid, cid, vid)) for imgs, (pid, cid, vid) in all_entries
               if not rp_already_done(pid, cid, vid, "chathistory_annotated_skinvl.json")]
    skipped = len(all_entries) - len(pending)
    print(f"  Real patient : {len(all_entries)} total | {skipped} skipped | {len(pending)} pending")
    if not pending:
        print("  [SKIP] All clean.")
        return

    os.makedirs(OUTPUT_REAL_PATIENT, exist_ok=True)
    t0      = time.time()
    results = run_all_questions(model, pending, desc="real_patient")
    print(f"  Inference done in {time.time()-t0:.1f}s — writing JSON...")

    for imgs, (pid, cid, vid) in pending:
        out_dir = os.path.join(OUTPUT_REAL_PATIENT, pid, cid, vid)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "chathistory_annotated_skinvl.json"), "w") as f:
            json.dump(results[(pid, cid, vid)], f, indent=2)
        dd = first3(results[(pid, cid, vid)].get("differential_diagnosis") or "")
        print(f"  ✓ {pid}/{cid}/{vid} | {dd}")


def run_dataset_wise(model: SkinVLModel):
    print("  Scanning dataset folders...")
    all_entries = collect_dataset_wise()
    done_set    = dw_already_done_set(OUTPUT_CSV)
    pending     = [(imgs, (d, n)) for imgs, (d, n) in all_entries if (d, n) not in done_set]
    skipped     = len(all_entries) - len(pending)
    print(f"  Dataset-wise : {len(all_entries)} total | {skipped} skipped | {len(pending)} pending")
    if not pending:
        print("  [SKIP] All clean.")
        return

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    t0      = time.time()
    results = run_all_questions(model, pending, desc="dataset_wise")
    print(f"  Inference done in {time.time()-t0:.1f}s — writing CSV...")

    rows = []
    for imgs, (dname, iname) in pending:
        rows.append({"dataset_name": dname, "image_name": iname, **results[(dname, iname)]})

    file_exists = os.path.exists(OUTPUT_CSV)
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, mode="a", header=not file_exists, index=False)
    print(f"  ✓ {len(rows)} rows written to {OUTPUT_CSV}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

MODES = {
    "open_source":  ("OPEN SOURCE GENERATION",  run_open_source),
    "real_patient": ("REAL PATIENT GENERATION",  run_real_patient),
    "dataset_wise": ("DATASET-WISE GENERATION",  run_dataset_wise),
}

def main(modes: List[str]):
    print("\n" + "=" * 70)
    print("  SKINVL-PUBMM DESCRIPTOR GENERATION")
    print("=" * 70)
    print(f"  Model  : {MODEL_PATH}")
    print(f"  Device : {DEVICE}")
    print(f"  Modes  : {', '.join(modes)}")
    print("=" * 70)

    model = SkinVLModel()

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
    parser = argparse.ArgumentParser(description="SkinVL-PubMM descriptor generation")
    parser.add_argument("--mode", choices=list(MODES.keys()), default=None)
    args = parser.parse_args()
    modes_to_run = [args.mode] if args.mode else list(MODES.keys())
    main(modes_to_run)