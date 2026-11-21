import os
import io
import json
import time
import argparse
from pathlib import Path
from PIL import Image
from typing import List, Tuple

import pandas as pd
from google import genai
from google.genai import types

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "PUT_YOUR_KEY_HERE")
GEMINI_MODEL   = "models/gemini-2.5-flash"

RESIZE_TO      = 512
SLEEP_BETWEEN  = 0.5          # seconds between API calls
MAX_RETRIES    = 3
MAX_IMAGES     = 10
IMAGE_PREFIX   = ""

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
OUTPUT_CSV = "/home/pratyushprasad/ACL/DATA/DATASET_WISE_TEST_DATA/generated_descriptors/gemini.csv"

SYSTEM_MESSAGE = """\
You are a dermatology assistant trained to analyze skin lesions. \
You will be given image(s) of the dermatological lesion(s). \
Provide only one structured JSON response with combined information from all images.

Ensure your response follows EXACTLY this JSON format (no extra text outside the JSON):

{
  "type_of_lesion": "<describe the lesion type>",
  "site": "<mention site of the lesion>",
  "count": "<mention the number of lesions>",
  "arrangement": "<describe the distribution>",
  "size": "<approximate lesion size>",
  "color_pattern": "<describe the color and pattern>",
  "border": "<mention if well-defined or ill-defined>",
  "surface_changes": "<describe scaling, crusting, ulceration, etc.>",
  "presence_of_exudate_or_discharge": "<yes/no, and describe if present>",
  "surrounding_skin_changes": "<mention any erythema, dryness, etc.>",
  "secondary_changes": "<mention any secondary changes>",
  "pattern_or_shape": "<describe any specific shape or distribution>",
  "additional_notes": "<mention any extra details>",
  "overall_description": "<provide a paragraph summarizing the lesion>",
  "differential_diagnosis": "<provide the possible differential diagnoses>"
}"""

ANSWER_TO_COL = {
    "type_of_lesion":                   "type_of_lesion",
    "site":                             "site",
    "count":                            "count",
    "arrangement":                      "arrangement",
    "size":                             "size",
    "color_pattern":                    "color_pattern",
    "border":                           "border",
    "surface_changes":                  "surface_changes",
    "presence_of_exudate_or_discharge": "presence_of_exudate_or_discharge",
    "surrounding_skin_changes":         "surrounding_skin_changes",
    "secondary_changes":                "secondary_changes",
    "pattern_or_shape":                 "pattern_or_shape",
    "additional_notes":                 "additional_notes",
    "overall_description":              "overall_description",
    "differential_diagnosis":           "differential_diagnosis",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

_client: genai.Client | None = None

def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client

def pil_to_part(image: Image.Image) -> types.Part:
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")

def _error_result(msg: str) -> dict:
    return {col: f"ERROR: {msg}" for col in ANSWER_TO_COL}

def call_gemini(images: List[Image.Image]) -> dict:
    parts: List[types.Part] = []
    for img in images:
        if img.width > RESIZE_TO or img.height > RESIZE_TO:
            img = img.resize((RESIZE_TO, RESIZE_TO), Image.Resampling.LANCZOS)
        parts.append(pil_to_part(img))

    client = get_client()

    base_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_MESSAGE,
        response_mime_type="application/json",
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    relaxed_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_MESSAGE,
        response_mime_type="application/json",
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",       threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        ],
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=parts,
                config=base_config,
            )

            candidate    = response.candidates[0] if response.candidates else None
            finish_reason = str(candidate.finish_reason) if candidate else "NO_CANDIDATE"

            if finish_reason not in ("FinishReason.STOP", "STOP", "1"):
                print(f"    [WARN] Blocked (finish_reason={finish_reason}) — retrying with relaxed safety ...")
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=parts,
                    config=relaxed_config,
                )

            text = response.text or ""
            if not text.strip():
                print(f"    [WARN] Empty response (finish_reason={finish_reason})")
                return _error_result("Empty response (safety block)")

            start = text.find("{")
            end   = text.rfind("}")
            if start == -1 or end == -1:
                print(f"    [WARN] No JSON braces in response. Raw: {text[:200]}")
                return _error_result("No JSON in response")

            parsed = json.loads(text[start : end + 1])
            return parsed

        except json.JSONDecodeError as e:
            print(f"    [WARN] JSON parse error attempt {attempt}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)

        except Exception as e:
            err = str(e)
            print(f"    [WARN] API error attempt {attempt}/{MAX_RETRIES}: {err[:120]}")
            if any(k in err for k in ("INVALID_ARGUMENT", "PERMISSION_DENIED", "NOT_FOUND")):
                break
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)

    return _error_result("All retries exhausted")

def dict_to_row(result_dict: dict) -> dict:
    row = {}
    for key, col in ANSWER_TO_COL.items():
        val = result_dict.get(key, "")
        if isinstance(val, list):
            val = " ".join(str(v) for v in val)
        row[col] = val
    return row

def preview(result_dict: dict) -> str:
    dd = result_dict.get("differential_diagnosis") or ""
    if isinstance(dd, list):
        dd = " ".join(str(v) for v in dd)
    return str(dd)[:50].replace("\n", " ")

def _pick_images(folder: str) -> List[str]:
    imgs = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if Path(f).suffix.lower() in IMAGE_EXTS and f.startswith(IMAGE_PREFIX)
    ])
    return imgs[:MAX_IMAGES]

def collect_open_source() -> List[Tuple[str, List[str]]]:
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

def collect_real_patient():
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

def collect_dataset_wise():
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
            entries.append((dname, Path(p).name, p))
    return entries

def run_open_source():
    entries = collect_open_source()
    total   = len(entries)
    print(f"  Open source patients : {total} — regenerating ALL (overwrite mode)")
    if not entries:
        print("  [SKIP] No data found.")
        return

    os.makedirs(OUTPUT_OPEN_SOURCE, exist_ok=True)
    completed  = 0
    start_time = time.time()

    for pid, image_paths in entries:
        time.sleep(SLEEP_BETWEEN)
        images     = [Image.open(p).convert("RGB") for p in image_paths]
        result     = call_gemini(images)
        out_dir    = os.path.join(OUTPUT_OPEN_SOURCE, pid)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "chathistory_annotated_gemini.json"), "w") as f:
            json.dump(result, f, indent=2)
        completed += 1
        rate = completed / (time.time() - start_time)
        print(f"  [{completed}/{total}] ({rate:.2f}/s) ✓ {pid} | {preview(result)}")

def run_real_patient():
    entries = collect_real_patient()
    total   = len(entries)
    print(f"  Real patient visits  : {total} — regenerating ALL (overwrite mode)")
    if not entries:
        print("  [SKIP] No data found.")
        return

    os.makedirs(OUTPUT_REAL_PATIENT, exist_ok=True)
    completed  = 0
    start_time = time.time()

    for pid, cid, vid, image_paths in entries:
        time.sleep(SLEEP_BETWEEN)
        images  = [Image.open(p).convert("RGB") for p in image_paths]
        result  = call_gemini(images)
        out_dir = os.path.join(OUTPUT_REAL_PATIENT, pid, cid, vid)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "chathistory_annotated_gemini.json"), "w") as f:
            json.dump(result, f, indent=2)
        completed += 1
        rate = completed / (time.time() - start_time)
        print(f"  [{completed}/{total}] ({rate:.2f}/s) ✓ {pid}/{cid}/{vid} | {preview(result)}")

def run_dataset_wise():
    print("  Scanning dataset folders...")
    entries = collect_dataset_wise()
    total   = len(entries)
    print(f"  Total images : {total} — regenerating ALL (overwrite mode)")
    if not entries:
        print("  [SKIP] No images found.")
        return

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    rows       = []
    completed  = 0
    start_time = time.time()

    for dname, iname, ipath in entries:
        time.sleep(SLEEP_BETWEEN)
        try:
            image  = Image.open(ipath).convert("RGB")
            result = call_gemini([image])
        except Exception as e:
            print(f"  [ERROR] {dname}/{iname}: {e}")
            result = _error_result(str(e))

        row = {"dataset_name": dname, "image_name": iname, **dict_to_row(result)}
        rows.append(row)
        completed += 1
        rate = completed / (time.time() - start_time)
        print(f"  [{completed}/{total}] ({rate:.2f}/s) ✓ {dname}/{iname} | {preview(result)}")

    pd.DataFrame(rows).to_csv(OUTPUT_CSV, mode="w", header=True, index=False)
    print(f"  ✓ {len(rows)} rows written to {OUTPUT_CSV}")

MODES = {
    "open_source":  ("OPEN SOURCE GENERATION",  run_open_source),
    "real_patient": ("REAL PATIENT GENERATION",  run_real_patient),
    "dataset_wise": ("DATASET-WISE GENERATION",  run_dataset_wise),
}

def main(modes: list):
    print("\n" + "=" * 70)
    print("  GEMINI 2.5 FLASH DESCRIPTOR GENERATION")
    print("=" * 70)
    print(f"  Model  : {GEMINI_MODEL}")
    print(f"  Resize : {RESIZE_TO}px")
    print(f"  Sleep  : {SLEEP_BETWEEN}s between calls")
    print(f"  Modes  : {', '.join(modes)}")
    print("=" * 70)

    for mode in modes:
        label, runner = MODES[mode]
        print(f"\n{'─' * 70}")
        print(f"  [{label}]")
        print(f"{'─' * 70}")
        t0 = time.time()
        runner()
        print(f"  Done in {time.time()-t0:.1f}s")

    print("\n" + "=" * 70)
    print("  ALL GENERATION COMPLETE")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini 2.5 Flash descriptor generation")
    parser.add_argument(
        "--mode",
        choices=list(MODES.keys()),
        default=None,
        help="Which mode to run. Omit to run all three.",
    )
    args = parser.parse_args()
    modes_to_run = [args.mode] if args.mode else list(MODES.keys())
    main(modes_to_run)