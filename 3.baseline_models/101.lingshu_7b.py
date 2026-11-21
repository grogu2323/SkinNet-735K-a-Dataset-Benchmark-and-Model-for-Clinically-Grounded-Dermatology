import os
import json
import asyncio
import base64
import io
import argparse
import time
from pathlib import Path

import pandas as pd
from PIL import Image
from openai import AsyncOpenAI

OPEN_SOURCE_DATA     = "/home/pratyushprasad/ACL/DATA/TRAIN_TEST_DATA/TEST_OPEN_SOURCE_FULL_DATA"
REAL_PATIENT_DATA    = "/home/pratyushprasad/ACL/DATA/TRAIN_TEST_DATA/TEST_REAL_PATIENT_FULL_DATA"
OUTPUT_OPEN_SOURCE   = "/home/pratyushprasad/ACL/DATA/GENERATED_DESCRIPTORS/TEST_OPEN_SOURCE_GEN_DATA"
OUTPUT_REAL_PATIENT  = "/home/pratyushprasad/ACL/DATA/GENERATED_DESCRIPTORS/TEST_REAL_PATIENT_GEN_DATA"

DATASET_FOLDERS = {
    "yuhos":           "/home/pratyushprasad/ACL/DATA/DATASET_WISE_TEST_DATA/yuhos",
    "scin":            "/home/pratyushprasad/ACL/DATA/DATASET_WISE_TEST_DATA/scin",
    "mm_skin_qa":      "/home/pratyushprasad/ACL/DATA/DATASET_WISE_TEST_DATA/mm_skin_qa",
    "fitzpatrick_17k": "/home/pratyushprasad/ACL/DATA/DATASET_WISE_TEST_DATA/fitzpatrick_17k",
    "biomedica":       "/home/pratyushprasad/ACL/DATA/DATASET_WISE_TEST_DATA/biomedica",
}
OUTPUT_CSV = "/home/pratyushprasad/ACL/DATA/DATASET_WISE_TEST_DATA/generated_descriptors/lingshu_7b.csv"

VLLM_BASE_URL    = "http://localhost:8010/v1"
MODEL_NAME       = "lingshu_7b"
API_KEY          = "EMPTY"
CONCURRENCY      = 32
REQUEST_TIMEOUT  = 600.0
MAX_IMAGES       = 6
IMAGE_PREFIX     = ""

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

MAX_TOKENS = [50] * 12 + [250, 512, 512]

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

def image_to_data_url(path: str) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = im.resize((256, 256), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"

def make_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=VLLM_BASE_URL,
        api_key=API_KEY,
        timeout=REQUEST_TIMEOUT,
        max_retries=2,
    )

def _answers_to_dict(answers: list) -> dict:
    return {ANSWER_TO_COL[i]: answers[i] for i in range(len(QUESTIONS))}

async def query_model(client: AsyncOpenAI, image_paths: list, sem: asyncio.Semaphore) -> dict:
    async with sem:
        try:
            image_urls = [image_to_data_url(p) for p in image_paths]
        except Exception as e:
            return {col: f"IMAGE_ERROR: {e}" for col in ANSWER_TO_COL.values()}

        answers = []
        for q_idx, question in enumerate(QUESTIONS):
            try:
                content = [
                    {"type": "image_url", "image_url": {"url": u}}
                    for u in image_urls
                ]
                content.append({"type": "text", "text": question})

                resp = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user",   "content": content},
                    ],
                    max_tokens=MAX_TOKENS[q_idx],
                    temperature=0.0,
                )
                answer = (resp.choices[0].message.content or "").strip()
            except Exception as e:
                answer = f"API_ERROR: {e}"
            answers.append(answer)

        return _answers_to_dict(answers)

def _pick_images(folder: str) -> list:
    images = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if Path(f).suffix.lower() in IMAGE_EXTS and f.startswith(IMAGE_PREFIX)
    ])
    return images[:MAX_IMAGES]

def collect_open_source() -> list:
    visits = []
    for pid in sorted(os.listdir(OPEN_SOURCE_DATA)):
        ppath = os.path.join(OPEN_SOURCE_DATA, pid)
        if not os.path.isdir(ppath):
            continue
        if not os.path.exists(os.path.join(ppath, "chathistory_annotated.json")):
            continue
        images = _pick_images(ppath)
        if images:
            visits.append((pid, images))
    return visits

def collect_real_patient() -> list:
    visits = []
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
                images = _pick_images(vpath)
                if images:
                    visits.append((pid, cid, vid, images))
    return visits

def collect_dataset_wise() -> list:
    all_images = []
    for dname, folder in DATASET_FOLDERS.items():
        if not os.path.isdir(folder):
            print(f"  [WARN] {dname} not found: {folder}")
            continue
        images = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if Path(f).suffix.lower() in IMAGE_EXTS
        ])
        print(f"  {dname}: {len(images)} images")
        for p in images:
            all_images.append((dname, Path(p).name, p))
    return all_images

async def run_open_source():
    visits = collect_open_source()
    total = len(visits)
    print(f"  Open source patients : {total}")
    if not visits:
        print("  [SKIP] No open source data found.")
        return

    os.makedirs(OUTPUT_OPEN_SOURCE, exist_ok=True)
    client = make_client()
    sem    = asyncio.Semaphore(CONCURRENCY)

    async def _task(pid, imgs):
        result = await query_model(client, imgs, sem)
        return result, pid

    tasks      = [asyncio.create_task(_task(pid, imgs)) for pid, imgs in visits]
    completed  = 0
    start_time = time.time()

    for coro in asyncio.as_completed(tasks):
        try:
            result, pid = await coro
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        out_dir = os.path.join(OUTPUT_OPEN_SOURCE, pid)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "chathistory_annotated_lingshu_7b.json"), "w") as f:
            json.dump(result, f, indent=2)

        completed += 1
        elapsed = time.time() - start_time
        rate    = completed / elapsed if elapsed > 0 else 0.0
        dd      = (result.get("differential_diagnosis") or "")[:50].replace("\n", " ")
        print(f"  [{completed}/{total}] ({rate:.2f}/s) ✓ {pid} | {dd}")

    await client.close()

async def run_real_patient():
    visits = collect_real_patient()
    total  = len(visits)
    print(f"  Real patient visits  : {total}")
    if not visits:
        print("  [SKIP] No real patient data found.")
        return

    os.makedirs(OUTPUT_REAL_PATIENT, exist_ok=True)
    client = make_client()
    sem    = asyncio.Semaphore(CONCURRENCY)

    async def _task(pid, cid, vid, imgs):
        result = await query_model(client, imgs, sem)
        return result, pid, cid, vid

    tasks      = [asyncio.create_task(_task(*v)) for v in visits]
    completed  = 0
    start_time = time.time()

    for coro in asyncio.as_completed(tasks):
        try:
            result, pid, cid, vid = await coro
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        out_dir = os.path.join(OUTPUT_REAL_PATIENT, pid, cid, vid)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "chathistory_annotated_lingshu_7b.json"), "w") as f:
            json.dump(result, f, indent=2)

        completed += 1
        elapsed = time.time() - start_time
        rate    = completed / elapsed if elapsed > 0 else 0.0
        dd      = (result.get("differential_diagnosis") or "")[:50].replace("\n", " ")
        print(f"  [{completed}/{total}] ({rate:.2f}/s) ✓ {pid}/{cid}/{vid} | {dd}")

    await client.close()

async def run_dataset_wise():
    print("  Scanning dataset folders...")
    all_images = collect_dataset_wise()
    total      = len(all_images)
    print(f"  Total images : {total}")
    if not all_images:
        print("  [SKIP] No images found.")
        return

    done: set = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            df_done = pd.read_csv(OUTPUT_CSV, usecols=["dataset_name", "image_name"])
            done    = set(zip(df_done["dataset_name"], df_done["image_name"]))
            print(f"  Resuming    : {len(done)} already done, {total - len(done)} pending")
        except Exception:
            pass

    pending = [(d, n, p) for d, n, p in all_images if (d, n) not in done]
    if not pending:
        print("  [SKIP] All images already processed.")
        return

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    client     = make_client()
    sem        = asyncio.Semaphore(CONCURRENCY)
    file_exists = os.path.exists(OUTPUT_CSV)

    async def _task(dname, iname, ipath):
        result = await query_model(client, [ipath], sem)
        return result, dname, iname

    tasks      = [asyncio.create_task(_task(d, n, p)) for d, n, p in pending]
    completed  = 0
    start_time = time.time()

    for coro in asyncio.as_completed(tasks):
        try:
            result, dname, iname = await coro
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        row = {"dataset_name": dname, "image_name": iname, **result}
        pd.DataFrame([row]).to_csv(
            OUTPUT_CSV,
            mode="a",
            header=not file_exists,
            index=False,
        )
        file_exists = True

        completed += 1
        elapsed = time.time() - start_time
        rate    = completed / elapsed if elapsed > 0 else 0.0
        dd      = (result.get("differential_diagnosis") or "")[:40].replace("\n", " ")
        print(f"  [{completed}/{len(pending)}] ({rate:.2f}/s) ✓ {dname}/{iname} | {dd}")

    await client.close()
    print(f"\n  ✓ CSV saved to: {OUTPUT_CSV}")

MODES = {
    "open_source":  ("OPEN SOURCE GENERATION",  run_open_source),
    "real_patient": ("REAL PATIENT GENERATION", run_real_patient),
    "dataset_wise": ("DATASET-WISE GENERATION", run_dataset_wise),
}

async def main(modes: list):
    print("\n" + "=" * 70)
    print("  LINGSHU-7B DESCRIPTOR GENERATION")
    print("=" * 70)
    print(f"  Server      : {VLLM_BASE_URL}")
    print(f"  Model       : {MODEL_NAME}")
    print(f"  Concurrency : {CONCURRENCY}")
    print(f"  Modes       : {', '.join(modes)}")
    print("=" * 70)

    for mode in modes:
        label, runner = MODES[mode]
        print(f"\n{'─' * 70}")
        print(f"  [{label}]")
        print(f"{'─' * 70}")
        t0 = time.time()
        await runner()
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

    print("\n" + "=" * 70)
    print("  ALL GENERATION COMPLETE")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lingshu-7B descriptor generation")
    parser.add_argument(
        "--mode",
        choices=list(MODES.keys()),
        default=None,
        help="Which mode to run. Omit to run all three.",
    )
    args = parser.parse_args()

    modes_to_run = [args.mode] if args.mode else list(MODES.keys())
    asyncio.run(main(modes_to_run))