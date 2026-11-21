import os
import json
import asyncio
import base64
import io
import time
from pathlib import Path
from PIL import Image
from openai import AsyncOpenAI
import pandas as pd

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
OUTPUT_DATASET_WISE_DIR = "/home/pratyushprasad/ACL/DATA/DATASET_WISE_TEST_DATA/generated_descriptors"

# ─────────────────────────────────────────────────────────────────────────────
# SERVER / MODEL CONFIG
# ─────────────────────────────────────────────────────────────────────────────

VLLM_BASE_URL   = "http://localhost:8001/v1"
MODEL_NAME      = "ours_27b"
API_KEY         = "EMPTY"
CONCURRENCY     = 64
REQUEST_TIMEOUT = 600.0
MAX_IMAGES      = 6
IMAGE_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

TEMPERATURES = [0.0, 0.1, 0.2, 0.3, 0.4]
TOP_PS       = [1.0, 0.9, 0.8, 0.7, 0.6]
COMBINATIONS = [(t, p) for t in TEMPERATURES for p in TOP_PS]

# ─────────────────────────────────────────────────────────────────────────────
# COT QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_ORDER = [
    'type_of_lesion',
    'site',
    'count',
    'arrangement',
    'size',
    'color_pattern',
    'border',
    'surface_changes',
    'presence_of_exudate_or_discharge',
    'surrounding_skin_changes',
    'secondary_changes',
    'pattern_or_shape',
    'additional_notes',
    'overall_description',
    'label',
]

QUESTIONS = {
    'type_of_lesion': "You are an Expert Dermatologist. What type of lesion is shown in this dermatological image? Identify the primary lesion type based on its morphology, such as macule, papule, plaque, nodule, vesicle, pustule, cyst, or ulcer. Answer in 10 words maximum.",
    'site': "You are an expert Dermatologist. Identify the anatomical location of the lesion and provide two levels of localization. Major site refers to the broad body region. Minor site refers to the more specific anatomical location within that region. Return both the major and minor site clearly.",
    'count': "You are an Expert Dermatologist. How many lesions are present in this image? Answer in 10 words maximum.",
    'arrangement': "You are an Expert Dermatologist. How are the lesions arranged in this image? Describe the overall lesion distribution/arrangement. Choose the most appropriate term(s), such as solitary, scattered, clustered, grouped, linear, annular, diffuse. If multiple patterns are present, mention the dominant pattern first. Answer in 10 words maximum.",
    'size': "You are an Expert Dermatologist. What is the size of the lesion(s) in this image? Answer in 10 words maximum.",
    'color_pattern': "You are an Expert Dermatologist. Describe the color pattern of the lesion and the surrounding skin. Identify the main colors visible within the lesion and any distinct color changes immediately surrounding it. Mention multiple colors when present. Answer in 32 words maximum.",
    'border': "You are an Expert Dermatologist. How well-defined is the border of this lesion? Assess whether the entire lesion has a clearly visible, distinct boundary separating it from the surrounding skin. Describe the border as well-defined, poorly defined, or partially defined. Consider the visibility of the border around the entire lesion, not its shape or regularity. Answer in 10 words maximum",
    'surface_changes': "You are an Expert Dermatologist. Describe the surface changes of the lesion relative to the surrounding skin. Determine whether the lesion is elevated, depressed, flat, or has another visible surface change. Focus on the physical surface appearance rather than color. Answer in 10 words maximum.",
    'presence_of_exudate_or_discharge': "You are an Expert Dermatologist. Assess whether the lesion shows any visible exudate or discharge. Look for fluid, pus, serous ooze, blood, crusted discharge or any active secretion on the lesion surface. If none is visible, state 'No exudate or discharge.' Answer in 10 words maximum.",
    'surrounding_skin_changes': "You are an Expert Dermatologist. Describe any visible changes in the skin surrounding the lesion. Assess for redness, discoloration, swelling, inflammation, thickening, scaling, pigmentation changes, or other abnormalities around the lesion. Focus only on the surrounding skin, not the lesion itself. If no changes are visible, state that the surrounding skin appears normal. Answer in 64 words maximum.",
    'secondary_changes': "You are an Expert Dermatologist. Identify any secondary changes associated with the lesion, such as excoriation, erosion, ulceration, crusting, fissuring, scarring, atrophy, lichenification, or bleeding. Describe only changes that are visibly present and distinguish them from the primary lesion. If no secondary changes are visible, state that none are present. Answer in 64 words maximum.",
    'pattern_or_shape': "You are an Expert Dermatologist. Describe the overall pattern and shape of the lesion. Assess its geometric or morphological form, such as round, oval, irregular, annular, linear, serpiginous, targetoid, or plaque-like. Describe the shape that is visibly apparent in the image without inferring features that cannot be seen. Answer in 64 words maximum.",
    'label': "You are an Expert Dermatologist. Based on the visible clinical features of the lesion, provide a differential diagnosis with a maximum of 5 possible diagnoses. Rank the diagnoses from most to least likely. Consider the lesion's morphology, color, distribution, border, surface characteristics, and anatomical location. Do not include diagnoses that are inconsistent with the visible findings. Answer in 256 words maximum.",
    'additional_notes': "You are an Expert Dermatologist. Based on the visible clinical features and the provided diagnosis, describe any additional relevant findings you can observe in the image that have not been captured by the other clinical descriptors. Include notable morphological, anatomical, or contextual features that may support or refine the diagnosis. Do not repeat information unnecessarily. Answer in 512 words maximum.",
    'overall_description': "You are an Expert Dermatologist. Describe the overall clinical appearance of the lesion(s) in detail. Integrate the lesion morphology, color, border, surface, distribution, anatomical location, and other relevant visible features, along with their diagnostic significance where appropriate. Provide a coherent clinical description rather than listing individual features separately. Answer in 512 words maximum.",
}

MAX_TOKENS = {
    'type_of_lesion':                   20,
    'site':                             40,
    'count':                            20,
    'arrangement':                      20,
    'size':                             20,
    'color_pattern':                    60,
    'border':                           20,
    'surface_changes':                  20,
    'presence_of_exudate_or_discharge': 20,
    'surrounding_skin_changes':         100,
    'secondary_changes':                100,
    'pattern_or_shape':                 100,
    'additional_notes':                 600,
    'overall_description':              600,
    'label':                            300,
}

SYSTEM_MESSAGE = (
    "You are an Expert Dermatologist analyzing medical images. "
    "Give me precise answers only. Do not give explanations. "
    "I donot want responses outside of the asked questions"
)

# ─────────────────────────────────────────────────────────────────────────────
# RESUME HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def os_already_done(pid: str, combo_tag: str) -> bool:
    fpath = os.path.join(OUTPUT_OPEN_SOURCE, pid, f"chathistory_{combo_tag}.json")
    if not os.path.exists(fpath):
        return False
    try:
        with open(fpath) as f:
            data = json.load(f)
        return not any("ERROR" in str(v) for v in data.values())
    except Exception:
        return False


def rp_already_done(pid: str, cid: str, vid: str, combo_tag: str) -> bool:
    fpath = os.path.join(OUTPUT_REAL_PATIENT, pid, cid, vid, f"chathistory_{combo_tag}.json")
    if not os.path.exists(fpath):
        return False
    try:
        with open(fpath) as f:
            data = json.load(f)
        return not any("ERROR" in str(v) for v in data.values())
    except Exception:
        return False


def dw_already_done(combo_tag: str) -> set:
    csv_path = os.path.join(OUTPUT_DATASET_WISE_DIR, f"{combo_tag}.csv")
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path)
        done = set()
        for _, row in df.iterrows():
            if not any("ERROR" in str(v) for v in row.values):
                done.add((str(row["dataset_name"]), str(row["image_name"])))
        return done
    except Exception:
        return set()

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_IMAGE_CACHE: dict[str, str] = {}
_IMAGE_LOCK = asyncio.Lock()  # prevents duplicate encoding of same image

async def image_to_data_url_async(path: str) -> str:
    """Encode image to base64 data URL, cached in memory (async-safe)."""
    if path in _IMAGE_CACHE:
        return _IMAGE_CACHE[path]
    async with _IMAGE_LOCK:
        if path in _IMAGE_CACHE:   # double-check after acquiring lock
            return _IMAGE_CACHE[path]
        loop = asyncio.get_event_loop()
        url = await loop.run_in_executor(None, _encode_image, path)
        _IMAGE_CACHE[path] = url
        return url


def _encode_image(path: str) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = im.resize((896, 896), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _pick_images(folder: str) -> list:
    imgs = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if Path(f).suffix.lower() in IMAGE_EXTS
    ])
    return imgs[:MAX_IMAGES]


def make_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=VLLM_BASE_URL,
        api_key=API_KEY,
        timeout=REQUEST_TIMEOUT,
        max_retries=2,
    )

# ─────────────────────────────────────────────────────────────────────────────
# COT INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

async def query_cot(
    client: AsyncOpenAI,
    image_urls: list,
    sem: asyncio.Semaphore,
    temperature: float,
    top_p: float,
) -> dict:
    async with sem:
        results = {}
        history = []

        for feat in FEATURE_ORDER:
            question = QUESTIONS[feat]

            user_content = (
                [{"type": "image_url", "image_url": {"url": u}} for u in image_urls]
                + [{"type": "text", "text": question}]
            )

            messages = (
                [{"role": "system", "content": SYSTEM_MESSAGE}]
                + history
                + [{"role": "user", "content": user_content}]
            )

            try:
                resp = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=MAX_TOKENS[feat],
                    temperature=temperature,
                    top_p=top_p,
                )
                answer = (resp.choices[0].message.content or "").strip()
            except Exception as e:
                answer = f"API_ERROR: {e}"

            results[feat] = answer
            history.append({"role": "user",     "content": user_content})
            history.append({"role": "assistant", "content": answer})

        return results

# ─────────────────────────────────────────────────────────────────────────────
# COLLECTORS
# ─────────────────────────────────────────────────────────────────────────────

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


def collect_dataset_wise() -> list:
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

# ─────────────────────────────────────────────────────────────────────────────
# WRITE HELPERS (thread-safe locks per output file)
# ─────────────────────────────────────────────────────────────────────────────

_csv_locks: dict[str, asyncio.Lock] = {}

def _get_csv_lock(combo_tag: str) -> asyncio.Lock:
    if combo_tag not in _csv_locks:
        _csv_locks[combo_tag] = asyncio.Lock()
    return _csv_locks[combo_tag]


async def flush_dataset_wise(combo_tag: str, row: dict):
    async with _get_csv_lock(combo_tag):
        os.makedirs(OUTPUT_DATASET_WISE_DIR, exist_ok=True)
        csv_path = os.path.join(OUTPUT_DATASET_WISE_DIR, f"{combo_tag}.csv")
        df_new = pd.DataFrame([row])
        if os.path.exists(csv_path):
            df_new.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df_new.to_csv(csv_path, mode="w", header=True, index=False)

# ─────────────────────────────────────────────────────────────────────────────
# PER-ENTRY TASKS  (one coroutine = one image × one combo)
# ─────────────────────────────────────────────────────────────────────────────

async def process_os_entry(client, pid, img_paths, sem, completed, total):
    pending_combos = [
        (t, p) for t, p in COMBINATIONS
        if not os_already_done(pid, f"OURS_27B_{p}_{t}")
    ]
    if not pending_combos:
        async with completed.get_lock():
            completed.value += 1
        return

    try:
        image_urls = [await image_to_data_url_async(p) for p in img_paths]
    except Exception as e:
        print(f"    [IMAGE_ERROR] {pid}: {e}")
        async with completed.get_lock():
            completed.value += 1
        return

    async def _one_combo(temperature, top_p):
        result = await query_cot(client, image_urls, sem, temperature, top_p)
        combo_tag = f"OURS_27B_{top_p}_{temperature}"
        out_dir = os.path.join(OUTPUT_OPEN_SOURCE, pid)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"chathistory_{combo_tag}.json"), "w") as f:
            json.dump(result, f, indent=2)

    await asyncio.gather(*[_one_combo(t, p) for t, p in pending_combos])

    async with completed.get_lock():
        completed.value += 1
        print(f"    [{completed.value}/{total}] {pid} — {len(pending_combos)} combos done")


async def process_rp_entry(client, pid, cid, vid, img_paths, sem, completed, total):
    pending_combos = [
        (t, p) for t, p in COMBINATIONS
        if not rp_already_done(pid, cid, vid, f"OURS_27B_{p}_{t}")
    ]
    if not pending_combos:
        async with completed.get_lock():
            completed.value += 1
        return

    try:
        image_urls = [await image_to_data_url_async(p) for p in img_paths]
    except Exception as e:
        print(f"    [IMAGE_ERROR] {pid}/{cid}/{vid}: {e}")
        async with completed.get_lock():
            completed.value += 1
        return

    async def _one_combo(temperature, top_p):
        result = await query_cot(client, image_urls, sem, temperature, top_p)
        combo_tag = f"OURS_27B_{top_p}_{temperature}"
        out_dir = os.path.join(OUTPUT_REAL_PATIENT, pid, cid, vid)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"chathistory_{combo_tag}.json"), "w") as f:
            json.dump(result, f, indent=2)

    await asyncio.gather(*[_one_combo(t, p) for t, p in pending_combos])

    async with completed.get_lock():
        completed.value += 1
        print(f"    [{completed.value}/{total}] {pid}/{cid}/{vid} — {len(pending_combos)} combos done")


async def process_dw_entry(client, dname, iname, ipath, sem, done_sets, completed, total):
    pending_combos = [
        (t, p) for t, p in COMBINATIONS
        if (dname, iname) not in done_sets[f"OURS_27B_{p}_{t}"]
    ]
    if not pending_combos:
        async with completed.get_lock():
            completed.value += 1
        return

    try:
        image_urls = [await image_to_data_url_async(ipath)]
    except Exception as e:
        print(f"    [IMAGE_ERROR] {dname}/{iname}: {e}")
        async with completed.get_lock():
            completed.value += 1
        return

    async def _one_combo(temperature, top_p):
        result = await query_cot(client, image_urls, sem, temperature, top_p)
        combo_tag = f"OURS_27B_{top_p}_{temperature}"
        await flush_dataset_wise(combo_tag, {"dataset_name": dname, "image_name": iname, **result})

    await asyncio.gather(*[_one_combo(t, p) for t, p in pending_combos])

    async with completed.get_lock():
        completed.value += 1
        if completed.value % 10 == 0 or completed.value == total:
            print(f"    [{completed.value}/{total}] dataset_wise progress")

# ─────────────────────────────────────────────────────────────────────────────
# SIMPLE ASYNC COUNTER WITH LOCK
# ─────────────────────────────────────────────────────────────────────────────

class Counter:
    def __init__(self):
        self.value = 0
        self._lock = asyncio.Lock()

    def get_lock(self):
        return self._lock

# ─────────────────────────────────────────────────────────────────────────────
# RUNNERS  — all images fired concurrently, semaphore throttles GPU load
# ─────────────────────────────────────────────────────────────────────────────

async def run_open_source(client, entries, sem):
    total = len(entries)
    print(f"\n  open_source : {total} images × up to {len(COMBINATIONS)} combos  [ALL PARALLEL]")
    completed = Counter()
    await asyncio.gather(*[
        process_os_entry(client, pid, img_paths, sem, completed, total)
        for pid, img_paths in entries
    ])


async def run_real_patient(client, entries, sem):
    total = len(entries)
    print(f"\n  real_patient : {total} images × up to {len(COMBINATIONS)} combos  [ALL PARALLEL]")
    completed = Counter()
    await asyncio.gather(*[
        process_rp_entry(client, pid, cid, vid, img_paths, sem, completed, total)
        for pid, cid, vid, img_paths in entries
    ])


async def run_dataset_wise(client, entries, sem):
    total = len(entries)
    print(f"\n  dataset_wise : {total} images × up to {len(COMBINATIONS)} combos  [ALL PARALLEL]")
    done_sets = {
        f"OURS_27B_{p}_{t}": dw_already_done(f"OURS_27B_{p}_{t}")
        for t, p in COMBINATIONS
    }
    completed = Counter()
    await asyncio.gather(*[
        process_dw_entry(client, dname, iname, ipath, sem, done_sets, completed, total)
        for dname, iname, ipath in entries
    ])

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    print("\n" + "=" * 70)
    print("  OURS_27B  —  COT INFERENCE  —  25 COMBINATIONS  (FULLY PARALLEL)")
    print("=" * 70)
    print(f"  Server      : {VLLM_BASE_URL}")
    print(f"  Model       : {MODEL_NAME}")
    print(f"  Concurrency : {CONCURRENCY}")
    print(f"  Combinations: {len(COMBINATIONS)}")
    print("=" * 70)

    print("\nCollecting entries...")
    os_entries = collect_open_source()
    rp_entries = collect_real_patient()
    print("  Scanning dataset folders...")
    dw_entries = collect_dataset_wise()
    print(f"  open_source  : {len(os_entries)}")
    print(f"  real_patient : {len(rp_entries)}")
    print(f"  dataset_wise : {len(dw_entries)}")

    client = make_client()
    sem    = asyncio.Semaphore(CONCURRENCY)

    t0 = time.time()
    await run_open_source(client, os_entries, sem)
    await run_real_patient(client, rp_entries, sem)
    await run_dataset_wise(client, dw_entries, sem)
    await client.close()

    print(f"\n  Total time: {time.time() - t0:.1f}s")
    print("=" * 70)
    print("  ALL DONE")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    asyncio.run(main())