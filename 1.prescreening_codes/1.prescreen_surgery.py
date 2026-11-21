import asyncio
import base64
import csv
import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import aiohttp
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

HOST                   = "http://localhost"
PORTS                  = list(range(8000, 8004))
CONCURRENCY_PER_SERVER = 100
N_ENCODERS             = min(32, os.cpu_count() or 8)
MAX_SIDE               = 896
JPEG_QUALITY           = 90
MAX_TOKENS             = 128
RETRIES                = 3
REQUEST_TIMEOUT_S      = 180
FLUSH_EVERY            = 1000
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
CSV_FIELDS = ["image_path", "LABEL", "CONFIDENCE", "REASON"]

VALID_LABELS = ("NON_SURGERY", "SURGERY")
SYSTEM   = "You are a medical image screening expert."
QUESTION = """You are screening dermatology images for active surgical content.
Task: Detect whether an active surgical or procedural scene is present.
REJECT (label as SURGERY):
- Ongoing surgery with instruments actively operating
- Suturing or stitching in progress
- Active operative procedures
- Scalpels or clamps cutting into tissue
ACCEPT (label as NON_SURGERY):
- Normal dermatology lesions
- Healed surgical scars
- Post-surgical skin without any active procedure visible
- Wounds without instruments present
OUTPUT FORMAT (exact, no extra text):
LABEL: <SURGERY or NON_SURGERY>
CONFIDENCE: <0-100>
REASON: <max 50 words>"""

DIVIDER = "=" * 70

def encode_image(path: str) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.BICUBIC)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def normalize_label(v: str) -> str:
    v = v.strip().upper().replace("-", "_").replace(" ", "_")
    if v.startswith("NON_SURGERY") or v.startswith("NONSURGERY"):
        return "NON_SURGERY"
    if v.startswith("SURGERY"):
        return "SURGERY"
    return v

def parse_response(response: str, img_path: str) -> dict:
    result = {"image_path": img_path, "LABEL": "", "CONFIDENCE": "", "REASON": ""}
    for line in response.split("\n"):
        line = line.replace("**", "").strip().lstrip("-*# ").strip()
        up = line.upper()
        if up.startswith("LABEL:"):
            result["LABEL"] = normalize_label(line.split(":", 1)[1])
        elif up.startswith("CONFIDENCE:"):
            result["CONFIDENCE"] = line.split(":", 1)[1].strip().rstrip("%").strip()
        elif up.startswith("REASON:"):
            result["REASON"] = line.split(":", 1)[1].strip()
    return result

def row_needs_reprocess(row: dict) -> bool:
    label  = str(row.get("LABEL", "") or "").strip()
    conf   = str(row.get("CONFIDENCE", "") or "").strip()
    reason = str(row.get("REASON", "") or "").strip()
    if label.lower()  in ("", "nan", "none"): return True
    if conf.lower()   in ("", "nan", "none"): return True
    if reason.lower() in ("", "nan", "none"): return True
    if reason.upper().startswith("ERROR:"):   return True
    return False

def err_row(path: str, msg: str) -> dict:
    msg = " ".join(str(msg).split())[:300]
    return {"image_path": path, "LABEL": "", "CONFIDENCE": "", "REASON": f"ERROR: {msg}"}

def collect_images(root: str) -> list:
    out = []
    for dirpath, _, files in os.walk(root, followlinks=True):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                out.append(os.path.abspath(os.path.join(dirpath, fn)))
    out.sort()
    return out

def read_csv_rows(path: str):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
        clean = (line.replace("\0", "") for line in f)
        return [r for r in csv.DictReader(clean) if (r.get("image_path") or "").strip()]

def load_done(out_csv: str) -> set:
    return {r["image_path"].strip() for r in read_csv_rows(out_csv) if not row_needs_reprocess(r)}

def compact_csv(out_csv: str):
    final = {}
    for r in read_csv_rows(out_csv):
        p = r["image_path"].strip()
        r = {k: (r.get(k) or "").strip() for k in CSV_FIELDS}
        r["image_path"] = p
        if p not in final or row_needs_reprocess(final[p]) or not row_needs_reprocess(r):
            final[p] = r
    tmp = out_csv + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        w.writerows(final.values())
    os.replace(tmp, out_csv)
    bad = sum(row_needs_reprocess(r) for r in final.values())
    surg = sum(r["LABEL"] == "SURGERY" for r in final.values())
    print(f"Compacted: {len(final)} rows | SURGERY={surg} | incomplete/ERROR={bad}")
    if bad:
        print("  -> rerun the same command to retry incomplete rows")

async def discover_endpoints(session):
    eps = []
    for port in PORTS:
        url = f"{HOST}:{port}"
        try:
            async with session.get(f"{url}/v1/models",
                                   timeout=aiohttp.ClientTimeout(total=5)) as r:
                data = await r.json()
                model = data["data"][0]["id"]
                eps.append((url, model))
                print(f"  [UP]   {url}  model={model}")
        except Exception as e:
            print(f"  [DOWN] {url}  ({type(e).__name__})")
    return eps

async def infer_one(path, url, model, session, loop, pool):
    try:
        b64 = await loop.run_in_executor(pool, encode_image, path)
    except Exception as e:
        return err_row(path, f"load: {e}")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": QUESTION},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }

    last_err = "unknown"
    for attempt in range(1, RETRIES + 1):
        try:
            async with session.post(f"{url}/v1/chat/completions", json=payload) as r:
                if r.status == 200:
                    data = await r.json()
                    text = data["choices"][0]["message"]["content"] or ""
                    row = parse_response(text, path)
                    if row_needs_reprocess(row):
                        row["REASON"] = row["REASON"] or f"ERROR: unparsed: {' '.join(text.split())[:200]}"
                    return row
                body = await r.text()
                last_err = f"HTTP {r.status}: {body[:200]}"
                if r.status == 400:
                    break
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        await asyncio.sleep(2 ** attempt)
    return err_row(path, last_err)

async def worker(url, model, session, q, out_q, loop, pool):
    while True:
        path = await q.get()
        if path is None:
            return
        await out_q.put(await infer_one(path, url, model, session, loop, pool))

async def writer(out_csv, out_q, total):
    new_file = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    f = open(out_csv, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_MINIMAL)
    if new_file:
        w.writeheader()
    n = errs = surg = 0
    t0 = time.time()
    try:
        while True:
            row = await out_q.get()
            if row is None:
                break
            w.writerow(row)
            n += 1
            errs += row_needs_reprocess(row)
            surg += row["LABEL"] == "SURGERY"
            if n % FLUSH_EVERY == 0 or n == total:
                f.flush()
                el = time.time() - t0
                rate = n / el if el else 0
                eta = (total - n) / rate / 3600 if rate else 0
                print(f"  {n}/{total} ({100*n/total:.2f}%) | {rate:.1f} img/s | "
                      f"ETA {eta:.2f} h | SURGERY={surg} ERR={errs}", flush=True)
    finally:
        f.flush()
        f.close()

async def amain(img_dir, out_csv):
    print(DIVIDER)
    print("SURGERY PRESCREEN — vLLM MULTI-SERVER")
    print(DIVIDER)

    print(f"Scanning {img_dir} ...")
    paths = collect_images(img_dir)
    done = load_done(out_csv)
    todo = [p for p in paths if p not in done]
    print(f"Images found: {len(paths)} | already done: {len(done)} | to process: {len(todo)}")
    if not todo:
        compact_csv(out_csv)
        return

    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        print("Discovering vLLM servers:")
        eps = await discover_endpoints(session)
        if not eps:
            print("No live servers. Exiting.")
            sys.exit(1)
        if len({m for _, m in eps}) > 1:
            print("  [WARN] servers report different model names — results will mix models!")

        n_workers = len(eps) * CONCURRENCY_PER_SERVER
        print(f"Workers: {n_workers} ({CONCURRENCY_PER_SERVER}/server) | encoders: {N_ENCODERS}")
        print(DIVIDER)

        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor(max_workers=N_ENCODERS) as pool:
            q = asyncio.Queue(maxsize=n_workers * 4)
            out_q = asyncio.Queue(maxsize=n_workers * 8)
            wtask = asyncio.create_task(writer(out_csv, out_q, len(todo)))
            workers = [asyncio.create_task(worker(url, model, session, q, out_q, loop, pool))
                       for url, model in eps for _ in range(CONCURRENCY_PER_SERVER)]

            for p in todo:
                await q.put(p)
            for _ in workers:
                await q.put(None)

            await asyncio.gather(*workers)
            await out_q.put(None)
            await wtask

    print(DIVIDER)
    compact_csv(out_csv)
    print("DONE")

def main():
    if len(sys.argv) != 3:
        print("Usage: python prescreen_surgery_vllm.py IMAGE_FOLDER_PATH OUTPUT_CSV_PATH")
        sys.exit(1)
    img_dir, out_csv = sys.argv[1], sys.argv[2]
    if not os.path.isdir(img_dir):
        print(f"Not a directory: {img_dir}")
        sys.exit(1)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    try:
        asyncio.run(amain(img_dir, out_csv))
    except KeyboardInterrupt:
        print("\nInterrupted — progress saved. Rerun the same command to resume.")

if __name__ == "__main__":
    main()