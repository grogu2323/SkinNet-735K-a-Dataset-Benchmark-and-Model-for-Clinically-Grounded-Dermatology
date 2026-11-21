import base64, io
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import torch
import uvicorn
from PIL import Image
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

QWEN_MODEL_PATH = "/home/pratyushprasad/ACL/MODELS/Qwen2-VL-72B-Instruct"
QWEN_PORT       = 8001

def make_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

def decode_image(data_url: str) -> Image.Image:
    header, b64 = data_url.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

class ContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[dict] = None

class Message(BaseModel):
    role: str
    content: str | list[ContentItem]

class ChatRequest(BaseModel):
    model: str = "qwen_72b"
    messages: list[Message]
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0

def run_inference(model, processor, req: ChatRequest) -> str:
    images = []
    conversation = []

    for msg in req.messages:
        if isinstance(msg.content, str):
            conversation.append({"role": msg.role, "content": msg.content})
            continue

        hf_content = []
        for item in msg.content:
            if item.type == "image_url" and item.image_url:
                images.append(decode_image(item.image_url["url"]))
                hf_content.append({"type": "image"})
            elif item.type == "text" and item.text:
                hf_content.append({"type": "text", "text": item.text})
        conversation.append({"role": msg.role, "content": hf_content})

    prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = processor(
        images=images if images else None,
        text=prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature if req.temperature > 0 else None,
            top_p=req.top_p if req.temperature > 0 else None,
            do_sample=req.temperature > 0,
            repetition_penalty=1.15,
        )

    generated = out[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()

def make_response(answer: str, model_name: str) -> dict:
    return {
        "id": "chatcmpl-bnb",
        "object": "chat.completion",
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

qwen_model     = None
qwen_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global qwen_model, qwen_processor

    print("\n[Qwen] Loading Qwen2-VL-72B-Instruct (4-bit nf4)...")
    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_PATH,
        quantization_config=make_bnb_config(),
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    qwen_processor = AutoProcessor.from_pretrained(
        QWEN_MODEL_PATH,
        trust_remote_code=True,
    )

    if not hasattr(qwen_processor, "tokenizer"):
        qwen_processor.tokenizer = AutoTokenizer.from_pretrained(
            QWEN_MODEL_PATH,
            trust_remote_code=True,
        )

    print(f"[Qwen] Model loaded!")
    print(f"[Qwen] Chat template present: "
          f"{qwen_processor.tokenizer.chat_template is not None}\n")
    yield

    print("\n[Qwen] Shutting down, freeing GPU memory...")
    del qwen_model
    del qwen_processor
    torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    try:
        answer = run_inference(qwen_model, qwen_processor, req)
        return make_response(answer, "qwen_72b")
    except Exception as e:
        return {
            "error": str(e),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": f"SERVER_ERROR: {e}"},
                "finish_reason": "error",
            }]
        }

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": "qwen_72b", "object": "model"}]}

@app.get("/health")
async def health():
    return {"status": "ok", "model": "qwen_72b", "port": QWEN_PORT}

if __name__ == "__main__":
    print("=" * 60)
    print("  Qwen2-VL-72B-Instruct BnB Server")
    print(f"  Port  : {QWEN_PORT}")
    print(f"  Model : {QWEN_MODEL_PATH}")
    print(f"  Quant : 4-bit nf4 (BitsAndBytes)")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=QWEN_PORT)