import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1" 
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import ast
import json
import math
import torch
from PIL import Image
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from transformers.trainer_utils import get_last_checkpoint

MODEL_PATH   = "/home/aidelab/24m2572/ACL/MODELS/BASE_4B"
OPEN_DIR     = "/home/aidelab/24m2572/ACL/DATA/TRAIN_TEST_DATA/TRAIN_OPEN_SOURCE_FULL_DATA"
REAL_DIR     = "/home/aidelab/24m2572/ACL/DATA/TRAIN_TEST_DATA/TRAIN_REAL_PATIENT_FULL_DATA"
OUTPUT_DIR   = "/home/aidelab/24m2572/ACL/FINETUNED/COT_4B"
LOGS_DIR     = os.path.join(OUTPUT_DIR, "logs")

IMG_SIZE          = 896
MAX_IMAGES        = 6
MIN_IMAGES        = 1
VAL_SIZE          = 0.15
SEED              = 64
LORA_RANK         = 16

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
    'label': "You are an Expert Dermatologist. Based on the visible clinical features of the lesion, provide a differential diagnosis with a maximum of 5 possible diagnoses. Rank the diagnoses from most to least likely. Consider the lesion’s morphology, color, distribution, border, surface characteristics, and anatomical location. Do not include diagnoses that are inconsistent with the visible findings. Answer in 256 words maximum.",
    'additional_notes': "You are an Expert Dermatologist. Based on the visible clinical features and the provided diagnosis, describe any additional relevant findings you can observe in the image that have not been captured by the other clinical descriptors. Include notable morphological, anatomical, or contextual features that may support or refine the diagnosis. Do not repeat information unnecessarily. Answer in 512 words maximum.",
    'overall_description': "You are an Expert Dermatologist. Describe the overall clinical appearance of the lesion(s) in detail. Integrate the lesion morphology, color, border, surface, distribution, anatomical location, and other relevant visible features, along with their diagnostic significance where appropriate. Provide a coherent clinical description rather than listing individual features separately. Answer in 512 words maximum.",
}

IMG_EXT = (".png", ".jpg", ".jpeg")

def clean_label(raw) -> str:
    if raw in (None, ""):
        return "Not specified"
    s = str(raw).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return ", ".join(str(x) for x in parsed)
        except (ValueError, SyntaxError):
            pass
    return s

def get_feature(ann: dict, feat: str) -> str:
    if feat == 'label':
        raw = ann.get('label')
        if raw in (None, ""):
            raw = ann.get('differential_diagnosis', '')
        return clean_label(raw)
    raw = ann.get(feat)
    return str(raw) if raw not in (None, "") else "Not specified"

def build_messages(ann: dict, n_images: int) -> List[Dict[str, Any]]:
    messages = [{
        "role": "system",
        "content": [{"type": "text", "text": "You are an Expert Dermatologist analyzing medical images. Give me precise answers only. Do not give explanations. I donot want responses outside of the asked questions"}],
    }]
    for i, feat in enumerate(FEATURE_ORDER):
        q = QUESTIONS[feat]
        a = get_feature(ann, feat)
        if i == 0:
            user_content = [{"type": "image"} for _ in range(n_images)] + [{"type": "text", "text": q}]
        else:
            user_content = [{"type": "text", "text": q}]
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": a}]})
    return messages

def chunk_images(paths: List[str]) -> List[List[str]]:
    if len(paths) <= MAX_IMAGES:
        return [paths]
    groups = [paths[i:i + MAX_IMAGES] for i in range(0, len(paths), MAX_IMAGES)]
    return groups

def load_images(paths: List[str]) -> List[Image.Image]:
    return [Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE)) for p in paths]

def find_annotation_dirs() -> List[str]:
    ann_dirs = []
    for base in (OPEN_DIR, REAL_DIR):
        if not os.path.isdir(base):
            print(f"WARNING: base dir missing: {base}")
            continue
        for root, _dirs, files in os.walk(base):
            if "chathistory_annotated.json" in files:
                ann_dirs.append(root)
    print(f"Found {len(ann_dirs)} annotated dirs across REAL + OPEN")
    return ann_dirs

def build_samples() -> List[Dict[str, Any]]:
    samples, skipped = [], []
    for d in find_annotation_dirs():
        json_path = os.path.join(d, "chathistory_annotated.json")
        try:
            with open(json_path) as f:
                ann = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            skipped.append((d, f"bad json: {e}"))
            continue

        imgs = sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))
        if not imgs:
            skipped.append((d, "no images"))
            continue

        if get_feature(ann, 'label') == "Not specified":
            skipped.append((d, "no label"))
            continue

        img_paths = [os.path.join(d, f) for f in imgs]
        for gi, group in enumerate(chunk_images(img_paths)):
            samples.append({
                "sample_id": f"{os.path.relpath(d, os.path.dirname(d))}__grp{gi}",
                "image_paths": group,
                "annotations": ann,
            })

    print(f"Built {len(samples)} samples ({len(skipped)} dirs skipped)")
    if skipped:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, "skipped.json"), "w") as f:
            json.dump([{"dir": d, "reason": r} for d, r in skipped], f, indent=2)
    return samples

class Trainer4B:
    def __init__(self, model_path: str):
        self.model_path = model_path
        os.makedirs(LOGS_DIR, exist_ok=True)
        self._setup()

    def _setup(self):
        torch_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            # bnb_4bit_quant_storage=torch_dtype,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            attn_implementation="eager",
            torch_dtype=torch_dtype,
            quantization_config=bnb,
            device_map={"": local_rank},
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.processor.tokenizer.padding_side = "right"

    def peft_config(self):
        return LoraConfig(
            lora_alpha=LORA_RANK * 2,
            lora_dropout=0.05,
            r=LORA_RANK,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            modules_to_save=["lm_head"],
        )

    def collate_fn(self, examples):
        texts, images = [], []
        for ex in examples:
            imgs = load_images(ex["image_paths"])
            msgs = build_messages(ex["annotations"], len(imgs))
            images.append(imgs)
            texts.append(self.processor.apply_chat_template(
                msgs, add_generation_prompt=False, tokenize=False).strip())

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == 262144] = -100
        try:
            boi = self.processor.tokenizer.convert_tokens_to_ids(
                self.processor.tokenizer.special_tokens_map.get("boi_token", "<image>"))
            labels[labels == boi] = -100
        except Exception:
            pass
        batch["labels"] = labels
        return batch

    def training_args(self):
        bf16 = torch.cuda.get_device_capability()[0] >= 8
        return SFTConfig(
            output_dir=OUTPUT_DIR,
            logging_dir=LOGS_DIR,
            num_train_epochs=10,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="paged_adamw_8bit",
            learning_rate=2e-5,
            logging_steps=50,
            save_strategy="epoch",
            eval_strategy="epoch",
            bf16=bf16, fp16=not bf16,
            max_grad_norm=1.0,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            push_to_hub=False,
            report_to="tensorboard",
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False,
            label_names=["labels"],
            dataloader_num_workers=2,
            ddp_find_unused_parameters=False,
            load_best_model_at_end=False,
            average_tokens_across_devices=True,
        )

    def train(self, train_ds, val_ds):
        trainer = SFTTrainer(
            model=self.model,
            args=self.training_args(),
            train_dataset=train_ds,
            eval_dataset=val_ds,
            peft_config=self.peft_config(),
            data_collator=self.collate_fn,
        )
        last_ckpt = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None
        trainer.train(resume_from_checkpoint=last_ckpt)
        trainer.save_model(OUTPUT_DIR)
        trainer.save_state()
        return trainer

def main():
    samples = build_samples()
    train_s, val_s = train_test_split(samples, test_size=VAL_SIZE, random_state=SEED)
    train_ds = Dataset.from_list(train_s)
    val_ds   = Dataset.from_list(val_s)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    trainer = Trainer4B(MODEL_PATH)
    trainer.train(train_ds, val_ds)

if __name__ == "__main__":
    main()