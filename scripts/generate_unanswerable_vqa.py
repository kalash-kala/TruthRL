#!/usr/bin/env python3
"""
Unified Visual-Cue + SAM-Guided Perturbation Pipeline
======================================================

Single-script pipeline that:
  1. Reads VQA data from a **parquet** file (with embedded images).
  2. Generates visual cues via Gemini Flash.
  3. Produces a masking plan via Gemini Flash.
  4. Segments + perturbs with Lang-SAM (all in-memory).
  5. Verifies unanswerability via Gemini Flash.
  6. Saves original + perturbed image pair to disk **only** on success.
  7. Appends clean + perturbed records to the output JSONL.
  8. Stops as soon as `n` successful unanswerable pairs are generated.

Deduplication: skips images already present in the output JSONL.

Usage:
    python generate_unanswerable_vqa.py \
        --input_parquet  /path/to/vqa_stratified_300.parquet \
        --output_jsonl   /path/to/unanswerable_sam_targeted.jsonl \
        --image_dir      /path/to/sam_perturbed_images \
        -n 50
"""

import os
import io
import re
import cv2
import json
import time
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from scipy.ndimage import gaussian_filter

import torch

from google import genai
from google.genai import types

from dotenv import load_dotenv


# ================================================================
# CONFIG
# ================================================================

GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_ENV_PATH = "/home/kalashkala/Perception-R1/.env"

PERTURBATION_TYPES = ["blur", "color_shift", "noise", "darken", "occlude_partial"]

STRENGTH_PARAMS = {
    "blur": {"medium": 8.0, "strong": 14.0},
    "color_shift": {
        "medium": {"hue_shift": 50, "sat_factor": 0.45},
        "strong": {"hue_shift": 90, "sat_factor": 0.25},
    },
    "noise": {"medium": 45, "strong": 70},
    "darken": {"medium": 0.30, "strong": 0.12},
    "occlude_partial": {"medium": 0.55, "strong": 0.80},
}


# ================================================================
# LOGGING
# ================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ================================================================
# IMAGE HELPERS
# ================================================================

def pil_to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def save_image(img: Image.Image, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=95)


# ================================================================
# GEMINI HELPERS
# ================================================================

def build_gemini_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set.")
    return genai.Client(api_key=api_key)


def extract_json_from_text(text: str) -> Dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError(f"Could not parse JSON from Gemini response: {text[:200]}")


# ================================================================
# VISUAL CUE GENERATION  (from build_visual_cues.py)
# ================================================================

def build_visual_cue_prompt(question: str, gold_answer: str, category: str) -> str:
    return f"""
You are creating atomic visual evidence annotations for a multimodal reasoning dataset.

Task:
Given the image and the question, extract only the visual evidence relevant to answering the question.

Instructions:
- Return 2 to 5 short atomic visual cues.
- Each cue must be directly visible in the image.
- Do not use background knowledge.
- Do not speculate.
- Do not provide chain-of-thought.
- Do not mention information irrelevant to the question.
- Prefer literal, compact observations.

Return JSON only in exactly this format:
{{
  "visual_cues": ["cue 1", "cue 2", "cue 3"],
  "short_reason": "one short sentence"
}}

Metadata:
- Question category: {category}

Question: {question}
Gold answer: {gold_answer}
""".strip()


def normalize_cue_output(obj: Dict) -> Dict:
    cues = obj.get("visual_cues", [])
    if not isinstance(cues, list):
        cues = []
    cues = [str(x).strip() for x in cues if str(x).strip()]
    cues = cues[:5]
    short_reason = str(obj.get("short_reason", "")).strip()
    return {"visual_cues": cues, "short_reason": short_reason}


def get_visual_cues(
    client: genai.Client,
    jpeg_bytes: bytes,
    question: str,
    gold_answer: str,
    category: str,
    max_retries: int = 3,
) -> Dict:
    prompt = build_visual_cue_prompt(question, gold_answer, category)
    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=2048,
                    response_mime_type="application/json",
                ),
            )
            parsed = extract_json_from_text(response.text)
            return normalize_cue_output(parsed)
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to get visual cues: {last_err}")


# ================================================================
# MASKING PLAN  (from sam_targeted_perturbation.py)
# ================================================================

def build_masking_plan_prompt(
    question: str,
    answer: str,
    category: str,
    visual_cues: List[str],
) -> str:
    cues_text = "\n".join(f"  - {c}" for c in visual_cues) if visual_cues \
        else "  - (none provided)"
    return f"""You are an adversarial perturbation planner for a visual question answering dataset.

Given an image, a question, the gold answer, and visual cues, plan which specific
visual components to degrade so that the question becomes GENUINELY UNCERTAIN.

CRITICAL RULES:
1. Do NOT suggest removing the entire subject. That makes it OBVIOUSLY unanswerable.
2. Instead, degrade SPECIFIC ATTRIBUTES or EVIDENCE that the question depends on.
3. Goal is UNCERTAINTY — the viewer should think "I can't tell" not "that thing is gone."
4. Each component MUST be a concrete, localizable object or region in the image.
5. The "text_prompt_for_sam" MUST be a SHORT noun phrase (2-4 words max) that a
   visual grounding model can detect. Examples: "red jersey", "soccer ball",
   "traffic light", "person face", "street sign". Do NOT use long descriptions.
6. Choose perturbation_type from: blur, color_shift, noise, darken, occlude_partial
7. Use perturbation_strength "strong" for all components.
8. Return 1 to 3 components — just enough to make the question uncertain.

Return JSON ONLY in this exact format:
{{
  "components_to_mask": [
    {{
      "description": "the red jersey on the soccer player",
      "text_prompt_for_sam": "red jersey",
      "perturbation_type": "color_shift",
      "perturbation_strength": "strong",
      "reasoning": "The question asks about color. Shifting the jersey color makes it ambiguous."
    }}
  ],
  "overall_strategy": "short sentence explaining the degradation strategy",
  "expected_uncertainty": "what the viewer would be uncertain about"
}}

Category: {category}
Question: {question}
Gold answer: {answer}

Visual cues from the clean image:
{cues_text}""".strip()


def get_masking_plan(
    client: genai.Client,
    jpeg_bytes: bytes,
    question: str,
    answer: str,
    category: str,
    visual_cues: List[str],
    max_retries: int = 3,
) -> Optional[Dict]:
    prompt = build_masking_plan_prompt(question, answer, category, visual_cues)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=4096,
                    response_mime_type="application/json",
                ),
            )
            parsed = extract_json_from_text(resp.text)
            components = parsed.get("components_to_mask", [])
            if not components:
                log.warning("Gemini returned empty masking plan, retrying...")
                time.sleep(2)
                continue
            for comp in components:
                if "text_prompt_for_sam" not in comp:
                    comp["text_prompt_for_sam"] = comp.get("description", "object")[:30]
                if "perturbation_type" not in comp:
                    comp["perturbation_type"] = "blur"
                if comp["perturbation_type"] not in PERTURBATION_TYPES:
                    comp["perturbation_type"] = "blur"
                if "perturbation_strength" not in comp:
                    comp["perturbation_strength"] = "strong"
            return parsed
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    log.error(f"Masking plan failed after {max_retries} retries: {last_err}")
    return None


# ================================================================
# LANG-SAM + PERTURBATION  (from sam_targeted_perturbation.py)
# ================================================================

def init_sam_model(sam_type: str = "sam2", device: str = "cuda:0"):
    from lang_sam import LangSAM
    try:
        model = LangSAM(sam_type)
        log.info(f"Initialized Lang-SAM with {sam_type}")
    except Exception as e:
        log.warning(f"Failed to init with {sam_type}: {e}. Trying default...")
        model = LangSAM()
        log.info("Initialized Lang-SAM with default model")
    return model


def predict_masks(
    sam_model,
    image_pil: Image.Image,
    text_prompt: str,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
) -> Optional[np.ndarray]:
    def _to_numpy(masks):
        if masks is None:
            return None
        if isinstance(masks, torch.Tensor):
            arr = masks.cpu().numpy()
        elif isinstance(masks, list):
            arr = np.array([
                m.cpu().numpy() if isinstance(m, torch.Tensor) else np.array(m)
                for m in masks
            ])
        elif isinstance(masks, np.ndarray):
            arr = masks
        else:
            return None
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        return arr.astype(bool) if arr.size > 0 else None

    # Attempt 1: newer API (v0.2+)
    try:
        results = sam_model.predict(
            [image_pil], [text_prompt],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        if results and len(results) > 0:
            r = results[0]
            masks = getattr(r, 'masks', None) or (r.get('masks') if isinstance(r, dict) else None)
            out = _to_numpy(masks)
            if out is not None:
                return out
    except (TypeError, AttributeError, Exception):
        pass

    # Attempt 2: older API (v0.1.x)
    try:
        masks, boxes, phrases, logits = sam_model.predict(
            image_pil, text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        out = _to_numpy(masks)
        if out is not None:
            return out
    except Exception:
        pass

    return None


# ── Mask post-processing ─────────────────────────────────────────

def feather_mask(mask: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    return gaussian_filter(mask.astype(np.float32), sigma=sigma)


def select_best_mask(masks: np.ndarray) -> np.ndarray:
    areas = [m.sum() for m in masks]
    return masks[np.argmax(areas)]


# ── Localized perturbation functions ──────────────────────────────

def masked_gaussian_blur(
    img_arr: np.ndarray, mask_float: np.ndarray, strength: str = "strong",
) -> np.ndarray:
    radius = STRENGTH_PARAMS["blur"].get(strength, 14.0)
    pil_img = Image.fromarray(img_arr)
    blurred = np.array(pil_img.filter(ImageFilter.GaussianBlur(radius=radius)))
    mask_3d = mask_float[:, :, np.newaxis]
    result = (mask_3d * blurred + (1.0 - mask_3d) * img_arr)
    return np.clip(result, 0, 255).astype(np.uint8)


def masked_color_shift(
    img_arr: np.ndarray, mask_float: np.ndarray, strength: str = "strong",
) -> np.ndarray:
    params = STRENGTH_PARAMS["color_shift"].get(strength, {"hue_shift": 90, "sat_factor": 0.25})
    hue_shift = params["hue_shift"]
    sat_factor = params["sat_factor"]
    hsv = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    bool_mask = mask_float > 0.5
    hsv[bool_mask, 0] = (hsv[bool_mask, 0] + hue_shift) % 180
    hsv[bool_mask, 1] *= sat_factor
    shifted = cv2.cvtColor(
        np.clip(hsv, 0, [179, 255, 255]).astype(np.uint8),
        cv2.COLOR_HSV2RGB,
    ).astype(np.float32)
    mask_3d = mask_float[:, :, np.newaxis]
    result = mask_3d * shifted + (1.0 - mask_3d) * img_arr
    return np.clip(result, 0, 255).astype(np.uint8)


def masked_noise(
    img_arr: np.ndarray, mask_float: np.ndarray, strength: str = "strong",
) -> np.ndarray:
    std = STRENGTH_PARAMS["noise"].get(strength, 70)
    noise = np.random.normal(0, std, img_arr.shape).astype(np.float32)
    mask_3d = mask_float[:, :, np.newaxis]
    noisy = img_arr.astype(np.float32) + mask_3d * noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def masked_darken(
    img_arr: np.ndarray, mask_float: np.ndarray, strength: str = "strong",
) -> np.ndarray:
    factor = STRENGTH_PARAMS["darken"].get(strength, 0.12)
    darkened = (img_arr.astype(np.float32) * factor)
    mask_3d = mask_float[:, :, np.newaxis]
    result = mask_3d * darkened + (1.0 - mask_3d) * img_arr
    return np.clip(result, 0, 255).astype(np.uint8)


def masked_occlude(
    img_arr: np.ndarray, mask_float: np.ndarray, strength: str = "strong",
) -> np.ndarray:
    alpha = STRENGTH_PARAMS["occlude_partial"].get(strength, 0.80)
    overlay_color = np.array([128, 128, 128], dtype=np.float32)
    occluded = img_arr.astype(np.float32) * (1 - alpha) + overlay_color * alpha
    mask_3d = mask_float[:, :, np.newaxis]
    result = mask_3d * occluded + (1.0 - mask_3d) * img_arr
    return np.clip(result, 0, 255).astype(np.uint8)


PERTURBATION_FN_MAP = {
    "blur": masked_gaussian_blur,
    "color_shift": masked_color_shift,
    "noise": masked_noise,
    "darken": masked_darken,
    "occlude_partial": masked_occlude,
}


def apply_masked_perturbation(
    img_arr: np.ndarray,
    mask_bool: np.ndarray,
    perturbation_type: str,
    strength: str = "strong",
    feather_sigma: float = 5.0,
) -> np.ndarray:
    mask_float = feather_mask(mask_bool, sigma=feather_sigma)
    fn = PERTURBATION_FN_MAP.get(perturbation_type, masked_gaussian_blur)
    return fn(img_arr, mask_float, strength)


# ── Fallback: global perturbation ─────────────────────────────────

def global_gaussian_blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def global_downsample(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((nw, nh), Image.Resampling.BILINEAR).resize(
        (w, h), Image.Resampling.BILINEAR)

def global_center_crop(img: Image.Image, ratio: float) -> Image.Image:
    w, h = img.size
    cw, ch = int(w * ratio), int(h * ratio)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize(
        (w, h), Image.Resampling.BILINEAR)

def global_random_occlusion(img: Image.Image, area_ratio: float) -> Image.Image:
    img = img.copy()
    w, h = img.size
    pw = ph = max(1, min(int(np.sqrt(w * h * area_ratio)), min(w, h)))
    x1, y1 = random.randint(0, w - pw), random.randint(0, h - ph)
    ImageDraw.Draw(img).rectangle([x1, y1, x1 + pw, y1 + ph], fill=(0, 0, 0))
    return img

def global_darken(img: Image.Image, area_ratio: float, darkness: float) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    h, w, _ = arr.shape
    pw = ph = max(1, min(int(np.sqrt(w * h * area_ratio)), min(w, h)))
    x1, y1 = random.randint(0, w - pw), random.randint(0, h - ph)
    arr[y1:y1+ph, x1:x1+pw] *= darkness
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


GLOBAL_FALLBACK_OPS = {
    "spatial_relational": [
        ("blur_strong",     lambda x: global_gaussian_blur(x, 4.0)),
        ("crop_strong",     lambda x: global_center_crop(x, 0.55)),
        ("occlusion_large", lambda x: global_random_occlusion(x, 0.20)),
    ],
    "attribute_recognition": [
        ("blur_strong",       lambda x: global_gaussian_blur(x, 4.0)),
        ("contrast_strong",   lambda x: ImageEnhance.Contrast(x).enhance(0.35)),
        ("darken_region",     lambda x: global_darken(x, 0.22, 0.08)),
    ],
    "counting": [
        ("downsample_strong", lambda x: global_downsample(x, 0.22)),
        ("occlusion_large",   lambda x: global_random_occlusion(x, 0.20)),
        ("crop_strong",       lambda x: global_center_crop(x, 0.55)),
    ],
    "existence_presence": [
        ("occlusion_large",   lambda x: global_random_occlusion(x, 0.20)),
        ("darken_region",     lambda x: global_darken(x, 0.22, 0.05)),
        ("crop_strong",       lambda x: global_center_crop(x, 0.55)),
    ],
    "activity_interaction": [
        ("blur_strong",       lambda x: global_gaussian_blur(x, 4.5)),
        ("crop_strong",       lambda x: global_center_crop(x, 0.55)),
        ("occlusion_large",   lambda x: global_random_occlusion(x, 0.18)),
    ],
    "scene_context": [
        ("downsample_strong", lambda x: global_downsample(x, 0.22)),
        ("blur_strong",       lambda x: global_gaussian_blur(x, 4.5)),
        ("crop_strong",       lambda x: global_center_crop(x, 0.55)),
    ],
    "object_recognition": [
        ("blur_strong",       lambda x: global_gaussian_blur(x, 4.0)),
        ("occlusion_large",   lambda x: global_random_occlusion(x, 0.18)),
        ("downsample_strong", lambda x: global_downsample(x, 0.22)),
    ],
}

DEFAULT_GLOBAL_OPS = [
    ("blur_strong",       lambda x: global_gaussian_blur(x, 4.0)),
    ("downsample_strong", lambda x: global_downsample(x, 0.22)),
    ("occlusion_large",   lambda x: global_random_occlusion(x, 0.18)),
    ("crop_strong",       lambda x: global_center_crop(x, 0.55)),
]


def apply_global_fallback(
    img: Image.Image, category: str,
) -> Tuple[Image.Image, str]:
    ops = GLOBAL_FALLBACK_OPS.get(category, DEFAULT_GLOBAL_OPS)
    name, fn = random.choice(ops)
    return fn(img), name


# ================================================================
# VERIFICATION  (from sam_targeted_perturbation.py)
# ================================================================

def build_verification_prompt(
    question: str, answer: str, category: str, perturbation_summary: str,
) -> str:
    return f"""You are verifying whether a perturbed image creates genuine UNCERTAINTY
for a visual question answering task.

Given the perturbed image and the question below:
1. Try to answer the question based ONLY on what you see in this image.
2. Do NOT use external knowledge.

Rules:
- If you can STILL confidently answer the question → mark ANSWERABLE.
- If you TRULY cannot determine the answer due to visual degradation → mark UNANSWERABLE.

Return JSON ONLY in this exact format:
{{
  "answerability": "ANSWERABLE or UNANSWERABLE",
  "confidence": "HIGH or MEDIUM or LOW or CANNOT_DETERMINE",
  "attempted_answer": "your best guess if forced",
  "uncertainty_reason": "what specifically is uncertain in the image",
  "visual_cues": ["cue 1 describing what IS visible", "cue 2 about degraded region", "cue 3"],
  "failure_type": "blur or color_ambiguity or partial_occlusion or noise or darkness or other"
}}

Category: {category}
Question: {question}
Original gold answer: {answer}
Perturbation applied: {perturbation_summary}""".strip()


def verify_and_get_cues(
    client: genai.Client,
    jpeg_bytes: bytes,
    question: str,
    answer: str,
    category: str,
    perturbation_summary: str,
    max_retries: int = 3,
) -> Optional[Dict]:
    prompt = build_verification_prompt(
        question, answer, category, perturbation_summary)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=4096,
                    response_mime_type="application/json",
                ),
            )
            parsed = extract_json_from_text(resp.text)
            ans = str(parsed.get("answerability", "")).strip().upper()
            if ans not in {"ANSWERABLE", "UNANSWERABLE"}:
                ans = "ANSWERABLE"
            parsed["answerability"] = ans
            cues = parsed.get("visual_cues", [])
            if not isinstance(cues, list):
                cues = []
            parsed["visual_cues"] = [str(c).strip() for c in cues if str(c).strip()][:5]
            return parsed
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    log.error(f"Verification failed after {max_retries} retries: {last_err}")
    return None


# ================================================================
# DEDUPLICATION
# ================================================================

def load_already_processed(jsonl_path: str) -> set:
    """Load set of image filenames already processed in the JSONL."""
    processed = set()
    if not os.path.exists(jsonl_path):
        return processed
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                fname = os.path.basename(rec.get("image_path", ""))
                if fname.startswith("PERTURBED_"):
                    fname = fname[len("PERTURBED_"):]
                if fname:
                    processed.add(fname)
            except json.JSONDecodeError:
                continue
    return processed


# ================================================================
# CORE PIPELINE: PROCESS ONE ROW
# ================================================================

def process_single_row(
    row: pd.Series,
    gemini_client: genai.Client,
    sam_model,
    args,
) -> Optional[Tuple[Dict, Dict]]:
    """
    Process a single parquet row through the full pipeline.
    All image work is done in-memory — nothing touches disk.

    Returns (clean_record, perturbed_record) on success, else None.
    """
    question_id = str(row["question_id"])
    filename = row["image"]["path"]
    question = row["question"]
    category = row["category"]
    question_type = row.get("question_type", "")
    answer_type = row.get("answer_type", "")
    gold_answer = row["multiple_choice_answer"]  # used for Gemini prompts

    # Build clean answers list: only "yes"-confidence answers
    raw_answers = row["answers"]
    clean_answers = []
    for a in raw_answers:
        if isinstance(a, dict) and a.get("answer_confidence") == "yes":
            clean_answers.append(a["answer"])
    if not clean_answers:
        # Fallback: use all answers if none have "yes" confidence
        clean_answers = [a["answer"] for a in raw_answers if isinstance(a, dict)]

    # ── Load image IN MEMORY ─────────────────────────────────────
    try:
        img_pil = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
    except Exception as e:
        log.warning(f"[{question_id}] Failed to decode image from parquet: {e}")
        return None

    jpeg_bytes = pil_to_jpeg_bytes(img_pil)

    # ── Step 1: Generate visual cues via Gemini ──────────────────
    try:
        cues = get_visual_cues(
            gemini_client, jpeg_bytes, question, gold_answer, category)
    except Exception as e:
        log.warning(f"[{question_id}] Visual cue generation failed: {e}")
        return None

    if args.sleep_interval > 0:
        time.sleep(args.sleep_interval)

    visual_cues = cues["visual_cues"]
    cue_short_reason = cues["short_reason"]

    # ── Step 2: Get masking plan from Gemini ──────────────────────
    masking_plan = get_masking_plan(
        gemini_client, jpeg_bytes, question, gold_answer, category, visual_cues)

    if args.sleep_interval > 0:
        time.sleep(args.sleep_interval)

    used_sam = False
    perturbation_types_applied = []
    components_masked = []
    perturbed_img = None

    if masking_plan and masking_plan.get("components_to_mask"):
        # ── Step 3: SAM + localized perturbation (in-memory) ─────
        img_arr = np.array(img_pil)
        any_mask_found = False

        for comp in masking_plan["components_to_mask"]:
            text_prompt = comp["text_prompt_for_sam"]
            ptype = comp.get("perturbation_type", "blur")
            strength = comp.get("perturbation_strength", "strong")

            masks = predict_masks(sam_model, img_pil, text_prompt)

            # Fallback: try broader prompt
            if masks is None:
                words = text_prompt.split()
                if len(words) > 1:
                    broader = words[-1]
                    log.info(f"[{question_id}] SAM missed '{text_prompt}', "
                             f"trying broader: '{broader}'")
                    masks = predict_masks(sam_model, img_pil, broader)

            if masks is None:
                log.info(f"[{question_id}] SAM found no mask for '{text_prompt}'")
                continue

            best_mask = select_best_mask(masks)
            mask_area = best_mask.sum() / best_mask.size

            if mask_area < 0.005:
                log.info(f"[{question_id}] Mask too small ({mask_area:.3%}) "
                         f"for '{text_prompt}', skipping")
                continue

            log.info(f"[{question_id}] Masking '{text_prompt}' "
                     f"(area={mask_area:.1%}, type={ptype})")

            img_arr = apply_masked_perturbation(
                img_arr, best_mask, ptype, strength,
                feather_sigma=args.feather_sigma,
            )
            any_mask_found = True
            perturbation_types_applied.append(ptype)
            components_masked.append(text_prompt)

        if any_mask_found:
            perturbed_img = Image.fromarray(img_arr)
            used_sam = True

    # ── Fallback to global perturbation ───────────────────────────
    if perturbed_img is None:
        log.info(f"[{question_id}] Falling back to global perturbation")
        perturbed_img, fallback_name = apply_global_fallback(img_pil, category)
        perturbation_types_applied = [fallback_name]
        components_masked = ["global"]

    perturbed_jpeg = pil_to_jpeg_bytes(perturbed_img)
    perturbation_summary = " + ".join(perturbation_types_applied)

    # ── Step 4: Verify unanswerability via Gemini ─────────────────
    verification = verify_and_get_cues(
        gemini_client, perturbed_jpeg, question, gold_answer,
        category, perturbation_summary,
    )

    if args.sleep_interval > 0:
        time.sleep(args.sleep_interval)

    if verification is None:
        log.warning(f"[{question_id}] Verification call failed, skipping")
        return None

    if verification["answerability"] != "UNANSWERABLE":
        # ── Retry with stronger global perturbation ───────────────
        log.info(f"[{question_id}] Still ANSWERABLE after perturbation. "
                 f"Retrying with stronger global fallback...")
        perturbed_img_v2, fallback_name_v2 = apply_global_fallback(
            perturbed_img, category)
        perturbation_types_applied.append(fallback_name_v2)
        perturbed_jpeg_v2 = pil_to_jpeg_bytes(perturbed_img_v2)
        perturbation_summary_v2 = " + ".join(perturbation_types_applied)

        verification = verify_and_get_cues(
            gemini_client, perturbed_jpeg_v2, question, gold_answer,
            category, perturbation_summary_v2,
        )

        if args.sleep_interval > 0:
            time.sleep(args.sleep_interval)

        if verification is None or verification["answerability"] != "UNANSWERABLE":
            log.info(f"[{question_id}] Still ANSWERABLE after retry, skipping")
            return None

        perturbed_img = perturbed_img_v2
        perturbation_summary = perturbation_summary_v2

    # ══════════════════════════════════════════════════════════════
    # CONFIRMED UNANSWERABLE — NOW save images to disk
    # ══════════════════════════════════════════════════════════════

    image_dir = Path(args.image_dir)

    # Save original
    original_save_path = image_dir / filename
    save_image(img_pil, str(original_save_path))

    # Save perturbed
    perturbed_filename = f"PERTURBED_{filename}"
    perturbed_save_path = image_dir / perturbed_filename
    save_image(perturbed_img, str(perturbed_save_path))

    # ── Build records ─────────────────────────────────────────────

    clean_record = {
        "id": f"{question_id}_clean",
        "source_id": question_id,
        "image_path": str(original_save_path),
        "question": question,
        "answers": clean_answers,
        "category": category,
        "question_type": question_type,
        "answer_type": answer_type,
        "variant": "clean",
        "perturbation_type": "none",
        "visual_cues": visual_cues,
        "cue_short_reason": cue_short_reason,
        "gemini_tag": {
            "answerability": "ANSWERABLE",
            "failure_type": "none",
            "short_reason": "Original image is clear and unperturbed.",
        },
    }

    perturbed_record = {
        "id": f"{question_id}_sam_perturbed",
        "source_id": question_id,
        "original_image_path": str(original_save_path),
        "image_path": str(perturbed_save_path),
        "question": question,
        "answers": ["I don't know"],
        "category": category,
        "question_type": question_type,
        "answer_type": answer_type,
        "variant": "sam_targeted" if used_sam else "global_fallback",
        "perturbation_type": perturbation_summary,
        "masking_details": {
            "components_masked": components_masked,
            "strategy": masking_plan.get("overall_strategy", "") if masking_plan else "global fallback",
            "expected_uncertainty": masking_plan.get("expected_uncertainty", "") if masking_plan else "",
            "used_sam": used_sam,
        },
        "visual_cues": verification.get("visual_cues", []),
        "uncertainty_reason": verification.get("uncertainty_reason", ""),
        "gemini_tag": {
            "answerability": "UNANSWERABLE",
            "confidence": verification.get("confidence", "CANNOT_DETERMINE"),
            "failure_type": verification.get("failure_type", "other"),
            "short_reason": verification.get("uncertainty_reason", ""),
            "attempted_answer": verification.get("attempted_answer", ""),
        },
    }

    return (clean_record, perturbed_record)


# ================================================================
# MAIN
# ================================================================

def main(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Load environment ──────────────────────────────────────────
    load_dotenv(dotenv_path=args.env_path)
    log.info(f"Loaded env from: {args.env_path}")

    # ── Initialize clients ────────────────────────────────────────
    gemini_client = build_gemini_client()
    log.info("Gemini client initialized")

    sam_model = init_sam_model(sam_type=args.sam_type, device=args.device)
    log.info("Lang-SAM model ready")

    # ── Load parquet ──────────────────────────────────────────────
    log.info(f"Loading parquet: {args.input_parquet}")
    df = pd.read_parquet(args.input_parquet)
    log.info(f"  {len(df)} total rows in parquet")

    # ── Build dedup set ───────────────────────────────────────────
    already_processed = load_already_processed(args.output_jsonl)
    log.info(f"  {len(already_processed)} images already processed (will skip)")

    # ── Prepare output dir ────────────────────────────────────────
    Path(args.image_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    # ── Counters ──────────────────────────────────────────────────
    unanswerable_count = 0
    skipped_dedup = 0
    answerable_skip_count = 0
    error_count = 0
    sam_success_count = 0
    fallback_count = 0

    # ── Process rows ──────────────────────────────────────────────
    with open(args.output_jsonl, "a", encoding="utf-8") as out_f:
        for idx, row in df.iterrows():
            filename = row["image"]["path"]
            question_id = str(row["question_id"])

            # Dedup check
            if filename in already_processed:
                skipped_dedup += 1
                continue

            log.info(f"[{idx + 1}/{len(df)}] Processing question_id={question_id} "
                     f"image={filename}")

            try:
                result = process_single_row(
                    row, gemini_client, sam_model, args)
            except Exception as e:
                log.error(f"[{question_id}] Unexpected error: {e}", exc_info=True)
                error_count += 1
                continue

            if result is None:
                answerable_skip_count += 1
                continue

            clean_record, perturbed_record = result

            # Write both records
            out_f.write(json.dumps(clean_record) + "\n")
            out_f.write(json.dumps(perturbed_record) + "\n")
            out_f.flush()

            # Track in dedup set for this session
            already_processed.add(filename)

            unanswerable_count += 1
            variant = perturbed_record.get("variant", "unknown")
            if variant == "sam_targeted":
                sam_success_count += 1
            else:
                fallback_count += 1

            components = perturbed_record.get("masking_details", {}).get(
                "components_masked", [])
            log.info(
                f"  ✅ UNANSWERABLE #{unanswerable_count} "
                f"(id={question_id}, variant={variant}, "
                f"components={components})"
            )

            # ── Stop early if target reached ──────────────────────
            if args.target_count > 0 and unanswerable_count >= args.target_count:
                log.info(f"🎯 Reached target of {args.target_count}. Stopping.")
                break

    # ── Summary ───────────────────────────────────────────────────
    log.info(f"\n{'=' * 60}")
    log.info(f"Pipeline Complete")
    log.info(f"  Skipped (already processed): {skipped_dedup}")
    log.info(f"  UNANSWERABLE saved:     {unanswerable_count} "
             f"(×2 = {unanswerable_count * 2} JSONL lines)")
    log.info(f"    ├─ SAM-targeted:      {sam_success_count}")
    log.info(f"    └─ Global fallback:   {fallback_count}")
    log.info(f"  ANSWERABLE skipped:     {answerable_skip_count}")
    log.info(f"  Errors:                 {error_count}")
    log.info(f"  Output JSONL:           {args.output_jsonl}")
    log.info(f"  Images saved to:        {args.image_dir}")
    log.info(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified Visual-Cue + SAM-Guided Perturbation Pipeline "
                    "for VQA Unanswerable Sample Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 50 unanswerable pairs
  python generate_unanswerable_vqa.py \\
      --input_parquet  /path/to/vqa_stratified_300.parquet \\
      --output_jsonl   /path/to/unanswerable_sam_targeted.jsonl \\
      --image_dir      /path/to/sam_perturbed_images \\
      -n 50

  # Process all rows (no early stop)
  python generate_unanswerable_vqa.py \\
      --input_parquet  /path/to/vqa_stratified_300.parquet \\
      --output_jsonl   /path/to/unanswerable_sam_targeted.jsonl \\
      --image_dir      /path/to/sam_perturbed_images
        """,
    )

    # I/O
    parser.add_argument("--input_parquet", type=str, required=True,
                        help="Input parquet file with VQA data + embedded images.")
    parser.add_argument("--output_jsonl", type=str, required=True,
                        help="Output JSONL path (append mode).")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory to save original + perturbed images.")

    # Target
    parser.add_argument("-n", "--target_count", type=int, default=0,
                        help="Stop after this many unanswerable pairs (0=all). "
                             "Default: 0")

    # Model config
    parser.add_argument("--sam_type", type=str, default="sam2",
                        choices=["sam2", "vit_h", "vit_l", "vit_b"],
                        help="SAM model variant for Lang-SAM. Default: sam2")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device for SAM. Default: cuda:0")
    parser.add_argument("--env_path", type=str,
                        default=DEFAULT_ENV_PATH,
                        help=f"Path to .env file. Default: {DEFAULT_ENV_PATH}")

    # Processing
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed. Default: 42")
    parser.add_argument("--sleep_interval", type=float, default=2.0,
                        help="Seconds between Gemini API calls. Default: 2.0")
    parser.add_argument("--feather_sigma", type=float, default=5.0,
                        help="Gaussian sigma for mask edge feathering. Default: 5.0")

    args = parser.parse_args()
    main(args)
