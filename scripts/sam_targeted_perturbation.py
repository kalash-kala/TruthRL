#!/usr/bin/env python3
"""
SAM-Guided Targeted Perturbation Pipeline
==========================================

Single-script pipeline that generates unanswerable VQA training samples
by selectively degrading question-critical image regions.

Pipeline per image:
  1. Gemini Flash → masking plan (which components to degrade and how)
  2. Lang-SAM    → pixel-precise masks from text prompts
  3. Localized perturbations applied ONLY within masked regions
  4. Gemini Flash → verify unanswerability + generate new visual cues
  5. If confirmed UNANSWERABLE → save original + perturbed record pair

Fallback: If Lang-SAM fails to segment any component, applies category-
aware global perturbation (blur, crop, etc.) instead.

Requirements:
    pip install torch torchvision lang-sam google-genai Pillow \
                numpy scipy python-dotenv pandas

Infrastructure: 2×A100 (160 GB VRAM total)

Usage:
    python sam_targeted_perturbation.py \
        --input_jsonl  /path/to/clean_vqa_with_visual_cues_tagged.jsonl \
        --output_jsonl /path/to/unanswerable_sam.jsonl \
        --image_dir    /path/to/output_images \
        --image_root   /path/to/source_images \
        --sleep_interval 2.0
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
DEFAULT_ENV_PATH = "/home/sriramg/kalashabhayk/Perception-R1/.env"
DEFAULT_IMAGE_ROOT = "/home/sriramg/kalashabhayk/visual-question-answering/processed_for_verl/images"

PERTURBATION_TYPES = ["blur", "color_shift", "noise", "darken", "occlude_partial"]

# Strength mappings
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

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def save_image(img: Image.Image, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=95)


def resolve_image_path(original_path: str, image_root: Optional[str]) -> str:
    """Remap image path: keep only the filename, prepend image_root."""
    if image_root:
        filename = os.path.basename(original_path)
        return os.path.join(image_root, filename)
    return original_path


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
# STEP 1: GEMINI MASKING PLAN
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
                    max_output_tokens=1024,
                    response_mime_type="application/json",
                ),
            )
            parsed = extract_json_from_text(resp.text)
            components = parsed.get("components_to_mask", [])
            if not components:
                log.warning("Gemini returned empty masking plan, retrying...")
                time.sleep(2)
                continue
            # Validate each component has required fields
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
# STEP 2: LANG-SAM + PERTURBATION
# ================================================================

def init_sam_model(sam_type: str = "sam2", device: str = "cuda:0"):
    """Initialize Lang-SAM model."""
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
    """
    Predict masks using Lang-SAM with fallback across API versions.
    Returns: numpy array of shape (N, H, W) dtype=bool, or None if failed.
    """
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

    # Attempt 1: newer API (v0.2+) — list-based
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

    # Attempt 2: older API (v0.1.x) — single image
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
    """Smooth mask edges for natural blending. Returns float mask [0, 1]."""
    return gaussian_filter(mask.astype(np.float32), sigma=sigma)


def select_best_mask(masks: np.ndarray) -> np.ndarray:
    """Select the mask with the largest area (most confident segmentation)."""
    areas = [m.sum() for m in masks]
    return masks[np.argmax(areas)]


# ── Localized perturbation functions ──────────────────────────────

def masked_gaussian_blur(
    img_arr: np.ndarray,
    mask_float: np.ndarray,
    strength: str = "strong",
) -> np.ndarray:
    """Apply Gaussian blur only within the masked region."""
    radius = STRENGTH_PARAMS["blur"].get(strength, 14.0)
    pil_img = Image.fromarray(img_arr)
    blurred = np.array(pil_img.filter(ImageFilter.GaussianBlur(radius=radius)))
    # Blend: result = mask * blurred + (1 - mask) * original
    mask_3d = mask_float[:, :, np.newaxis]
    result = (mask_3d * blurred + (1.0 - mask_3d) * img_arr)
    return np.clip(result, 0, 255).astype(np.uint8)


def masked_color_shift(
    img_arr: np.ndarray,
    mask_float: np.ndarray,
    strength: str = "strong",
) -> np.ndarray:
    """Shift hue and desaturate within the masked region."""
    params = STRENGTH_PARAMS["color_shift"].get(strength, {"hue_shift": 90, "sat_factor": 0.25})
    hue_shift = params["hue_shift"]
    sat_factor = params["sat_factor"]

    hsv = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    bool_mask = mask_float > 0.5

    # Shift hue (OpenCV hue range: 0-179)
    hsv[bool_mask, 0] = (hsv[bool_mask, 0] + hue_shift) % 180
    # Desaturate
    hsv[bool_mask, 1] *= sat_factor

    shifted = cv2.cvtColor(
        np.clip(hsv, 0, [179, 255, 255]).astype(np.uint8),
        cv2.COLOR_HSV2RGB,
    ).astype(np.float32)

    mask_3d = mask_float[:, :, np.newaxis]
    result = mask_3d * shifted + (1.0 - mask_3d) * img_arr
    return np.clip(result, 0, 255).astype(np.uint8)


def masked_noise(
    img_arr: np.ndarray,
    mask_float: np.ndarray,
    strength: str = "strong",
) -> np.ndarray:
    """Add Gaussian noise within the masked region."""
    std = STRENGTH_PARAMS["noise"].get(strength, 70)
    noise = np.random.normal(0, std, img_arr.shape).astype(np.float32)
    mask_3d = mask_float[:, :, np.newaxis]
    noisy = img_arr.astype(np.float32) + mask_3d * noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def masked_darken(
    img_arr: np.ndarray,
    mask_float: np.ndarray,
    strength: str = "strong",
) -> np.ndarray:
    """Darken the masked region."""
    factor = STRENGTH_PARAMS["darken"].get(strength, 0.12)
    darkened = (img_arr.astype(np.float32) * factor)
    mask_3d = mask_float[:, :, np.newaxis]
    result = mask_3d * darkened + (1.0 - mask_3d) * img_arr
    return np.clip(result, 0, 255).astype(np.uint8)


def masked_occlude(
    img_arr: np.ndarray,
    mask_float: np.ndarray,
    strength: str = "strong",
) -> np.ndarray:
    """Partially occlude the masked region with a semi-transparent overlay."""
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
    """Apply a single localized perturbation within a feathered mask."""
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
    """Apply a strong global perturbation when SAM fails."""
    ops = GLOBAL_FALLBACK_OPS.get(category, DEFAULT_GLOBAL_OPS)
    name, fn = random.choice(ops)
    return fn(img), name


# ================================================================
# STEP 3: GEMINI VERIFICATION + NEW VISUAL CUES
# ================================================================

def build_verification_prompt(
    question: str,
    answer: str,
    category: str,
    perturbation_summary: str,
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
                    max_output_tokens=512,
                    response_mime_type="application/json",
                ),
            )
            parsed = extract_json_from_text(resp.text)
            # Normalize
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
# MAIN PIPELINE
# ================================================================

def process_single_record(
    record: Dict,
    gemini_client: genai.Client,
    sam_model,
    args,
) -> Optional[List[Dict]]:
    """
    Process a single VQA record through the full pipeline.
    Returns a list of 2 records [original, perturbed] if successful, else None.
    """
    rec_id = record["id"]
    original_image_path = resolve_image_path(record["image_path"], args.image_root)
    question = record["question"]
    answer = record["answer"]
    category = record.get("category", "object_recognition")
    visual_cues = record.get("visual_cues", [])

    # ── Load image ────────────────────────────────────────────────
    if not os.path.exists(original_image_path):
        log.warning(f"[{rec_id}] Image not found: {original_image_path}")
        return None

    try:
        original_img = load_image(original_image_path)
    except Exception as e:
        log.warning(f"[{rec_id}] Failed to load image: {e}")
        return None

    jpeg_bytes = pil_to_jpeg_bytes(original_img)

    # ── Step 1: Get masking plan from Gemini ──────────────────────
    masking_plan = get_masking_plan(
        gemini_client, jpeg_bytes, question, answer, category, visual_cues)

    if args.sleep_interval > 0:
        time.sleep(args.sleep_interval)

    used_sam = False
    perturbation_types_applied = []
    components_masked = []

    if masking_plan and masking_plan.get("components_to_mask"):
        # ── Step 2: SAM + localized perturbation ──────────────────
        img_arr = np.array(original_img)
        any_mask_found = False

        for comp in masking_plan["components_to_mask"]:
            text_prompt = comp["text_prompt_for_sam"]
            ptype = comp.get("perturbation_type", "blur")
            strength = comp.get("perturbation_strength", "strong")

            # Try primary prompt
            masks = predict_masks(sam_model, original_img, text_prompt)

            # Fallback: try broader prompt (drop adjective)
            if masks is None:
                words = text_prompt.split()
                if len(words) > 1:
                    broader = words[-1]  # last word (usually the noun)
                    log.info(f"[{rec_id}] SAM missed '{text_prompt}', "
                             f"trying broader: '{broader}'")
                    masks = predict_masks(sam_model, original_img, broader)

            if masks is None:
                log.info(f"[{rec_id}] SAM found no mask for '{text_prompt}'")
                continue

            best_mask = select_best_mask(masks)
            mask_area = best_mask.sum() / best_mask.size

            # Skip tiny masks (< 0.5% of image) — likely noise
            if mask_area < 0.005:
                log.info(f"[{rec_id}] Mask too small ({mask_area:.3%}) "
                         f"for '{text_prompt}', skipping")
                continue

            log.info(f"[{rec_id}] Masking '{text_prompt}' "
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
        else:
            perturbed_img = None
    else:
        perturbed_img = None

    # ── Fallback to global perturbation ───────────────────────────
    if perturbed_img is None:
        log.info(f"[{rec_id}] Falling back to global perturbation")
        perturbed_img, fallback_name = apply_global_fallback(
            original_img, category)
        perturbation_types_applied = [fallback_name]
        components_masked = ["global"]

    perturbed_jpeg = pil_to_jpeg_bytes(perturbed_img)
    perturbation_summary = " + ".join(perturbation_types_applied)

    # ── Step 3: Verify unanswerability via Gemini ─────────────────
    verification = verify_and_get_cues(
        gemini_client, perturbed_jpeg, question, answer,
        category, perturbation_summary,
    )

    if args.sleep_interval > 0:
        time.sleep(args.sleep_interval)

    if verification is None:
        log.warning(f"[{rec_id}] Verification call failed, skipping")
        return None

    if verification["answerability"] != "UNANSWERABLE":
        # ── Retry with stronger global perturbation ───────────────
        log.info(f"[{rec_id}] Still ANSWERABLE after perturbation. "
                 f"Retrying with stronger global fallback...")
        perturbed_img_v2, fallback_name_v2 = apply_global_fallback(
            perturbed_img, category)
        perturbation_types_applied.append(fallback_name_v2)
        perturbed_jpeg_v2 = pil_to_jpeg_bytes(perturbed_img_v2)
        perturbation_summary_v2 = " + ".join(perturbation_types_applied)

        verification = verify_and_get_cues(
            gemini_client, perturbed_jpeg_v2, question, answer,
            category, perturbation_summary_v2,
        )

        if args.sleep_interval > 0:
            time.sleep(args.sleep_interval)

        if verification is None or verification["answerability"] != "UNANSWERABLE":
            log.info(f"[{rec_id}] Still ANSWERABLE after retry, skipping")
            return None

        perturbed_img = perturbed_img_v2
        perturbed_jpeg = perturbed_jpeg_v2
        perturbation_summary = perturbation_summary_v2

    # ══════════════════════════════════════════════════════════════
    # CONFIRMED UNANSWERABLE — save images + build records
    # ══════════════════════════════════════════════════════════════

    image_filename = os.path.basename(record["image_path"])
    image_dir = Path(args.image_dir)

    # Save original
    original_save_path = image_dir / image_filename
    save_image(original_img, str(original_save_path))

    # Save perturbed
    perturbed_filename = f"PERTURBED_{image_filename}"
    perturbed_save_path = image_dir / perturbed_filename
    save_image(perturbed_img, str(perturbed_save_path))

    # Record 1: Original (ANSWERABLE)
    original_record = {
        "id": f"{rec_id}_clean",
        "source_id": str(rec_id),
        "image_path": str(original_save_path),
        "question": question,
        "answer": answer,
        "category": category,
        "variant": "clean",
        "perturbation_type": "none",
        "visual_cues": visual_cues,
        "cue_short_reason": record.get("cue_short_reason", ""),
        "gemini_tag": {
            "answerability": "ANSWERABLE",
            "failure_type": "none",
            "short_reason": "Original image is clear and unperturbed.",
        },
    }

    # Record 2: Perturbed (UNANSWERABLE)
    perturbed_record = {
        "id": f"{rec_id}_sam_perturbed",
        "source_id": str(rec_id),
        "original_image_path": str(original_save_path),
        "image_path": str(perturbed_save_path),
        "question": question,
        "answer": "I don't know",
        "category": category,
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

    return [original_record, perturbed_record]


def main(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Load environment ──────────────────────────────────────────
    env_path = args.env_path or DEFAULT_ENV_PATH
    load_dotenv(dotenv_path=env_path)
    log.info(f"Loaded env from: {env_path}")

    # ── Initialize clients ────────────────────────────────────────
    gemini_client = build_gemini_client()
    log.info("Gemini client initialized")

    sam_model = init_sam_model(sam_type=args.sam_type, device=args.device)
    log.info("Lang-SAM model ready")

    # ── Load input JSONL ──────────────────────────────────────────
    log.info(f"Loading input: {args.input_jsonl}")
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        all_records = [json.loads(line) for line in f if line.strip()]
    log.info(f"  {len(all_records)} total records")

    # ── Apply row range ───────────────────────────────────────────
    total = len(all_records)
    start = args.start_row if args.start_row is not None else 0
    end = args.end_row if args.end_row is not None else total
    start = max(0, min(start, total))
    end = max(start, min(end, total))
    records = all_records[start:end]
    log.info(f"Processing rows {start}–{end - 1} ({len(records)} records)")

    # ── Prepare output ────────────────────────────────────────────
    Path(args.image_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    unanswerable_count = 0
    answerable_skip_count = 0
    error_count = 0
    sam_success_count = 0
    fallback_count = 0

    with open(args.output_jsonl, "a", encoding="utf-8") as out_f:
        for local_idx, record in enumerate(records):
            rec_id = record["id"]
            log.info(f"[{local_idx + 1}/{len(records)}] Processing id={rec_id}")

            try:
                result = process_single_record(
                    record, gemini_client, sam_model, args)
            except Exception as e:
                log.error(f"[{rec_id}] Unexpected error: {e}", exc_info=True)
                error_count += 1
                continue

            if result is None:
                answerable_skip_count += 1
                continue

            # Write both records
            for rec in result:
                out_f.write(json.dumps(rec) + "\n")
            out_f.flush()

            unanswerable_count += 1
            variant = result[1].get("variant", "unknown")
            if variant == "sam_targeted":
                sam_success_count += 1
            else:
                fallback_count += 1

            components = result[1].get("masking_details", {}).get(
                "components_masked", [])
            log.info(
                f"  ✅ UNANSWERABLE #{unanswerable_count} "
                f"(id={rec_id}, variant={variant}, "
                f"components={components})"
            )

            # ── Stop early if target reached ──────────────────────
            if args.target_count > 0 and unanswerable_count >= args.target_count:
                log.info(f"🎯 Reached target of {args.target_count}. Stopping.")
                break

    # ── Summary ───────────────────────────────────────────────────
    log.info(f"\n{'=' * 60}")
    log.info(f"Pipeline Complete")
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
        description="SAM-Guided Targeted Perturbation Pipeline "
                    "for VQA Unanswerable Sample Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run
  python sam_targeted_perturbation.py \\
      --input_jsonl  /path/to/clean_vqa_with_visual_cues_tagged.jsonl \\
      --output_jsonl /path/to/unanswerable_sam.jsonl \\
      --image_dir    /path/to/output_images

  # Test with first 10 images
  python sam_targeted_perturbation.py \\
      --input_jsonl  /path/to/input.jsonl \\
      --output_jsonl /path/to/output.jsonl \\
      --image_dir    /path/to/images \\
      --start_row 0 --end_row 10
        """,
    )

    # I/O
    parser.add_argument("--input_jsonl", type=str, required=True,
                        help="Input JSONL with visual cues (clean, tagged).")
    parser.add_argument("--output_jsonl", type=str, required=True,
                        help="Output JSONL for unanswerable records.")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory to save original + perturbed images.")
    parser.add_argument("--image_root", type=str, default=DEFAULT_IMAGE_ROOT,
                        help="Root directory for source images. Overrides the "
                             "directory in image_path, keeping only filename. "
                             f"Default: {DEFAULT_IMAGE_ROOT}")

    # Model config
    parser.add_argument("--sam_type", type=str, default="sam2",
                        choices=["sam2", "vit_h", "vit_l", "vit_b"],
                        help="SAM model variant for Lang-SAM. Default: sam2")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device for SAM. Default: cuda:0")
    parser.add_argument("--env_path", type=str, default=None,
                        help=f"Path to .env file. Default: {DEFAULT_ENV_PATH}")

    # Processing
    parser.add_argument("--start_row", type=int, default=None,
                        help="First row to process (0-indexed, inclusive).")
    parser.add_argument("--end_row", type=int, default=None,
                        help="Last row to process (0-indexed, exclusive).")
    parser.add_argument("--target_count", type=int, default=0,
                        help="Stop after this many unanswerable pairs (0=all).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed. Default: 42")
    parser.add_argument("--sleep_interval", type=float, default=2.0,
                        help="Seconds between Gemini API calls. Default: 2.0")
    parser.add_argument("--feather_sigma", type=float, default=5.0,
                        help="Gaussian sigma for mask edge feathering. Default: 5.0")

    args = parser.parse_args()
    main(args)
