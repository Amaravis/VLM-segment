# refine_bbox.py

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import json
import re

import torch
from PIL import Image as PILImage
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import cv2

#    e.g. from the earlier hf_hub_download + unzip code



@dataclass
class RefineStepResult:
    step: int
    bbox: List[float]      # [x1, y1, x2, y2] in THIS image's coords (resized image)
    response: str          # answer (option / reasoning / referring)
    decision: str          # "refine" or "stop"
    reason: str            # explanation
    raw_text: str          # raw model output text


def bbox_area(b: List[float]) -> float:
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(b1: List[float], b2: List[float]) -> float:
    x1, y1, x2, y2 = b1
    x1b, y1b, x2b, y2b = b2

    ix1 = max(x1, x1b)
    iy1 = max(y1, y1b)
    ix2 = min(x2, x2b)
    iy2 = min(y2, y2b)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    union = bbox_area(b1) + bbox_area(b2) - inter
    if union <= 0:
        return 0.0
    return inter / union


def clamp_bbox_to_image(b: List[float], width: int, height: int) -> List[int]:
    """
    Clamp bbox coordinates to image bounds and ensure x2 > x1, y2 > y1.
    """
    x1, y1, x2, y2 = b
    x1 = max(0, min(int(round(x1)), width - 1))
    x2 = max(0, min(int(round(x2)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    y2 = max(0, min(int(round(y2)), height - 1))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return [x1, y1, x2, y2]


class QwenRunner:
    """
    Thin wrapper around your Qwen2.5-VL model + processor for one-sample inference.
    """

    def __init__(self, model, processor: AutoProcessor):
        self.model = model
        self.processor = processor

    def generate(self, messages, max_new_tokens: int = 512) -> str:
        """
        messages: list of {"role": "...", "content": [...]}
                  content entries: {"type": "image", "image": PIL.Image}
                                   {"type": "text",  "text":  str}
        """
        if isinstance(messages, dict):
            messages = [messages]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        out_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return out_text


def extract_answer_json(output_text: str) -> Dict:
    """
    Expect:
      <think> ... </think>
      <answer>
      { ... JSON ... }
      </answer>

    Returns the JSON object as a Python dict.
    Raises if we truly can't find any JSON.
    """
    m_answer = re.search(r"<answer>(.*?)</answer>", output_text, re.S | re.I)
    if m_answer:
        answer_block = m_answer.group(1)
    else:
        answer_block = output_text

    m_json = re.search(r"\{.*\}", answer_block, re.S)
    if not m_json:
        raise ValueError("No JSON object found in model output.")
    json_str = m_json.group(0)
    return json.loads(json_str)


def _response_hint(question_type: str, options: Optional[List[str]]) -> str:
    if question_type == "referring":
        return (
            'The "response" field should be a short phrase like '
            '"The object is found." or "The object is not found."'
        )
    elif question_type == "reasoning":
        return (
            'The "response" field should be a short word or phrase '
            "that answers the question."
        )
    elif question_type == "option":
        assert options is not None and len(options) > 0
        opt_str = ", ".join(options)
        return (
            f'The "response" field must be ONE of the options: {opt_str}. '
            "Prefer returning just the option ID (e.g. 'A') or the exact option string."
        )
    else:
        raise ValueError(f"Unknown question_type: {question_type}")


def build_initial_message(
    image: PILImage,
    question: str,
    question_type: str,
    options: Optional[List[str]] = None,
) -> List[Dict]:
    width, height = image.size
    response_hint = _response_hint(question_type, options)

    QUESTION_TEMPLATE = f"""
You are localizing the region in the image that is most relevant to the question.

  Question: "{question}"
  Image resolution: width={width}, height={height} (pixels).

Task:
1. Predict a bounding box in this image coordinate system that covers the object/region
   needed to answer the question.
2. Also answer the question.
3. Use integer pixel coordinates: 0 <= x < {width}, 0 <= y < {height}.

Output format:
- First, think step-by-step in <think>...</think>.
- Then put ONLY a single JSON object inside <answer>...</answer>.

The JSON MUST have:

<answer>
{{
  "bbox": [x_min, y_min, x_max, y_max],
  "response": "...",
  "decision": "refine",
  "reason": "short explanation of why this is a good initial box"
}}
</answer>

Where:
- "bbox" is in THIS image's pixel coordinates (no normalization).
- "decision" is always "refine" at this initial step.
- {response_hint}

Do NOT include any extra keys or text outside the JSON object.
"""

    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": QUESTION_TEMPLATE},
        ],
    }]


def build_refine_message(
    image: PILImage,
    crop: PILImage,
    question: str,
    question_type: str,
    current_bbox: List[float],
    step: int,
    max_steps: int,
    options: Optional[List[str]] = None,
) -> List[Dict]:
    width, height = image.size
    x1, y1, x2, y2 = current_bbox
    bbox_token = f"<box>({int(x1)},{int(y1)}),({int(x2)},{int(y2)})</box>"

    response_hint = _response_hint(question_type, options)

    QUESTION_TEMPLATE = f"""
You are REFINING a bounding box for the object/region relevant to this question:

  Question: "{question}"
  Image resolution: width={width}, height={height} (pixels).
  Current bbox: {bbox_token}
    which corresponds to [x_min={int(x1)}, y_min={int(y1)}, x_max={int(x2)}, y_max={int(y2)}].

You are given TWO images:
1. The full resized image.
2. A cropped view of the current bbox.

Goals at refinement step {step}/{max_steps}:
1. Decide whether to keep refining or stop.
2. If refining, slightly shift and/or shrink the bbox so it tightly covers the object
   with a small amount of background:
   - Prefer shrinking over expanding.
   - Never cut off any visible part of the object.
3. Also answer the question.

Stopping rule (conceptual):
- If the box already covers the object with about 10–20% extra background,
  and further shrinking would likely cut off the object or change area by < ~5%,
  set "decision": "stop".

Output format:
- First, think step-by-step in <think>...</think>, explicitly considering:
  - How well the box covers the object.
  - Whether to shrink, shift, or stop.
- Then put ONLY a single JSON object inside <answer>...</answer>:

<answer>
{{
  "bbox": [x_min, y_min, x_max, y_max],
  "response": "...",
  "decision": "refine" | "stop",
  "reason": "short explanation of why you chose this bbox and decision"
}}
</answer>

Where:
- "bbox" MUST use the same coordinate system as the resized image (width={width}, height={height}).
- Coordinates MUST be integers with 0 <= x < {width}, 0 <= y < {height}.
- If "decision" is "stop", "bbox" should be your final best guess.
- {response_hint}

Do NOT include any extra text or keys outside that JSON object.
"""

    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "image", "image": crop},
            {"type": "text", "text": QUESTION_TEMPLATE},
        ],
    }]


class IterativeBBoxRefiner:
    """
    Controller that runs:
      - initial bbox prediction
      - iterative refinement with original + cropped images
      - JSON-based decisions ("refine"/"stop") + geometry stopping
    """

    def __init__(
        self,
        qwen_runner: QwenRunner,
        max_steps: int = 4,
        min_rel_area_change: float = 0.03,   # 3%
        min_iou_for_convergence: float = 0.98,
    ):
        self.qwen = qwen_runner
        self.max_steps = max_steps
        self.min_rel_area_change = min_rel_area_change
        self.min_iou_for_convergence = min_iou_for_convergence

    def _run_step(
        self,
        message: List[Dict],
        step: int,
        image_size: Tuple[int, int],
        fallback_bbox: Optional[List[float]] = None,
    ) -> RefineStepResult:
        """
        Run one refinement step, but:
        - if bbox JSON is missing/invalid, DO NOT raise.
        - use fallback rules:
          * step == 0  and no fallback_bbox -> full image bbox
          * step >= 1 and fallback_bbox     -> reuse last valid bbox
        """
        width, height = image_size
        raw = self.qwen.generate(message, max_new_tokens=512)

        data: Dict = {}
        bbox = None
        response = ""
        decision = "refine"
        reason = ""

        invalid_bbox = False

        try:
            data = extract_answer_json(raw)
            bbox = data.get("bbox", None)
            if not bbox or len(bbox) != 4:
                invalid_bbox = True
        except Exception:
            invalid_bbox = True

        if invalid_bbox:
            # choose fallback bbox
            if fallback_bbox is not None:
                bbox = fallback_bbox
                reason = "Used previous valid bbox because model output was invalid."
                # safer to stop refinement if model keeps failing
                decision = "stop" if step > 0 else "refine"
            else:
                # step 0, no previous bbox: use full image
                bbox = [0, 0, width - 1, height - 1]
                reason = "Used full-image bbox because model output was invalid."
                decision = "refine"  # allow later steps to refine
            response = ""
        else:
            bbox = clamp_bbox_to_image(bbox, width, height)
            response = str(data.get("response", "")).strip()
            decision = str(data.get("decision", "refine")).strip().lower()
            if decision not in ("refine", "stop"):
                decision = "refine"
            reason = str(data.get("reason", "")).strip()

        bbox = clamp_bbox_to_image(bbox, width, height)

        return RefineStepResult(
            step=step,
            bbox=bbox,
            response=response,
            decision=decision,
            reason=reason,
            raw_text=raw,
        )

    def refine(
        self,
        image: PILImage,
        question: str,
        question_type: str,
        options: Optional[List[str]] = None,
        initial_bbox: Optional[List[float]] = None,
    ) -> List[RefineStepResult]:
        """
        Full refinement:
          - If no initial bbox: ask Qwen for it on the resized image.
          - Then iteratively refine with original+crop views (in that resized space).
          - Stop when model says "stop", geometry converges, or max_steps reached.

        All bboxes in returned RefineStepResult objects are in `image.size` coordinates.
        """
        width, height = image.size
        results: List[RefineStepResult] = []

        # step 0: initial bbox
        if initial_bbox is None:
            init_msg = build_initial_message(
                image=image,
                question=question,
                question_type=question_type,
                options=options,
            )
            r0 = self._run_step(
                init_msg,
                step=0,
                image_size=(width, height),
                fallback_bbox=None,
            )
            results.append(r0)
            current_bbox = r0.bbox
        else:
            current_bbox = clamp_bbox_to_image(initial_bbox, width, height)

        prev_bbox = current_bbox
        prev_area = bbox_area(prev_bbox)

        # refinement loop
        for step in range(1, self.max_steps + 1):
            x1, y1, x2, y2 = current_bbox
            crop = image.crop((x1, y1, x2, y2))

            refine_msg = build_refine_message(
                image=image,
                crop=crop,
                question=question,
                question_type=question_type,
                current_bbox=current_bbox,
                step=step,
                max_steps=self.max_steps,
                options=options,
            )

            r = self._run_step(
                refine_msg,
                step=step,
                image_size=(width, height),
                fallback_bbox=current_bbox,
            )
            results.append(r)

            new_bbox = r.bbox
            new_area = bbox_area(new_bbox)
            iou = bbox_iou(prev_bbox, new_bbox)
            rel_area_change = abs(new_area - prev_area) / (prev_area + 1e-6)

            should_stop_geom = (
                rel_area_change < self.min_rel_area_change
                or iou >= self.min_iou_for_convergence
            )

            if r.decision == "stop" or should_stop_geom or step == self.max_steps:
                break

            prev_bbox = new_bbox
            prev_area = new_area
            current_bbox = new_bbox

        return results



if __name__ == "__main__":
    # Where your Finers-4k images live
    IMAGES_ROOT = Path("/scratch/svaidy33/hf_cache/datasets/all_images/all_images/")
    SAVE_ROOT = Path("./refine_vis")
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    # Where to save JSON results
    RESULTS_PATH = SAVE_ROOT / "refine_results.json"

    # Load dataset (uses HF_HOME/HF_DATASETS_CACHE)
    dataset = load_dataset("Jiazuo98/Finers-4k-benchmark")
    test_ds = dataset["test"]

    print("Loading Qwen model !!")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        padding_side="left",
    )

    # Create runner + refiner ONCE
    qrunner = QwenRunner(model, processor)
    refiner = IterativeBBoxRefiner(
        qrunner,
        max_steps=5,
        min_rel_area_change=0.03,
        min_iou_for_convergence=0.98,
    )

    all_results = []  # collect sample-wise results

    for idx, example in enumerate(test_ds):
        anno = example["annotations"]

        question_type  = anno["Q-type"]
        input_question = anno["Q"]
        gt_answer      = anno["A"]
        image_name     = anno["image_path"]

        if "options" in anno.keys():
            options = anno["options"]
        else:
            options = None

        image_path = IMAGES_ROOT / image_name
        image_path = str(image_path)

        # Read image
        image_np = cv2.imread(image_path)
        if image_np is None:
            print(f"[WARN] Failed to read image: {image_path}")
            # even for failures, you could append a record if you want
            continue

        # BGR -> RGB for PIL/Qwen
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_pil = PILImage.fromarray(image_rgb)

        # Run refinement on original resolution
        results = refiner.refine(
            image=image_pil,
            question=input_question,
            question_type=question_type,  # "referring" | "reasoning" | "option"
            options=options,
            initial_bbox=None,
        )

        if not results:
            print(f"[{idx}] No refinement result for {image_name}")
            all_results.append({
                "index": idx,
                "image_name": image_name,
                "question_type": question_type,
                "question": input_question,
                "gt_answer": gt_answer,
                "steps": [],
                "final_response": None,
                "final_bbox": None,
                "vis_path": None,
            })
            continue

        print(f"[{idx}] {image_name}")

        # ---- store all steps ----
        steps_serialized = []
        for r in results:
            # clamp each step's bbox for safety, but keep original too if you like
            bx = clamp_bbox_to_image(
                r.bbox,
                width=image_pil.width,
                height=image_pil.height,
            )
            print(f"  Step {r.step}: bbox={bx}, decision={r.decision}, resp={r.response!r}")
            steps_serialized.append({
                "step": r.step,
                "bbox": [int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])],
                "response": r.response,
                "decision": r.decision,
                "reason": r.reason,
                "raw_text": r.raw_text,
            })

        # final step
        final = results[-1]
        # clamp bbox of final step
        x1, y1, x2, y2 = clamp_bbox_to_image(
            final.bbox,
            width=image_pil.width,
            height=image_pil.height,
        )
        final_bbox = [int(x1), int(y1), int(x2), int(y2)]

        print("  Final bbox (image coords):", final_bbox)
        print("  Final response:", final.response)

        # ---- draw only the final box on the image ----
        vis_img = image_np.copy()
        cv2.rectangle(
            vis_img,
            (final_bbox[0], final_bbox[1]),
            (final_bbox[2], final_bbox[3]),
            (0, 0, 255),  # red in BGR
            3,
        )
        cv2.putText(
            vis_img,
            f"{idx}",
            (final_bbox[0], max(final_bbox[1] - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        stem = Path(image_name).stem
        out_path = SAVE_ROOT / f"{idx:05d}_{stem}_bbox.jpg"
        cv2.imwrite(str(out_path), vis_img)
        print("  Saved vis to:", out_path)

        # ---- store structured result (all steps + final) ----
        all_results.append({
            "index": idx,
            "image_name": image_name,
            "question_type": question_type,
            "question": input_question,
            "gt_answer": gt_answer,
            "steps": steps_serialized,
            "final_response": final.response,
            "final_bbox": final_bbox,
            "vis_path": str(out_path),
        })

        # Optional: limit for quick debugging
        # if idx >= 49:
        #     break

    # Write all results to JSON file
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved {len(all_results)} samples with full refinement steps to {RESULTS_PATH}")
