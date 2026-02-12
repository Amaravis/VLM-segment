# refine_bbox.py

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import json
import re
import ast

import torch
from PIL import Image as PILImage
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from pathlib import Path
import os
import cv2

# ----------------- config knobs for compute -----------------
MAX_STEPS = 4              # number of refinement steps used in the class default
# ------------------------------------------------------------
RESIZE_W = 1920
RESIZE_H = 1080
MAX_NEW_TOKENS = 256       # allow room for think + JSON at each step


@dataclass
class RefineStepResult:
    step: int
    bbox: List[float]                 # [x1, y1, x2, y2] in THIS image's coords (full image)
    points: Optional[List[List[float]]]  # [[x1, y1], [x2, y2]] in same coords
    response: str                     # answer (option / reasoning / referring)
    decision: str                     # "refine" or "stop"
    reason: str                       # explanation from JSON
    raw_text: str                     # raw model output text (think + json)
    think: Optional[str] = None       # extracted "think" reasoning block


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


def default_points_from_bbox(b: List[float]) -> List[List[float]]:
    """
    Heuristic fallback: generate two points inside bbox.
    - One at the center.
    - One at the upper-left quarter of the box.
    """
    x1, y1, x2, y2 = b
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    px2 = 0.5 * (x1 + cx)
    py2 = 0.5 * (y1 + cy)
    return [[cx, cy], [px2, py2]]


def clamp_points_to_image(
    points: Optional[List[List[float]]],
    width: int,
    height: int,
) -> Optional[List[List[int]]]:
    if points is None:
        return None
    clamped: List[List[int]] = []
    for p in points:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            continue
        x, y = p
        xi = max(0, min(int(round(float(x))), width - 1))
        yi = max(0, min(int(round(float(y))), height - 1))
        clamped.append([xi, yi])
    return clamped or None


class QwenRunner:
    """
    Thin wrapper around your Qwen2.5-VL model + processor for one-sample inference.
    """

    def __init__(self, model, processor: AutoProcessor):
        self.model = model
        self.processor = processor

    def generate(self, messages, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        """
        messages: dict or list of {"role": "...", "content": [...]}
                  content entries: {"type": "image", "image": PIL.Image}
                                   {"type": "text",  "text":  str}
        """
        # Normalize to list for apply_chat_template / process_vision_info
        if isinstance(messages, dict):
            messages = [messages]

        # Build chat text
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Extract visual inputs
        image_inputs, video_inputs = process_vision_info(messages)

        device = next(self.model.parameters()).device

        # Prepare inputs (batch size = 1)
        inputs = self.processor(
            text=[text],          # list of length 1
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        input_ids = inputs.input_ids  # shape: [1, seq_len]

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )  # shape: [1, seq_len + new_len]

        # Take only the newly generated tokens for this single example
        new_tokens = generated_ids[0, input_ids.shape[1]:]  # 1D tensor

        # Decode a single sequence
        out_text = self.processor.decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        print(out_text)
        return out_text


# -------------------- robust answer JSON / think extraction -------------------- #

def _strip_code_fences(text: str) -> str:
    """
    Remove markdown ``` fences but KEEP the content inside.
    Works for ```json ... ``` and plain ``` ... ```.
    """
    text = text.strip()
    text = text.replace("```", "")
    return text.strip()


def _balance_brackets(s: str) -> str:
    """
    Very simple bracket balancer for { } and [ ].
    Heuristic but usually OK for model-generated JSON without
    literal '{' or '[' inside strings.
    """
    opens = {'{': '}', '[': ']'}
    closes = {'}': '{', ']': '['}
    stack = []
    out = []

    for ch in s:
        if ch in opens:
            stack.append(ch)
            out.append(ch)
        elif ch in closes:
            if stack and stack[-1] == closes[ch]:
                stack.pop()
                out.append(ch)
            else:
                # stray closing, ignore
                continue
        else:
            out.append(ch)

    while stack:
        op = stack.pop()
        out.append(opens[op])

    return "".join(out)


def _extract_think_block(header: str) -> Optional[str]:
    """
    Extract the free-form 'think' block that appears before the JSON.

    Expected / encouraged pattern:

        think:
        ...some text...

        json:
        {
           ...
        }
    """
    if not header:
        return None

    s = header.strip()
    if not s:
        return None

    lower = s.lower()

    # If there's an explicit 'json:' label, drop everything from there on.
    json_pos = lower.rfind("json:")
    if json_pos != -1:
        s = s[:json_pos].rstrip()
        lower = s.lower()

    # If there's an explicit 'think:' label, drop it.
    think_pos = lower.find("think:")
    if think_pos != -1:
        s = s[think_pos + len("think:"):].strip()

    return s or None


def extract_think_and_answer_json(output_text: str) -> Tuple[Optional[str], Dict]:
    """
    Format:

      think:
      <free-form reasoning>

      json:
      {
        "bbox_2d": [x_min, y_min, x_max, y_max],
        "points_2d": [[px1, py1], [px2, py2]],
        "response": "...",
        "decision": "refine" | "stop",
        "reason": "..."
      }

    Strategy:
      1) Strip code fences.
      2) Find first '{' and last '}'.
      3) Interpret text before '{' as a possible 'think' block.
      4) Balance brackets and parse the JSON.
    """
    text = _strip_code_fences(output_text)

    open_idx = text.find('{')
    close_idx = text.rfind('}')
    if open_idx == -1 or close_idx == -1 or close_idx <= open_idx:
        raise ValueError("No JSON-like braces found in model output.")

    # Everything before the first '{' is potential 'think:' + 'json:' header.
    header = text[:open_idx]
    think_block = _extract_think_block(header)

    candidate = text[open_idx:close_idx + 1].strip()
    candidate = _balance_brackets(candidate)

    # Try strict JSON first
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return think_block, obj
    except Exception:
        pass

    # Fallback: Python dict style (single quotes, etc.)
    try:
        obj = ast.literal_eval(candidate)
        if isinstance(obj, dict):
            return think_block, obj
    except Exception as e:
        raise ValueError(f"Failed to parse answer JSON: {e}")

    raise ValueError("Failed to parse answer JSON for unknown reasons.")


def extract_answer_json(output_text: str) -> Dict:
    """
    Backwards-compatible helper that returns only the JSON part.
    If you also want the 'think' block, use `extract_think_and_answer_json`.
    """
    _, obj = extract_think_and_answer_json(output_text)
    return obj


# -------------------- prompt helpers -------------------- #

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
        # Just show the options list directly
        return (
            f"This is a multiple-choice question with options: {options}. "
            'The "response" field must indicate the correct option (e.g. "A", "B", "C") '
            "or the exact option text."
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

    # Show options inline for option questions (MVQA-style)
    options_block = ""
    if question_type == "option" and options is not None:
        options_block = f"\nOptions: {options}\n"

    # Task-aware example question, response, and think (from your figure)
    if question_type == "reasoning":
        # OVQA-style
        example_question = "What is the shape of the trash can placed on the overpass?"
        example_response_value = "cylindrical."
        example_think = (
            "1. Identify the overpass in the image.\n"
            "2. Locate the trash can on the overpass.\n"
            "3. Focus on the region around the trash can.\n"
            "4. Determine the shape of the trash can by inspecting its outline.\n"
            "5. The trash can appears to be cylindrical based on its round shape."
        )
    elif question_type == "option":
        # MVQA-style
        example_question = (
            "What color shirt is the person wearing who is wearing a hat? "
            "(A) White  (B) Blue  (C) Red  (D) Green."
        )
        example_response_value = "A"
        example_think = (
            "1. First, identify the person wearing a hat in the image. The person "
            "is near the truck.\n"
            "2. Check the color of this person's shirt; it appears white.\n"
            "3. Compare the shirt color with the given options (A) White, "
            "(B) Blue, (C) Red, (D) Green.\n"
            "4. The shirt color matches option (A) White.\n"
            "5. The relevant region to focus on is around the person near the truck, "
            "specifically the upper part of the image where the person is standing."
        )
    else:  # referring / IS-style
        example_question = "The green sign on the left side of the large indicator board."
        example_response_value = "The object is found."
        example_think = (
            "1. Identify the large indicator board in the image.\n"
            "2. Locate the green sign on the left side of the large indicator board.\n"
            "3. Determine the region around the green sign.\n"
            "4. Crop a box that tightly surrounds the green sign.\n"
            "5. Check if the object (green sign) is fully within the bounding box; "
            "if so, the object is found."
        )

    QUESTION_TEMPLATE = f"""
You are localizing the region in the image that is most relevant to the question.

  Question: "{question}"
  Image resolution: width={width}, height={height} (pixels).
{options_block}Task:
1. Predict a bounding box in this image coordinate system that covers the object/region
   needed to answer the question.
2. Also answer the question itself.
3. Additionally, choose TWO points inside the object/region that are useful for
   segmentation (for example, the center and another salient point on the object).
4. Use integer pixel coordinates: 0 <= x < {width}, 0 <= y < {height}.

Output format (VERY IMPORTANT, MUST FOLLOW EXACTLY):

1. First, write a short reasoning section starting with the line:
   think:
   On the following line(s), describe your thought process in a few numbered steps,
   noticing objects, their shapes, colors, clothes, positions, and how they match
   the question.

2. Then write a JSON section starting with the line:
   json:
   Immediately after this line, output a single valid JSON object.

The JSON object must have exactly these keys:
  "bbox_2d", "points_2d", "response", "decision", "reason".

The JSON schema (illustrative):

json:
{{
  "bbox_2d": [x_min, y_min, x_max, y_max],
  "points_2d": [[px1, py1], [px2, py2]],
  "response": "...",
  "decision": "refine",
  "reason": "short explanation of why this is a good initial box and point choice"
}}

Where:
- "bbox_2d" is in THIS image's pixel coordinates (no normalization).
- "bbox_2d" coordinates MUST be integers.
- "points_2d" MUST be a list of exactly two points [[px1, py1], [px2, py2]] inside
  the object/region (also in this image's coordinates).
- "decision" is always "refine" at this initial step.
- {response_hint}

Example (FORMAT ONLY, not the real answer):

Example question (for illustration only):
{example_question}

think:
{example_think}

json:
{{
  "bbox_2d": [10, 20, 100, 200],
  "points_2d": [[40, 60], [80, 140]],
  "response": "{example_response_value}",
  "decision": "refine",
  "reason": "The box tightly covers the most relevant region with a small margin, "
            "and both points are inside the target object."
}}
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
    question: str,
    question_type: str,
    current_bbox: List[float],
    step: int,
    max_steps: int,
    options: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Refinement step:
    - Uses the SAME full image at each step (global context).
    - Bbox is updated in the same coordinate system.
    """
    width, height = image.size
    x1, y1, x2, y2 = current_bbox
    bbox_token = f"<box>({int(x1)},{int(y1)}),({int(x2)},{int(y2)})</box>"
    response_hint = _response_hint(question_type, options)

    options_block = ""
    if question_type == "option" and options is not None:
        options_block = f"\nOptions: {options}\n"

    # Task-aware example question, think, and response for refinement
    if question_type == "reasoning":
        example_question = "What is the shape of the trash can placed on the overpass?"
        example_response_value = "cylindrical."
        example_think = (
            "1. I check the current box and see that it includes the trash can on "
            "the overpass but also extra background.\n"
            "2. I shift and shrink the box so it more tightly surrounds the trash "
            "can while keeping it fully inside.\n"
            "3. In the refined box, the outline of the trash can is clear.\n"
            "4. Its body looks like a round tube, so its shape is cylindrical.\n"
            "5. Since the box is now tight around the trash can, further shrinking "
            "would risk cutting it off, so I stop refining."
        )
    elif question_type == "option":
        example_question = (
            "What color shirt is the person wearing who is wearing a hat? "
            "(A) White  (B) Blue  (C) Red  (D) Green."
        )
        example_response_value = "A"
        example_think = (
            "1. I inspect the current box and confirm it contains the person wearing "
            "a hat near the truck, but also some extra background.\n"
            "2. I look at the shirt inside the box and see that it is white.\n"
            "3. I compare this with the options (A) White, (B) Blue, (C) Red, "
            "(D) Green and it still matches option (A) White.\n"
            "4. I tighten the box so it focuses on the person’s upper body and hat, "
            "reducing unnecessary background.\n"
            "5. The refined box clearly shows the person with the white shirt, so I "
            "keep option A as the answer and stop refining."
        )
    else:  # referring
        example_question = "The green sign on the left side of the large indicator board."
        example_response_value = "The object is found."
        example_think = (
            "1. I look at the current box and see that it contains the large "
            "indicator board and part of the area around it.\n"
            "2. On the left side of the board inside the box, I can see the green "
            "sign described in the question.\n"
            "3. I shrink the box so it more tightly surrounds the green sign while "
            "keeping the entire sign inside.\n"
            "4. I verify that no important part of the green sign is cut off.\n"
            "5. The refined box clearly contains the green sign, so the object is "
            "found and I stop refining."
        )

    QUESTION_TEMPLATE = f"""
You are REFINING a bounding box for the object/region relevant to this question.

  Question: "{question}"
  Image resolution: width={width}, height={height} (pixels).
  Current bbox_2d: {bbox_token}
{options_block}You always see the full image, so you must reason using GLOBAL context,
but treat the current bbox as your current best guess region.

Goals at refinement step {step}/{max_steps}:
1. Decide whether to keep refining or stop.
2. If refining, slightly shift and/or shrink the bbox so it tightly covers the object
   with a small amount of background:
   - Prefer shrinking over expanding.
   - Never cut off any visible part of the object.
3. Also answer the question.
4. Choose TWO points inside the refined object/region that are helpful for
   segmentation (for example, one near the center and one on another part of the
   object).

Stopping rule (conceptual):
- If the box already covers the object with about 10–20% extra background,
  and further shrinking would likely cut off the object or change area by < ~5%,
  set "decision": "stop".

Output format (VERY IMPORTANT, MUST FOLLOW EXACTLY):

1. First, write a short reasoning section starting with the line:
   think:
   On the following line(s), describe your reasoning in a few concrete steps:
   what you see inside the current box, how you adjust it (if needed), how you
   pick the points, and how this supports your final answer.

2. Then write a JSON section starting with the line:
   json:
   Immediately after this line, output a single valid JSON object.

The JSON object must have exactly these keys:
  "bbox_2d", "points_2d", "response", "decision", "reason".

The JSON schema (illustrative):

json:
{{
  "bbox_2d": [x_min, y_min, x_max, y_max],
  "points_2d": [[px1, py1], [px2, py2]],
  "response": "...",
  "decision": "refine" | "stop",
  "reason": "short explanation of why you chose this bbox, these points, and decision"
}}

Where:
- "bbox_2d" MUST use the same coordinate system as this image (width={width}, height={height}).
- Coordinates MUST be integers with 0 <= x < {width}, 0 <= y < {height}.
- "points_2d" MUST be two integer points [[px1, py1], [px2, py2]] inside the
  chosen box and object.
- If "decision" is "stop", "bbox_2d" should be your final best guess.
- {response_hint}

Example (FORMAT ONLY, not the real answer):

Example question (for illustration only):
{example_question}

think:
{example_think}

json:
{{
  "bbox_2d": [12, 22, 98, 195],
  "points_2d": [[30, 60], [70, 160]],
  "response": "{example_response_value}",
  "decision": "stop",
  "reason": "The box is tight around the most relevant region and the points are "
            "placed on the object for segmentation."
}}
"""

    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": QUESTION_TEMPLATE},
        ],
    }]


class IterativeBBoxRefiner:
    """
    Controller that runs:
      - initial bbox + points prediction
      - iterative refinement on the SAME full image
      - JSON-based decisions ("refine"/"stop") + geometry stopping
    """

    def __init__(
        self,
        qwen_runner: QwenRunner,
        max_steps: int = MAX_STEPS,
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
        - points_2d are also parsed; if missing or invalid, we fall back to
          simple points derived from the bbox.
        """
        width, height = image_size
        raw = self.qwen.generate(message, max_new_tokens=MAX_NEW_TOKENS)

        bbox = None
        points: Optional[List[List[float]]] = None
        response = ""
        decision = "refine"
        reason = ""
        invalid_bbox = False
        think_text: Optional[str] = None

        try:
            think_text, data = extract_think_and_answer_json(raw)
            bbox = data.get("bbox_2d", data.get("bbox", None))
            points = data.get("points_2d", data.get("points", None))

            # Parse points if present
            if points is not None:
                if (
                    isinstance(points, (list, tuple))
                    and len(points) == 2
                    and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in points)
                ):
                    points = [[float(p[0]), float(p[1])] for p in points]
                else:
                    points = None

            # Parse bbox
            if (
                not isinstance(bbox, (list, tuple)) or
                len(bbox) != 4
            ):
                invalid_bbox = True
            else:
                bbox = [float(v) for v in bbox]
                response = str(data.get("response", "")).strip()
                decision = str(data.get("decision", "refine")).strip().lower()
                if decision not in ("refine", "stop"):
                    decision = "refine"
                reason = str(data.get("reason", "")).strip()
        except Exception:
            invalid_bbox = True
            think_text = None

        if invalid_bbox:
            if fallback_bbox is not None:
                bbox = fallback_bbox
                reason = "Used previous valid bbox because model output was invalid."
                decision = "stop" if step > 0 else "refine"
            else:
                bbox = [0, 0, width - 1, height - 1]
                reason = "Used full-image bbox because model output was invalid."
                decision = "refine"
            response = ""
            points = None

        # Clamp bbox and points (with fallback points if needed)
        bbox = clamp_bbox_to_image(bbox, width, height)

        if points is None:
            points = default_points_from_bbox(bbox)

        points = clamp_points_to_image(points, width, height)

        return RefineStepResult(
            step=step,
            bbox=bbox,
            points=points,
            response=response,
            decision=decision,
            reason=reason,
            raw_text=raw,
            think=think_text,
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
          - If no initial bbox: ask Qwen for it on the full image.
          - Then iteratively refine using the SAME full image.
          - Stop when model says "stop", geometry converges, or max_steps reached.

        All bboxes/points in returned RefineStepResult objects are in `image.size`
        coordinates (i.e., resized coords if you pass a resized image).
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
            refine_msg = build_refine_message(
                image=image,
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


# ------------------------ script entry (example, optional) ------------------------
"""
if __name__ == "__main__":
    # Where your Finers-4k images live
    IMAGES_ROOT = Path("/content/drive/MyDrive/dataset/all_images/all_images/")
    SAVE_ROOT = Path("/content/drive/MyDrive/Projects/Fines/refine_vis")
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
        attn_implementation="eager",
        device_map="auto",
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
        if idx >= 10:
            break
        anno = example["annotations"]

        question_type  = anno["Q-type"]      # "referring" | "reasoning" | "option"
        input_question = anno["Q"]
        gt_answer      = anno["A"]
        image_name     = anno["image_path"]

        if "options" in anno.keys():
            options = anno["options"]
        else:
            options = None

        image_path = IMAGES_ROOT / image_name
        image_path = str(image_path)

        # ---- read ORIGINAL image (for final coords & visualization) ----
        image_np = cv2.imread(image_path)
        if image_np is None:
            print(f"[WARN] Failed to read image: {image_path}")
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

        # BGR -> RGB for PIL/Qwen
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]

        # ---- make a 1920x1080 resized image for Qwen ----
        image_resized_rgb = cv2.resize(
            image_rgb,
            (RESIZE_W, RESIZE_H),            # (width, height)
            interpolation=cv2.INTER_AREA,
        )

        # factors to map bbox/points from resized -> original
        sx = float(orig_w) / float(RESIZE_W)
        sy = float(orig_h) / float(RESIZE_H)

        image_pil_resized = PILImage.fromarray(image_resized_rgb)

        # ---- run refinement on the RESIZED image ----
        results = refiner.refine(
            image=image_pil_resized,
            question=input_question,
            question_type=question_type,
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

        # ---- store all steps (resized + original coords) ----
        steps_serialized = []
        for r in results:
            # clamp bbox in resized coords
            bx_resized = clamp_bbox_to_image(
                r.bbox,
                width=image_pil_resized.width,
                height=image_pil_resized.height,
            )
            x1_r, y1_r, x2_r, y2_r = bx_resized

            # map bbox to original coords
            bbox_orig = [
                int(round(x1_r * sx)),
                int(round(y1_r * sy)),
                int(round(x2_r * sx)),
                int(round(y2_r * sy)),
            ]
            bbox_orig = clamp_bbox_to_image(bbox_orig, orig_w, orig_h)

            # map points to original coords
            points_resized = r.points or []
            points_orig = []
            for px, py in points_resized:
                px_o = int(round(px * sx))
                py_o = int(round(py * sy))
                px_o = max(0, min(px_o, orig_w - 1))
                py_o = max(0, min(py_o, orig_h - 1))
                points_orig.append([px_o, py_o])

            print(
                f"  Step {r.step}: "
                f"bbox_resized={bx_resized}, "
                f"bbox_orig={bbox_orig}, "
                f"points_resized={points_resized}, "
                f"points_orig={points_orig}, "
                f"decision={r.decision}, "
                f"resp={r.response!r}"
            )

            steps_serialized.append({
                "step": r.step,
                "bbox_resized": [
                    int(bx_resized[0]),
                    int(bx_resized[1]),
                    int(bx_resized[2]),
                    int(bx_resized[3]),
                ],
                "bbox_orig": bbox_orig,
                "points_resized": points_resized,
                "points_orig": points_orig,
                "response": r.response,
                "decision": r.decision,
                "reason": r.reason,
                "raw_text": r.raw_text,
                "think": r.think,
            })

        # ---- final step -> original coords ----
        final = results[-1]
        fx1_r, fy1_r, fx2_r, fy2_r = clamp_bbox_to_image(
            final.bbox,
            width=image_pil_resized.width,
            height=image_pil_resized.height,
        )
        final_bbox_orig = [
            int(round(fx1_r * sx)),
            int(round(fy1_r * sy)),
            int(round(fx2_r * sx)),
            int(round(fy2_r * sy)),
        ]
        final_bbox_orig = clamp_bbox_to_image(final_bbox_orig, orig_w, orig_h)

        final_points_orig = []
        for px, py in (final.points or []):
            px_o = int(round(px * sx))
            py_o = int(round(py * sy))
            px_o = max(0, min(px_o, orig_w - 1))
            py_o = max(0, min(py_o, orig_h - 1))
            final_points_orig.append([px_o, py_o])

        print("  Final bbox (orig coords):", final_bbox_orig)
        print("  Final points (orig coords):", final_points_orig)
        print("  Final response:", final.response)

        # ---- draw final box on ORIGINAL image and save ----
        vis_img = image_np.copy()
        cv2.rectangle(
            vis_img,
            (final_bbox_orig[0], final_bbox_orig[1]),
            (final_bbox_orig[2], final_bbox_orig[3]),
            (0, 0, 255),  # red in BGR
            3,
        )
        cv2.putText(
            vis_img,
            f"{idx}",
            (final_bbox_orig[0], max(final_bbox_orig[1] - 10, 0)),
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
            "final_bbox": final_bbox_orig,
            "final_points": final_points_orig,
            "vis_path": str(out_path),
        })

    # Write all results to JSON file
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved {len(all_results)} samples with full refinement steps to {RESULTS_PATH}")
"""

import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from datasets import load_dataset
from tqdm import tqdm
import pdb
import os
from PIL import Image as PILImage, Image
import re
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import cv2
import pycocotools
from loguru import logger
import ast
import difflib
import time
import sys
from pathlib import Path

# import the refiner pieces
#from refine_bbox import (
#    QwenRunner,
#    IterativeBBoxRefiner,
#    clamp_bbox_to_image,
#    MAX_STEPS,
#)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--cascade_reasoning_model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")  # unused now
    parser.add_argument("--segmentation_model_path", type=str, default="/scratch/svaidy33/Fines/checkpoints/sam2-hiera-large/sam2_hiera_large.pt")
    parser.add_argument("--segmentation_config_path", type=str, default="sam2_hiera_l")
    parser.add_argument("--test_json_path", type=str, default="the path of the json file of the test data")
    parser.add_argument('--resize_size', type=str, default="(1920, 1080)",
                        help='Resize image to a tuple string like "(1920, 1080)"')
    parser.add_argument('--cascade_resize_size', type=str, default="(512, 512)",
                        help='(unused now, kept for compat)')
    parser.add_argument("--image_path", type=str,
                        default="/scratch/svaidy33/hf_cache/datasets/all_images/all_images/")
    parser.add_argument("--save_results", action="store_true", default=True)
    parser.add_argument("--dynamic_box", action="store_true", default=False)  # unused now
    parser.add_argument("--save_path", default="/scratch/svaidy33/Fines/logs", type=str)
    parser.add_argument(
        "--qa_stage",
        default="stage2",
        type=str,
        choices=["stage1", "stage2", "no_qa"],
        help="qa behavior; for refine we treat stage1/stage2 the same",
    )

    # In Colab / notebooks, sys.argv has junk — ignore it and use defaults.
    # In normal CLI usage, parse real command-line args.
    if "ipykernel" in sys.argv[0] or "colab" in sys.argv[0]:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    return args


# ----------------- original utility code (mostly unchanged) ----------------- #

def is_inside(small_box, large_box):
    sx_min, sy_min, sx_max, sy_max = small_box
    lx_min, ly_min, lx_max, ly_max = large_box

    return (
        sx_min >= lx_min and
        sy_min >= ly_min and
        sx_max <= lx_max and
        sy_max <= ly_max
    )


def get_bbox_from_mask(mask, width, height):
    y_coords, x_coords = np.nonzero(mask)
    x_min = x_coords.min()
    x_max = x_coords.max()
    y_min = y_coords.min()
    y_max = y_coords.max()
    return (x_min, y_min, x_max, y_max)


def load_data_from_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    annotations = data["annotations"]
    return annotations


def get_mask_from_points(anno, img):
    height, width = img.shape[:2]
    points = anno["points"]
    label_value = 1  # target

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
    cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)
    return mask


def intersectionAndUnionCPU(output, target, K, ignore_index=255):
    assert output.shape == target.shape, "output and target must have the same shape"
    mask = target != ignore_index
    output = output[mask]
    target = target[mask]

    intersection = output[output == target]
    area_intersection = np.bincount(intersection, minlength=K)
    area_output = np.bincount(output, minlength=K)
    area_target = np.bincount(target, minlength=K)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def draw_bbox(image, cropped_image, bbox, gt_bbox, text,
              cropped_mask, gt_mask, restored_mask,
              output_path, image_name, data_type,
              hr_box=None, restored_box=None,
              color=(0, 0, 255), text_color=(0, 0, 255),
              thickness=4, font=cv2.FONT_HERSHEY_SIMPLEX,
              font_scale=1, font_thickness=2):

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_name = image_name.split(".")[0]
    output_path = os.path.join(output_path, image_name)
    output_path = os.path.join(output_path, data_type)
    os.makedirs(output_path, exist_ok=True)

    if restored_mask is not None:
        color_mask = np.zeros_like(image, dtype=np.uint8)
        color_mask[:, :] = (0, 0, 255)
        mask_indices = restored_mask > 0
        image[mask_indices] = cv2.addWeighted(image, 0.5, color_mask, 0.5, 0)[mask_indices]
        file_path = os.path.join(output_path, "visual_only_mask.jpg")
        cv2.imwrite(file_path, image)

    x_min, y_min, x_max, y_max = map(int, bbox)

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x = x_min + (x_max - x_min - text_width) // 2
    text_y = y_min - 10
    if text_y - text_height - baseline < 0:
        text_y = y_min + text_height + baseline + 10

    bg_x1 = text_x
    bg_y1 = text_y - text_height - baseline
    bg_x2 = text_x + text_width
    bg_y2 = text_y + baseline
    cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)

    cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, font_thickness)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

    if restored_box is not None:
        restored_x_min, restored_y_min, restored_x_max, restored_y_max = map(int, restored_box)
        cv2.rectangle(image, (restored_x_min, restored_y_min), (restored_x_max, restored_y_max), color, thickness)

    if hr_box is not None:
        hr_x_min, hr_y_min, hr_x_max, hr_y_max = map(int, hr_box)
        cv2.rectangle(image, (hr_x_min, hr_y_min), (hr_x_max, hr_y_max), (255, 0, 0), thickness)

    if gt_bbox is not None:
        gt_xmin, gt_ymin, gt_xmax, gt_ymax = map(int, gt_bbox)
        cv2.rectangle(image, (gt_xmin, gt_ymin), (gt_xmax, gt_ymax), (0, 255, 0), thickness)
        cv2.putText(image, "GT", (gt_xmin, gt_ymin - 10), font, font_scale, (0, 255, 0), font_thickness)

    file_path = os.path.join(output_path, "ori_imag_with_crop_box.jpg")
    cv2.imwrite(file_path, image)

    if gt_mask is not None:
        color_mask = np.zeros_like(image, dtype=np.uint8)
        color_mask[:, :] = (0, 255, 0)
        mask_indices = gt_mask > 0
        image[mask_indices] = cv2.addWeighted(image, 0.5, color_mask, 0.5, 0)[mask_indices]
        file_path = os.path.join(output_path, "ori_imag_with_gt_mask.jpg")
        cv2.imwrite(file_path, image)

    if cropped_image is not None and cropped_mask is not None:
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
        color_mask = np.zeros_like(cropped_image, dtype=np.uint8)
        color_mask[:, :] = (0, 0, 255)
        mask_indices = cropped_mask > 0
        cropped_image[mask_indices] = cv2.addWeighted(cropped_image, 0.5, color_mask, 0.5, 0)[mask_indices]
        file_path = os.path.join(output_path, "crop_img_with_pred_mask.jpg")
        cv2.imwrite(file_path, cropped_image)

    if restored_mask is not None:
        color_mask = np.zeros_like(image, dtype=np.uint8)
        color_mask[:, :] = (0, 0, 255)
        mask_indices = restored_mask > 0
        image[mask_indices] = cv2.addWeighted(image, 0.5, color_mask, 0.5, 0)[mask_indices]
        file_path = os.path.join(output_path, "ori_imag_with_restored_mask.jpg")
        cv2.imwrite(file_path, image)


# ----------------- main evaluation ----------------- #

def main():
    args = parse_args()
    logger.add("sample.log")
    logger.info("Loading Qwen model !!")
    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.reasoning_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation = "flash_attention_2",
        device_map="auto",
    )
    reasoning_model.eval()

    logger.info("Loading SAM2 model !!")
    segmentation_model = SAM2ImagePredictor(build_sam2(args.segmentation_config_path, args.segmentation_model_path))

    processor = AutoProcessor.from_pretrained(args.reasoning_model_path, padding_side="left")
    processor.tokenizer.padding_side = "left"

    # refine controller
    qrunner = QwenRunner(reasoning_model, processor)
    refiner = IterativeBBoxRefiner(
        qrunner,
        max_steps=MAX_STEPS,
        min_rel_area_change=0.03,
        min_iou_for_convergence=0.98,
    )

    logger.info("Loading Annotations !!")

    # If you want to use HF dataset directly, ignore test_json_path.
    # Otherwise, you can swap in load_data_from_json(args.test_json_path).
    dataset = load_dataset("Jiazuo98/Finers-4k-benchmark")
    test_ds = dataset["validation"]

    logger.info("Q-A Stage:{}".format(args.qa_stage))

    option_correct = 0
    option_num = 0
    reasoning_correct = 0
    reasoning_num = 0
    intersection_meter = []
    union_meter = []
    acc_iou_meter = []

    seg_metric_dict = {
        "s": {"intersection_meter": [], "union_meter": [], "acc_iou_meter": []},
        "xs": {"intersection_meter": [], "union_meter": [], "acc_iou_meter": []},
        "xxs": {"intersection_meter": [], "union_meter": [], "acc_iou_meter": []},
        "all": {"intersection_meter": [], "union_meter": [], "acc_iou_meter": []},
    }
    qa_metric_dict = {
        "option": {
            "colors": {"num": 0, "correct": 0},
            "shape": {"num": 0, "correct": 0},
            "others": {"num": 0, "correct": 0},
            "position": {"num": 0, "correct": 0},
            "avg": {"num": 0, "correct": 0},
        },
        "reasoning": {
            "colors": {"num": 0, "correct": 0},
            "shape": {"num": 0, "correct": 0},
            "others": {"num": 0, "correct": 0},
            "position": {"num": 0, "correct": 0},
            "avg": {"num": 0, "correct": 0},
        },
    }

    def update_seg_metric(seg_metric_dict, areas, intersection, union, acc_iou):
        XXS_TH = 0.017
        S_TH = 0.055
        if areas > S_TH:
            key = "s"
        elif areas < XXS_TH:
            key = "xxs"
        else:
            key = "xs"
        seg_metric_dict[key]["intersection_meter"].append(intersection)
        seg_metric_dict[key]["union_meter"].append(union)
        seg_metric_dict[key]["acc_iou_meter"].append(acc_iou)

        seg_metric_dict["all"]["intersection_meter"].append(intersection)
        seg_metric_dict["all"]["union_meter"].append(union)
        seg_metric_dict["all"]["acc_iou_meter"].append(acc_iou)
        return seg_metric_dict

    def update_qa_metric_correct(qa_metric_dict, data_type, attribute):
        qa_metric_dict[data_type][attribute]["correct"] += 1
        qa_metric_dict[data_type]["avg"]["correct"] += 1
        return qa_metric_dict

    def update_qa_metric_num(qa_metric_dict, data_type, attribute):
        qa_metric_dict[data_type][attribute]["num"] += 1
        return qa_metric_dict

    def final_metric(qa_metric_dict, seg_metric_dict, reasoning_num, option_num):
        qa_metric_dict["option"]["avg"]["num"] = option_num
        qa_metric_dict["reasoning"]["avg"]["num"] = reasoning_num

        for data_type in qa_metric_dict.keys():
            for attribute in qa_metric_dict[data_type].keys():
                qa_metric_dict[data_type][attribute]["acc"] = float(
                    qa_metric_dict[data_type][attribute]["correct"]
                    / (qa_metric_dict[data_type][attribute]["num"] + 1e-10)
                )

        for scale in seg_metric_dict.keys():
            iou_class = sum(seg_metric_dict[scale]["intersection_meter"]) / (
                sum(seg_metric_dict[scale]["union_meter"]) + 1e-10
            )
            acc_iou_meter_sum = sum(seg_metric_dict[scale]["acc_iou_meter"])
            seg_metric_dict[scale]["giou"] = acc_iou_meter_sum / len(seg_metric_dict[scale]["acc_iou_meter"])
            seg_metric_dict[scale]["ciou"] = iou_class[1]

        return qa_metric_dict, seg_metric_dict

    def save_metric(qa_metric_dict, seg_metric_dict, save_path):
        collected_metric = {"seg": {"s": {}, "xs": {}, "xxs": {}, "all": {}}}

        collected_metric["seg"]["s"]["giou"] = float(seg_metric_dict["s"]["giou"][1])
        collected_metric["seg"]["xs"]["giou"] = float(seg_metric_dict["xs"]["giou"][1])
        collected_metric["seg"]["xxs"]["giou"] = float(seg_metric_dict["xxs"]["giou"][1])
        collected_metric["seg"]["all"]["giou"] = float(seg_metric_dict["all"]["giou"][1])

        collected_metric["seg"]["s"]["ciou"] = float(seg_metric_dict["s"]["ciou"])
        collected_metric["seg"]["xs"]["ciou"] = float(seg_metric_dict["xs"]["ciou"])
        collected_metric["seg"]["xxs"]["ciou"] = float(seg_metric_dict["xxs"]["ciou"])
        collected_metric["seg"]["all"]["ciou"] = float(seg_metric_dict["all"]["ciou"])

        collected_metric["qa"] = qa_metric_dict

        with open(os.path.join(save_path, "results_all.json"), "w") as f:
            json.dump(collected_metric, f, indent=4)
        logger.info(collected_metric)

    box_is_valid_num = 0
    all_num = 0

    for idx, example in enumerate(test_ds):
        if idx >= 300:
            break

        anno = example["annotations"]
        question_type = anno["Q-type"]
        input_question = anno["Q"]
        gt_answer = anno["A"]
        mask_points = anno["points"]
        image_name = anno["image_path"]
        options = anno.get("options", None)
        attribute = anno["attribute"]
        area_percent = anno["area_percent"]

        image_path = os.path.join(args.image_path, image_name)
        logger.info("Reading image from: {}".format(image_path))

        image_np = cv2.imread(image_path)
        if image_np is None:
            logger.info("Failed to read image, skipping.")
            continue
        image_ori = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_ori)
        original_width, original_height = image_pil.size

        mask_json = get_mask_from_points(anno, image_np)

        logger.info("User question: {}".format(input_question))
        if options is not None:
            logger.info("Options: {}".format(options))

        resize_size = ast.literal_eval(args.resize_size)  # (w, h)
        logger.info("resize into :{}".format(resize_size))
        RESIZE_W, RESIZE_H = resize_size

        image_resized_rgb = cv2.resize(
            image_ori,
            (RESIZE_W, RESIZE_H),
            interpolation=cv2.INTER_AREA,
        )
        image_pil_resized = Image.fromarray(image_resized_rgb)

        sx = original_width / RESIZE_W
        sy = original_height / RESIZE_H

        # -------- Qwen iterative refinement (bbox + points) -------- #
        results = refiner.refine(
            image=image_pil_resized,
            question=input_question,
            question_type=question_type,
            options=options,
            initial_bbox=None,
        )

        if not results:
            bbox = None
            points = None
            final_response = ""
        else:
            final_step = results[-1]

            # bbox from resized coords -> original
            fx1_r, fy1_r, fx2_r, fy2_r = clamp_bbox_to_image(final_step.bbox, RESIZE_W, RESIZE_H)
            final_bbox_orig = [
                int(round(fx1_r * sx)),
                int(round(fy1_r * sy)),
                int(round(fx2_r * sx)),
                int(round(fy2_r * sy)),
            ]
            final_bbox_orig = clamp_bbox_to_image(final_bbox_orig, original_width, original_height)
            bbox = final_bbox_orig

            # points from resized coords -> original
            final_points_orig: List[List[int]] = []
            if final_step.points is not None:
                for px, py in final_step.points:
                    px_c = max(0.0, min(float(px), RESIZE_W - 1))
                    py_c = max(0.0, min(float(py), RESIZE_H - 1))
                    px_o = int(round(px_c * sx))
                    py_o = int(round(py_c * sy))
                    px_o = max(0, min(px_o, original_width - 1))
                    py_o = max(0, min(py_o, original_height - 1))
                    final_points_orig.append([px_o, py_o])

            # safety fallback if something went wrong
            if not final_points_orig:
                x1, y1, x2, y2 = bbox
                cx = int(round(0.5 * (x1 + x2)))
                cy = int(round(0.5 * (y1 + y2)))
                final_points_orig = [[cx, cy], [x1, y1]]

            points = final_points_orig
            final_response = (final_step.response or "").lower()

            logger.info(f"Refine steps: {[ (r.step, r.bbox, r.points, r.decision, r.response) for r in results ]}")
            logger.info(f"Final bbox (orig coords): {bbox}")
            logger.info(f"Final points (orig coords): {points}")
            logger.info(f"Final response: {final_response}")

        if bbox is not None:
            gt_bbox = get_bbox_from_mask(mask_json, original_width, original_height)

            if is_inside(gt_bbox, bbox):
                box_is_valid_num += 1
            all_num += 1

            # SAM2 segmentation using bbox + Qwen points
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                segmentation_model.set_image(image_ori)
                if points is not None:
                    point_coords = np.array(points, dtype=np.float32)  # shape (2, 2)
                    point_labels = np.ones(len(points), dtype=np.int32)
                else:
                    point_coords = None
                    point_labels = None

                masks, scores, _ = segmentation_model.predict(
                    box=bbox,
                    point_coords=point_coords,
                    point_labels=point_labels,
                )
                sorted_ind = np.argsort(scores)[::-1]
                masks = masks[sorted_ind]

            mask = masks[0].astype(bool)
            restored_mask = mask

            intersection, union, acc_iou = 0.0, 0.0, 0.0
            intersection_i, union_i, _ = intersectionAndUnionCPU(
                restored_mask, mask_json, 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0

            intersection_meter.append(intersection)
            union_meter.append(union)
            acc_iou_meter.append(acc_iou)

            seg_metric_dict = update_seg_metric(
                seg_metric_dict,
                area_percent,
                intersection,
                union,
                acc_iou,
            )

            if args.qa_stage != "no_qa":
                gt_answer_lower = gt_answer.lower()
                if question_type == "option":
                    logger.info("-option-")
                    logger.info('final_response: {}'.format(final_response))
                    logger.info('gt_answer: {}'.format(gt_answer_lower))
                    pattern = rf"(?:\b|[\(\[\{{'\" ]){gt_answer_lower}(?:\b|[\)\]\}}'\" ,.!?])"
                    if re.search(pattern, final_response):
                        option_correct += 1
                        qa_metric_dict = update_qa_metric_correct(qa_metric_dict, question_type, attribute)
                        logger.info("选择正确")
                    qa_metric_dict = update_qa_metric_num(qa_metric_dict, question_type, attribute)
                    option_num += 1
                elif question_type == "reasoning":
                    logger.info("-reasoning-")
                    logger.info('final_response: {}'.format(final_response))
                    logger.info('gt_answer: {}'.format(gt_answer_lower))
                    tokens = final_response.split()
                    for word in tokens:
                        similarity = difflib.SequenceMatcher(None, word, gt_answer_lower).ratio()
                        if similarity >= 0.8:
                            reasoning_correct += 1
                            qa_metric_dict = update_qa_metric_correct(qa_metric_dict, question_type, attribute)
                            logger.info("推理正确")
                            break
                    qa_metric_dict = update_qa_metric_num(qa_metric_dict, question_type, attribute)
                    reasoning_num += 1
            else:
                logger.info("No QA, No metric update!!")

            if args.save_results:
                draw_bbox(
                    image_ori, None, bbox, gt_bbox,
                    input_question, None, mask_json, restored_mask,
                    args.save_path, image_name, question_type,
                    hr_box=None, restored_box=None,
                )

        else:
            logger.info("No bbox from refine, create empty mask !!")
            restored_mask = np.zeros((mask_json.shape[0], mask_json.shape[1])).astype(bool)
            intersection, union, acc_iou = 0.0, 0.0, 0.0
            intersection_i, union_i, _ = intersectionAndUnionCPU(
                restored_mask, mask_json, 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0

            intersection_meter.append(intersection)
            union_meter.append(union)
            acc_iou_meter.append(acc_iou)

            seg_metric_dict = update_seg_metric(
                seg_metric_dict,
                area_percent,
                intersection,
                union,
                acc_iou,
            )
            all_num += 1

            if args.qa_stage != "no_qa":
                if question_type == "option":
                    option_num += 1
                    qa_metric_dict = update_qa_metric_num(qa_metric_dict, question_type, attribute)
                elif question_type == "reasoning":
                    reasoning_num += 1
                    qa_metric_dict = update_qa_metric_num(qa_metric_dict, question_type, attribute)

    iou_class = sum(intersection_meter) / (sum(union_meter) + 1e-10)
    ciou = iou_class[1]
    acc_iou_meter_sum = sum(acc_iou_meter)
    giou = acc_iou_meter_sum / len(acc_iou_meter)
    box_valid_num_acc = box_is_valid_num / (all_num + 1e-10)

    if option_num == 0:
        print("no options")
        option_acc = -1
    else:
        option_acc = option_correct / option_num

    if reasoning_num == 0:
        print("no reasoning_num")
        reasoning_acc = -1
    else:
        reasoning_acc = reasoning_correct / reasoning_num

    logger.info("intersection_meter.sum:{}".format(sum(intersection_meter)))
    logger.info("union_meter.sum:{}".format(sum(union_meter)))
    logger.info("acc_iou_meter.avg:{}".format(acc_iou_meter_sum / len(acc_iou_meter)))

    logger.info("giou: {}, ciou: {}".format(giou, iou_class))
    logger.info("giou: {:.4f}, ciou: {:.4f}".format(giou[1], ciou))
    logger.info(
        "box_valid_num_acc: {:.4f}, option_acc: {:.4f}, reasoning_acc: {:.4f}".format(
            box_valid_num_acc, option_acc, reasoning_acc
        )
    )

    qa_metric_dict, seg_metric_dict = final_metric(qa_metric_dict, seg_metric_dict, reasoning_num, option_num)

    if args.save_results:
        save_metric(qa_metric_dict, seg_metric_dict, args.save_path)
        result = {
            "giou": float(giou[1]),
            "ciou": float(ciou),
            "box_valid_num_acc": float(box_valid_num_acc),
            "option_acc": float(option_acc),
            "reasoning_acc": float(reasoning_acc),
        }
        with open(os.path.join(args.save_path, "results.json"), "w") as f:
            json.dump(result, f, indent=4)


if __name__ == "__main__":
    main()
