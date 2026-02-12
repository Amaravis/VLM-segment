import math
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset, load_from_disk
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from verl.utils.model_utils import compute_position_id_with_mask

import verl.utils.torch_functional as verl_F
from verl.models.transformers.qwen2_5_vl import get_rope_index


# ============================== Collate ============================== #


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Standard collate: stack tensor fields, keep lists for non-tensor fields.
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        # Do NOT stack multi_modal_inputs (dict) or multi_modal_data (dicts/PIL)
        if key not in ["multi_modal_inputs", "multi_modal_data"]:
            tensors[key] = torch.stack(value, dim=0)

    # multi_modal_inputs stays as a list[dict] in non_tensors
    if "multi_modal_inputs" in tensors:
        non_tensors["multi_modal_inputs"] = tensors.pop("multi_modal_inputs")

    return {**tensors, **non_tensors}


# ============================== Image pre-process ============================== #

def process_image(image: ImageObject, max_pixels: Optional[int], min_pixels: Optional[int]) -> ImageObject:
    """
    Resize image if needed to keep pixels within [min_pixels, max_pixels].
    For your setup, you can set both to 1920*1080 so this is effectively a no-op.
    """
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width = int(image.width * resize_factor)
        height = int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width = int(image.width * resize_factor)
        height = int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


# ============================== Prompt helpers ============================== #

def _response_hint(question_type: str, options: Optional[List[str]]) -> str:
    """
    Small hint to clarify what the 'response' field should be.
    """
    if question_type == "referring":
        return (
            'For referring questions, set "response" to '
            '"The object is found." or "The object is not found." as appropriate.'
        )
    elif question_type == "option":
        return (
            'For option questions, the "response" field must contain the correct '
            'option (e.g. "A").'
        )
    else:
        return (
            'For reasoning questions, "response" should be a short phrase that '
            'directly answers the question (no more than a few words).'
        )


# --------- INITIAL PROMPT (WITH EXAMPLE) --------- #

def build_initial_prompt_text(
    question: str,
    question_type: str,
    width: int,
    height: int,
    options: Optional[List[str]] = None,
) -> str:
    """
    Text-only initial prompt (used with <image> placeholder).
    This version INCLUDES a format example.
    """
    response_hint = _response_hint(question_type, options)

    options_block = ""
    if question_type == "option" and options is not None:
        options_block = f"\nOptions: {options}\n"

    if question_type == "reasoning":
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
    else:  # referring
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
1. Predict a coarse bounding box in this image coordinate system that covers the object/region
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
- JSON must use double quotes for all keys and string values.
- Do NOT output any text after the JSON object.
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
    return QUESTION_TEMPLATE


# --------- INITIAL PROMPT (NO EXAMPLE) --------- #

def build_initial_prompt_text_no_example(
    question: str,
    question_type: str,
    width: int,
    height: int,
    options: Optional[List[str]] = None,
) -> str:
    """
    Same initial prompt, but WITHOUT the long example block.
    Lighter on tokens.
    """
    response_hint = _response_hint(question_type, options)

    options_block = ""
    if question_type == "option" and options is not None:
        options_block = f"\nOptions: {options}\n"

    QUESTION_TEMPLATE = f"""
You are localizing the region in the image that is most relevant to the question.

  Question: "{question}"
  Image resolution: width={width}, height={height} (pixels).
{options_block}Task:
1. Predict a coarse bounding box in this image coordinate system that covers the object/region
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

The JSON schema (FORMAT ONLY):

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
- JSON must use double quotes for all keys and string values.
- Do NOT output any text after the JSON object.
- {response_hint}
"""
    return QUESTION_TEMPLATE


# --------- REFINE PROMPT (WITH EXAMPLE) --------- #

def build_refine_prompt_text(
    question: str,
    question_type: str,
    width: int,
    height: int,
    current_bbox: List[float],
    step: int,
    max_steps: int,
    options: Optional[List[str]] = None,
) -> str:
    """
    Text-only refine prompt (used with <image> placeholder).
    This version INCLUDES a format/example block.
    """
    x1, y1, x2, y2 = current_bbox
    bbox_token = f"<box>({int(x1)},{int(y1)}),({int(x2)},{int(y2)})</box>"
    response_hint = _response_hint(question_type, options)

    options_block = ""
    if question_type == "option" and options is not None:
        options_block = f"\nOptions: {options}\n"

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
2. If refining, slightly shift and/or shrink the bbox so it covers the object
   with a small amount of background:
   - Prefer shrinking over expanding.
   - Never cut off any visible part of the object.
   - The refined box should have some overlap with the current box (no random jumps).
3. Also answer the question.
4. Choose TWO points inside the refined object/region that are helpful for
   segmentation (for example, one near the center and one on another part of the
   object).

Stopping rule (conceptual):
- If the box already covers the object with about 10–20% extra background,
  and further shrinking would likely cut off the object or change area only slightly,
  set "decision": "stop" and keep the bbox nearly unchanged.

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
- "decision" MUST be exactly "refine" or "stop" (lowercase).
- JSON must use double quotes for all keys and string values.
- Do NOT output any text after the JSON object.
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
    return QUESTION_TEMPLATE


# --------- REFINE PROMPT (NO EXAMPLE) --------- #

def build_refine_prompt_text_no_example(
    question: str,
    question_type: str,
    width: int,
    height: int,
    current_bbox: List[float],
    step: int,
    max_steps: int,
    options: Optional[List[str]] = None,
) -> str:
    """
    Refine prompt WITHOUT the long example block.
    """
    x1, y1, x2, y2 = current_bbox
    bbox_token = f"<box>({int(x1)},{int(y1)}),({int(x2)},{int(y2)})</box>"
    response_hint = _response_hint(question_type, options)

    options_block = ""
    if question_type == "option" and options is not None:
        options_block = f"\nOptions: {options}\n"

    QUESTION_TEMPLATE = f"""
You are REFINING a bounding box for the object/region relevant to this question.

  Question: "{question}"
  Image resolution: width={width}, height={height} (pixels).
  Current bbox_2d: {bbox_token}
{options_block}You always see the full image, so you must reason using GLOBAL context,
but treat the current bbox as your current best guess region.

Goals at refinement step {step}/{max_steps}:
1. Decide whether to keep refining or stop.
2. If refining, slightly shift and/or shrink the bbox so it covers the object
   with a small amount of background:
   - Prefer shrinking over expanding.
   - Never cut off any visible part of the object.
   - The refined box should have some overlap with the current box (no random jumps).
3. Also answer the question.
4. Choose TWO points inside the refined object/region that are helpful for
   segmentation (for example, one near the center and one on another part of the
   object).

Stopping rule (conceptual):
- If the box already covers the object with about 10–20% extra background,
  and further shrinking would likely cut off the object or change area only slightly,
  set "decision": "stop" and keep the bbox nearly unchanged.

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

The JSON schema (FORMAT ONLY):

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
- "decision" MUST be exactly "refine" or "stop" (lowercase).
- JSON must use double quotes for all keys and string values.
- Do NOT output any text after the JSON object.
- {response_hint}
"""
    return QUESTION_TEMPLATE


# ============================== RLHFDataset ============================== #

class RLHFDataset(Dataset):
    """
    RLHF / GRPO dataset for Finers-style iterative grounding.

    Key differences from your original version:

    - DOES NOT expand <image> into vision_start / image_pad / vision_end tokens.
    - Uses processor(...) to build model inputs for training.
    - Provides:
        * multi_modal_data   -> used by vLLM rollout
        * multi_modal_inputs -> used by FSDP actor/critic forward
    """

    def __init__(
        self,
        unified_qa_model: bool,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: ProcessorMixin,
        prompt_key: str = "problem",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        system_prompt: Optional[str] = None,
        max_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
        load_from_disk_flag: bool = False,
        use_prompt_examples: bool = False,
    ):
        self.unified_qa_model = unified_qa_model  # kept for config compatibility
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.system_prompt = system_prompt or (
            "You are a vision-language assistant that localizes objects, refines "
            "bounding boxes, chooses segmentation points, and answers questions. "
            "You must strictly follow the requested output format: think: then json:."
        )
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.use_prompt_examples = use_prompt_examples

        if load_from_disk_flag:
            self.dataset = load_from_disk(data_path)["train"]
        else:
            self.dataset = load_dataset(data_path)["train"]

    def __len__(self):
        return len(self.dataset)

    def _build_prompt_body(
        self,
        row: Dict[str, Any],
        image_width: int,
        image_height: int,
    ) -> str:
        question = row["problem"]
        question_type = row["type"]
        options = row.get("options", None)
        stage = row.get("stage", "initial")
        input_level = row.get("input_level", None)

        if stage == "initial":
            if self.use_prompt_examples:
                return build_initial_prompt_text(
                    question=question,
                    question_type=question_type,
                    width=image_width,
                    height=image_height,
                    options=options,
                )
            else:
                return build_initial_prompt_text_no_example(
                    question=question,
                    question_type=question_type,
                    width=image_width,
                    height=image_height,
                    options=options,
                )

        # refine stage
        if row.get("input_bbox") is None:
            current_bbox = row.get("gt_bbox_resized")
        else:
            current_bbox = row.get("input_bbox")

        if input_level == "coarse":
            step = 1
        elif input_level == "medium":
            step = 2
        else:
            step = 1
        max_steps = 3

        if self.use_prompt_examples:
            return build_refine_prompt_text(
                question=question,
                question_type=question_type,
                width=image_width,
                height=image_height,
                current_bbox=current_bbox,
                step=step,
                max_steps=max_steps,
                options=options,
            )
        else:
            return build_refine_prompt_text_no_example(
                question=question,
                question_type=question_type,
                width=image_width,
                height=image_height,
                current_bbox=current_bbox,
                step=step,
                max_steps=max_steps,
                options=options,
            )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.dataset[index]

        img_id = row["id"]
        question = row["problem"]
        question_type = row["type"]
        options = row.get("options", None)

        stage = row.get("stage", "initial")
        input_level = row.get("input_level", None)
        output_level = row.get("output_level", None)
        size_bucket = row.get("size_bucket", None)

        gt_bbox_resized = row.get("gt_bbox_resized", None)
        input_bbox = row.get("input_bbox", None)
        target_bbox = row.get("target_bbox", None)
        target_points = row.get("target_points", None)
        gt_answer_text = row.get("gt_answer_text", None)
        answer_json = row.get("answer", None)

        # ---- Image ----
        image = row["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = process_image(image, self.max_pixels, self.min_pixels)
        width, height = image.size

        # ---- Prompt text (keep <image> literally) ----
        prompt_body = self._build_prompt_body(
            row,
            image_width=width,
            image_height=height,
        )
        user_content = [
            {"type": "image"},                          # 1) image block
            {"type": "text", "text": prompt_body},      # 2) text block
        ]

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": self.system_prompt},
                ],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        # Text with a literal "<image>" placeholder
        prompt_str: str = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # This is the canonical raw prompt text we’ll log / reconstruct from
        raw_prompt = prompt_str

        # ---- multi_modal_data for vLLM ----
        # vLLM expects "image" key and a list of PIL images / arrays
        #multi_modal_data = {"image": [image]}

        # ---- HF processor for training (multi_modal_inputs + text ids) ----
        # Call processor on *batched* text + images
        model_inputs = self.processor(
            text=[raw_prompt],
            images=[image],
            return_tensors="pt",
        )
        # model_inputs:
        #   - input_ids:      (1, L_raw)
        #   - attention_mask: (1, L_raw)
        #   - pixel_values, image_grid_thw, etc. depending on processor

        # --- Pad / truncate to max_prompt_length with your new helper ---
        input_ids = model_inputs["input_ids"]          # (1, L_raw)
        attention_mask = model_inputs["attention_mask"]  # (1, L_raw)

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # Squeeze batch dimension back to 1D tensors for this single sample
        input_ids = input_ids[0]            # (max_prompt_length,)
        attention_mask = attention_mask[0]  # (max_prompt_length,)

        # ---- Position ids (Qwen2.5-VL M-RoPE) ----
        image_grid_thw = model_inputs.get("image_grid_thw", None)
        if image_grid_thw is not None:
            # model_inputs["image_grid_thw"]: (1, num_images, 3) -> squeeze batch dim
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )
        else:
            # Text-only fallback (shouldn’t usually happen for Qwen2.5-VL)
            position_ids = torch.clip(
                attention_mask.cumsum(dim=0) - 1,
                min=0,
                max=None,
            )

        # ---- multi_modal_inputs for FSDP ----
        # Everything except input_ids / attention_mask becomes multi_modal_inputs.
        multi_modal_inputs: Dict[str, Any] = {}
        for k, v in model_inputs.items():
            if k in ["input_ids", "attention_mask"]:
                continue
            if isinstance(v, torch.Tensor) and v.size(0) == 1:
                multi_modal_inputs[k] = v[0]   # drop batch dim
            else:
                multi_modal_inputs[k] = v

        # ---- Build row_dict ----
        row_dict: Dict[str, Any] = {}

        # tensors
        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids

        # raw prompt ids are used by VERL for logging / text reconstruction
        raw_prompt_ids = self.tokenizer.encode(
            raw_prompt,
            add_special_tokens=False,
        )

        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}."
                )
            
        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # multimodal hooks
        #row_dict["multi_modal_data"] = multi_modal_data       # for vLLM
        row_dict["multi_modal_inputs"] = multi_modal_inputs   # for FSDP

        # keep original PIL for debugging / visualizations if you want
        row_dict["images"] = [image]

        # ---- Non-tensor metadata for reward / debugging ----
        row_dict["id"] = [img_id]
        row_dict["problem"] = [question]
        row_dict["question_type"] = [question_type]
        row_dict["options_list"] = [options]

        row_dict["stage"] = [stage]
        row_dict["input_level"] = [input_level]
        row_dict["output_level"] = [output_level]
        row_dict["size_bucket"] = [size_bucket]

        row_dict["gt_bbox_resized"] = [gt_bbox_resized]
        row_dict["input_bbox"] = [input_bbox]
        row_dict["target_bbox"] = [target_bbox]
        row_dict["target_points"] = [target_points]

        row_dict["gt_answer_text"] = [gt_answer_text]
        row_dict["answer"] = [answer_json]

        # For compatibility with older seg reward code
        row_dict["solution"] = [answer_json]

        return row_dict

"""
class RLHFDataset(Dataset):
 
    def __init__(
        self,
        unified_qa_model: bool,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "problem",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        system_prompt: Optional[str] = None,
        max_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
        load_from_disk_flag: bool = False,
        use_prompt_examples: bool = False,  # <--- NEW FLAG
    ):
        self.unified_qa_model = unified_qa_model  # kept for config compatibility
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.system_prompt = system_prompt or (
            "You are a vision-language assistant that localizes objects, refines "
            "bounding boxes, chooses segmentation points, and answers questions. "
            "You must strictly follow the requested output format: think: then json:."
        )
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.use_prompt_examples = use_prompt_examples

        if load_from_disk_flag:
            self.dataset = load_from_disk(data_path)["train"]
        else:
            # If you saved the RL dataset as a Parquet folder, you can also use:
            # self.dataset = load_dataset("parquet", data_files={"train": f"{data_path}/*.parquet"})["train"]
            self.dataset = load_dataset(data_path)["train"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]

        img_id = row["id"]
        question = row["problem"]
        question_type = row["type"]           # "referring" | "reasoning" | "option"
        options = row.get("options", None)

        stage = row.get("stage", "initial")   # "initial" | "refine"
        input_level = row.get("input_level", None)
        output_level = row.get("output_level", None)
        size_bucket = row.get("size_bucket", None)

        gt_bbox_resized = row.get("gt_bbox_resized", None)
        input_bbox = row.get("input_bbox", None)
        target_bbox = row.get("target_bbox", None)
        target_points = row.get("target_points", None)
        gt_answer_text = row.get("gt_answer_text", None)
        answer_json = row.get("answer", None)  # GT JSON for this step

        image = row["image"]
        if not isinstance(image, Image.Image):
            # HF Image feature or other format
            image = Image.fromarray(image)
        image = process_image(image, self.max_pixels, self.min_pixels)
        width, height = image.size

        # ---------- Build prompt text ---------- #
        if stage == "initial":
            if self.use_prompt_examples:
                prompt_body = build_initial_prompt_text(
                    question=question,
                    question_type=question_type,
                    width=width,
                    height=height,
                    options=options,
                )
            else:
                prompt_body = build_initial_prompt_text_no_example(
                    question=question,
                    question_type=question_type,
                    width=width,
                    height=height,
                    options=options,
                )
        else:
            if input_bbox is None:
                current_bbox = gt_bbox_resized
            else:
                current_bbox = input_bbox

            if input_level == "coarse":
                step = 1
            elif input_level == "medium":
                step = 2
            else:
                step = 1
            max_steps = 3

            if self.use_prompt_examples:
                prompt_body = build_refine_prompt_text(
                    question=question,
                    question_type=question_type,
                    width=width,
                    height=height,
                    current_bbox=current_bbox,
                    step=step,
                    max_steps=max_steps,
                    options=options,
                )
            else:
                prompt_body = build_refine_prompt_text_no_example(
                    question=question,
                    question_type=question_type,
                    width=width,
                    height=height,
                    current_bbox=current_bbox,
                    step=step,
                    max_steps=max_steps,
                    options=options,
                )

        user_content = "<image>" + prompt_body

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        row_dict: Dict[str, Any] = {}

        # ---------- Vision tokens + image processing ---------- #
        row_dict["images"] = [image]

        raw_prompt = prompt.replace(
            "<image>", "<|vision_start|><|image_pad|><|vision_end|>"
        )
        image_inputs = self.processor.image_processor(
            row_dict["images"],
            return_tensors="pt",
        )
        image_grid_thw = image_inputs["image_grid_thw"]
        row_dict.update(image_inputs)

        if image_grid_thw is not None:
            merge_length = self.processor.image_processor.merge_size ** 2
            index = 0
            while "<image>" in prompt:
                prompt = prompt.replace(
                    "<image>",
                    "<|vision_start|>"
                    + "<|placeholder|>" * int(image_grid_thw[index].prod() // merge_length)
                    + "<|vision_end|>",
                    1,
                )
                index += 1

            prompt = prompt.replace("<|placeholder|>", self.processor.image_token)

        # ---------- Tokenize full prompt ---------- #
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # ---------- Position ids ---------- #
        if "pixel_values" in row_dict:
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )
        else:
            position_ids = torch.clip(
                attention_mask.cumsum(dim=0) - 1,
                min=0,
                max=None,
            )

        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(
            raw_prompt,
            add_special_tokens=False,
        )

        # ---------- Non-tensor metadata for reward ---------- #
        row_dict["id"] = [img_id]
        row_dict["problem"] = [question]
        row_dict["question_type"] = [question_type]
        row_dict["options_list"] = [options]

        row_dict["stage"] = [stage]
        row_dict["input_level"] = [input_level]
        row_dict["output_level"] = [output_level]
        row_dict["size_bucket"] = [size_bucket]

        row_dict["gt_bbox_resized"] = [gt_bbox_resized]
        row_dict["input_bbox"] = [input_bbox]
        row_dict["target_bbox"] = [target_bbox]
        row_dict["target_points"] = [target_points]

        row_dict["gt_answer_text"] = [gt_answer_text]
        row_dict["answer"] = [answer_json]

        # For compatibility with older seg reward code
        row_dict["solution"] = [answer_json]

        return row_dict

"""