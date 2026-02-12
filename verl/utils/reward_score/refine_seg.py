import re
import json
import math
import difflib

# If you always resize to 1920x1080 for Qwen
IMAGE_W = 1920
IMAGE_H = 1080
IMAGE_AREA = IMAGE_W * IMAGE_H

# Same thresholds you used for segmentation buckets (for point reward)
XXS_TH = 0.017
S_TH = 0.055


# ======================= Low-level helpers ======================= #

def _extract_json(s: str):
    """
    Extract the first JSON object in a string and parse it.
    We assume one top-level JSON object (your 'json:' block).
    """
    try:
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        json_str = s[start:end + 1]
        return json.loads(json_str)
    except Exception:
        return None


def _bbox_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    inter_x1 = max(x1, x1g)
    inter_y1 = max(y1, y1g)
    inter_x2 = min(x2, x2g)
    inter_y2 = min(y2, y2g)

    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0

    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = _bbox_area(box1)
    area2 = _bbox_area(box2)
    union = area1 + area2 - inter
    return float(inter) / union if union > 0 else 0.0


def _bbox_l1(box1, box2):
    """
    Average L1 distance over the 4 coordinates.
    """
    return (
        abs(box1[0] - box2[0]) +
        abs(box1[1] - box2[1]) +
        abs(box1[2] - box2[2]) +
        abs(box1[3] - box2[3])
    ) / 4.0


def _size_bucket_from_area_ratio(area_ratio: float) -> str:
    """
    For point reward: derive bucket from area ratio if needed.
    """
    if area_ratio < XXS_TH:
        return "xxs"
    elif area_ratio > S_TH:
        return "s"
    else:
        return "xs"


# ======================= Format rewards ======================= #

def iterative_thinking_format_reward(predict_str: str) -> float:
    """
    Reward if the model uses the expected 'think:' + 'json:' pattern
    and actually has a JSON block we can parse.
    """
    s = predict_str.lower()
    has_think = "think:" in s
    has_json = "json:" in s
    has_parsable_json = _extract_json(predict_str) is not None
    return 1.0 if (has_think and has_json and has_parsable_json) else 0.0


def iterative_segmentation_format_reward(predict_str: str) -> float:
    """
    Reward if the JSON has the expected keys and shapes:
      - bbox_2d: length 4
      - points_2d: list of 2 [x,y] pairs
      - response: non-empty string
      - decision: "refine" or "stop"
      - reason: non-empty string
    """
    try:
        data = _extract_json(predict_str)
        if data is None:
            return 0.0

        required_keys = ["bbox_2d", "points_2d", "response", "decision", "reason"]
        for k in required_keys:
            if k not in data:
                return 0.0

        bbox = data["bbox_2d"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            return 0.0

        pts = data["points_2d"]
        if not isinstance(pts, list) or len(pts) != 2:
            return 0.0
        for p in pts:
            if not isinstance(p, list) or len(p) != 2:
                return 0.0

        resp = data["response"]
        if not isinstance(resp, str) or resp.strip() == "":
            return 0.0

        decision = str(data["decision"]).strip().lower()
        if decision not in ["refine", "stop"]:
            return 0.0

        reason = data["reason"]
        if not isinstance(reason, str) or reason.strip() == "":
            return 0.0

        return 1.0
    except Exception:
        return 0.0


# ======================= BBox rewards (level-dependent only) ======================= #

def bbox_iou_reward(
    predict_str: str,
    ground_truth: str,
    level: str = "medium",  # "coarse" | "medium" | "fine"
) -> float:
    """
    IoU reward with thresholds that depend ONLY on target level:

      coarse  -> IoU >= 0.30
      medium  -> IoU >= 0.50
      fine    -> IoU >= 0.70

    Reward is 1.0 if IoU >= threshold, else 0.0.
    No scaling based on GT / target box size.
    """
    try:
        pred = _extract_json(predict_str)
        gt = _extract_json(ground_truth)
        if pred is None or gt is None:
            return 0.0

        pred_box = pred.get("bbox_2d")
        target_box = gt.get("bbox_2d")
        if not (isinstance(pred_box, list) and len(pred_box) == 4):
            return 0.0
        if not (isinstance(target_box, list) and len(target_box) == 4):
            return 0.0

        thr_map = {
            "coarse": 0.40,
            "medium": 0.60,
            "fine":   0.70,
        }
        thr = thr_map.get(level, 0.50)

        iou = _bbox_iou(pred_box, target_box)
        return 1.0 if iou >= thr else 0.0
    except Exception:
        return 0.0


def bbox_l1_reward(
    predict_str: str,
    ground_truth: str,
    level: str = "medium",
) -> float:
    """
    L1 distance reward with FIXED thresholds per level (no GT-size scaling):

      coarse -> average L1 <= 30 px
      medium -> average L1 <= 15 px
      fine   -> average L1 <= 8 px

    Reward is 1.0 if within threshold, else 0.0.
    """
    try:
        pred = _extract_json(predict_str)
        gt = _extract_json(ground_truth)
        if pred is None or gt is None:
            return 0.0

        pred_box = pred.get("bbox_2d")
        target_box = gt.get("bbox_2d")
        if not (isinstance(pred_box, list) and len(pred_box) == 4):
            return 0.0
        if not (isinstance(target_box, list) and len(target_box) == 4):
            return 0.0

        thr_map = {
            "coarse": 30.0,
            "medium": 15.0,
            "fine":   8.0,
        }
        thr = thr_map.get(level, 15.0)

        d = _bbox_l1(pred_box, target_box)
        return 1.0 if d <= thr else 0.0
    except Exception:
        return 0.0


# ======================= Point reward (unchanged, size-aware) ======================= #

def point_weighted_reward(
    predict_str: str,
    ground_truth: str,
    size_bucket: str = None,  # "xxs" | "xs" | "s"
    dist_threshold: float = 100.0,
) -> float:
    """
    Point reward, weighted by how small the target bbox / object is.

    - Base reward: 1 if BOTH predicted points are inside predicted bbox AND
      the best pairing distance to GT points < dist_threshold.
    - Weight (by size_bucket):
        xxs -> 1.5
        xs  -> 1.0
        s   -> 0.7
    If size_bucket is None, we derive it from the target box area ratio.
    """
    try:
        pred = _extract_json(predict_str)
        gt = _extract_json(ground_truth)
        if pred is None or gt is None:
            return 0.0

        pred_box = pred.get("bbox_2d")
        pred_pts = pred.get("points_2d")
        target_box = gt.get("bbox_2d")
        gt_pts = gt.get("points_2d")

        if not (isinstance(pred_box, list) and len(pred_box) == 4):
            return 0.0
        if not (isinstance(pred_pts, list) and len(pred_pts) == 2):
            return 0.0
        if not (isinstance(target_box, list) and len(target_box) == 4):
            return 0.0
        if not (isinstance(gt_pts, list) and len(gt_pts) == 2):
            return 0.0

        pred_box = [float(v) for v in pred_box]
        target_box = [float(v) for v in target_box]
        pred_pts = [[float(p[0]), float(p[1])] for p in pred_pts]
        gt_pts = [[float(p[0]), float(p[1])] for p in gt_pts]

        def points_in_box(pt, box):
            return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]

        if not (points_in_box(pred_pts[0], pred_box) and points_in_box(pred_pts[1], pred_box)):
            return 0.0

        def pair_distance(p1, p2):
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        d1 = pair_distance(pred_pts[0], gt_pts[0]) + pair_distance(pred_pts[1], gt_pts[1])
        d2 = pair_distance(pred_pts[0], gt_pts[1]) + pair_distance(pred_pts[1], gt_pts[0])
        best = min(d1, d2) / 2.0

        if best >= dist_threshold:
            return 0.0

        base_reward = 1.0

        # use provided size_bucket if available, otherwise derive from target box area
        if size_bucket is None:
            area = _bbox_area(target_box)
            area_ratio = area / (IMAGE_AREA + 1e-9)
            size_bucket = _size_bucket_from_area_ratio(area_ratio)

        bucket_mult_map = {
            "xxs": 1.5,
            "xs": 1.0,
            "s": 0.7,
        }
        weight = bucket_mult_map.get(size_bucket, 1.0)

        return base_reward * weight

    except Exception:
        return 0.0


# ======================= Improvement reward ======================= #

def improvement_reward(
    predict_str: str,
    ground_truth: str,
    input_bbox,
    margin: float = 0.01,
) -> float:
    """
    Shaping reward based on improvement over the input box.

    Compare IoU(input_bbox, target_bbox) vs IoU(pred_bbox, target_bbox):

    - If IoU_pred > IoU_input + margin:  positive reward ~= (IoU_pred - IoU_input)
    - If IoU_pred < IoU_input - margin:  negative reward ~= (IoU_pred - IoU_input)
    - If difference is small (within margin): 0
    """
    try:
        if input_bbox is None:
            return 0.0
        if not isinstance(input_bbox, (list, tuple)) or len(input_bbox) != 4:
            return 0.0

        pred = _extract_json(predict_str)
        gt = _extract_json(ground_truth)
        if pred is None or gt is None:
            return 0.0

        target_box = gt.get("bbox_2d")
        pred_box = pred.get("bbox_2d")
        if not (isinstance(target_box, list) and len(target_box) == 4):
            return 0.0
        if not (isinstance(pred_box, list) and len(pred_box) == 4):
            return 0.0

        input_box = [float(v) for v in input_bbox]
        target_box = [float(v) for v in target_box]
        pred_box = [float(v) for v in pred_box]

        iou_input = _bbox_iou(input_box, target_box)
        iou_pred = _bbox_iou(pred_box, target_box)
        delta = iou_pred - iou_input

        if abs(delta) <= margin:
            return 0.0

        delta = max(-1.0, min(1.0, delta))
        return delta
    except Exception:
        return 0.0


# ======================= Text reward ======================= #

def text_reward(predict_str: str, ground_truth: str) -> float:
    """
    Compare predicted 'response' vs GT 'response' in the JSON.
    Same logic as your previous code but reading from JSON.
    """
    try:
        pred = _extract_json(predict_str)
        gt = _extract_json(ground_truth)
        if pred is None or gt is None:
            return 0.0

        output_text = str(pred.get("response", "")).strip()
        gt_response = str(gt.get("response", "")).strip().lower()
        if not gt_response:
            return 0.0

        # Referring: "The object is found."
        if gt_response == "the object is found.":
            referring_keywords = ["is found", "is detected"]
            return 1.0 if any(k in output_text.lower() for k in referring_keywords) else 0.0

        # Multiple-choice: "A" / "B" / "C" / "D"
        if gt_response in ["a", "b", "c", "d", "A", "B", "C", "D"]:
            output = output_text.lower()
            option = gt_response.lower()
            pattern = rf"(?:\b|[\(\[\{{'\" ]){option}(?:\b|[\)\]\}}'\" ,.!?])"
            return 1.0 if re.search(pattern, output) else 0.0

        # Open-ended: fuzzy string match
        gt_str = gt_response.lower()
        output_words = output_text.split()
        if len(output_words) > 3:
            return 0.0
        for word in output_words:
            similarity = difflib.SequenceMatcher(None, word.lower(), gt_str).ratio()
            if similarity >= 0.8:
                return 1.0
        return 0.0

    except Exception:
        return 0.0


# ======================= Decision reward ======================= #

def decision_reward(predict_str: str, ground_truth: str) -> float:
    """
    Reward correct 'decision'.

    - If pred == GT: +1.0
    - If mismatch:   0.0
    """
    try:
        pred = _extract_json(predict_str)
        gt = _extract_json(ground_truth)
        if pred is None or gt is None:
            return 0.0

        pred_dec = str(pred.get("decision", "")).strip().lower()
        gt_dec = str(gt.get("decision", "")).strip().lower()

        if gt_dec not in ["refine", "stop"] or pred_dec not in ["refine", "stop"]:
            return 0.0

        return 1.0 if pred_dec == gt_dec else 0.0

    except Exception:
        return 0.0


# ======================= Top-level combined reward ======================= #

def seg_iterative_compute_score(
    predict_str: str,
    ground_truth: str,
    input_bbox=None,
    output_level: str = "medium",   # rec["output_level"]
    size_bucket: str = None         # rec["size_bucket"] (for points)
) -> float:
    """
    Main reward function to plug into VERL.

    predict_str: full model completion (think + json)
    ground_truth: GT answer JSON string from your dataset (rec["answer"]).
    input_bbox: the input box for this step (rec["input_bbox"]),
                same coords as bbox_2d, or None for initial.
    output_level: "coarse" | "medium" | "fine" (rec["output_level"]).
    size_bucket: "xxs" | "xs" | "s" (rec["size_bucket"]), used only for point weighting.
    """
    print("------------------------------- predict_str -------------------------------")
    print(predict_str)

    thinking_format = iterative_thinking_format_reward(predict_str)
    seg_format = iterative_segmentation_format_reward(predict_str)
    iou = bbox_iou_reward(predict_str, ground_truth, level=output_level)
    box_l1 = bbox_l1_reward(predict_str, ground_truth, level=output_level)
    points = point_weighted_reward(predict_str, ground_truth, size_bucket=size_bucket)
    text = text_reward(predict_str, ground_truth)
    decision = decision_reward(predict_str, ground_truth)
    improve = improvement_reward(predict_str, ground_truth, input_bbox)

    reward = (
        thinking_format
        + seg_format
        + iou
        + box_l1
        + points
        + text
        + decision
        + improve
    )

    print(
        "------------------------------- reward breakdown -------------------------------"
    )
    print(
        f"thinking_format: {thinking_format}, "
        f"segmentation_format: {seg_format}, "
        f"iou (level={output_level}): {iou}, "
        f"box_l1 (level={output_level}): {box_l1}, "
        f"points (bucket={size_bucket}): {points}, "
        f"text: {text}, "
        f"decision: {decision}, "
        f"improvement: {improve}, "
        f"total: {reward}"
    )

    return reward
