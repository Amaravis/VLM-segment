import re
import json
import math
import difflib

# If you always resize to 1920x1080 for Qwen
IMAGE_W = 1920
IMAGE_H = 1080
IMAGE_AREA = IMAGE_W * IMAGE_H

# Same thresholds you used for segmentation buckets (for point reward)
XXS_TH = 0.00017
S_TH = 0.00055


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


def _center_xy(box):
    x1, y1, x2, y2 = box
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def _center_l1(box1, box2):
    """
    L1 distance between box centers in pixels.
    """
    c1x, c1y = _center_xy(box1)
    c2x, c2y = _center_xy(box2)
    return abs(c1x - c2x) + abs(c1y - c2y)


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


def _bbox_intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _asym_delta(
    delta: float,
    margin: float,
    pos_scale: float = 1.0,
    neg_scale: float = 0.3,
    neg_clip: float = -0.2,
) -> float:
    """
    Asymmetric shaping:
      - ignore tiny deltas within margin
      - reward positive deltas more than you punish negative deltas
      - cap negative penalty with neg_clip (e.g. -0.2)
    """
    if abs(delta) <= margin:
        return 0.0
    if delta > 0:
        return float(pos_scale * delta)
    # negative: softer and clipped
    return float(max(neg_clip, neg_scale * delta))


# ======================= Format rewards ======================= #

def iterative_thinking_format_reward(predict_str: str) -> float:
    s = predict_str.lower()
    has_think = "think:" in s
    has_json = "json:" in s
    has_parsable_json = _extract_json(predict_str) is not None
    return 1.0 if (has_think and has_json and has_parsable_json) else 0.0


def iterative_segmentation_format_reward(predict_str: str) -> float:
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


# ======================= BBox rewards ======================= #

def bbox_coverage_reward(
    predict_str: str,
    ground_truth: str,
) -> float:
    """
    coverage = area(pred ∩ target) / area(target)
    - 0 if no overlap
    - 1 if pred fully covers target
    """
    try:
        pred = _extract_json(predict_str)
        gt = _extract_json(ground_truth)
        if pred is None or gt is None:
            return 0.0

        pred_box = pred.get("bbox_2d")
        tgt_box = gt.get("bbox_2d")
        if not (isinstance(pred_box, list) and len(pred_box) == 4):
            return 0.0
        if not (isinstance(tgt_box, list) and len(tgt_box) == 4):
            return 0.0

        pred_box = [float(v) for v in pred_box]
        tgt_box  = [float(v) for v in tgt_box]

        inter = _bbox_intersection_area(pred_box, tgt_box)
        tgt_area = _bbox_area(tgt_box)
        if tgt_area <= 0.0:
            return 0.0

        cov = inter / (tgt_area + 1e-9)
        return float(max(0.0, min(1.0, cov)))
    except Exception:
        return 0.0


def bbox_iou_reward(
    predict_str: str,
    ground_truth: str,
    level: str = "medium",  # "coarse" | "medium" | "fine"
) -> float:
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

        thr_map = {"coarse": 0.40, "medium": 0.60, "fine": 0.70}
        thr = thr_map.get(level, 0.50)

        iou = _bbox_iou(pred_box, target_box)
        if iou <= 0.0:
            return 0.0
        return 1.0 if iou >= thr else float(iou / thr)
    except Exception:
        return 0.0


def bbox_l1_reward(
    predict_str: str,
    ground_truth: str,
    level: str = "medium",  # "coarse" | "medium" | "fine"
) -> float:
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

        thr_map = {"coarse": 30.0, "medium": 15.0, "fine": 8.0}
        thr = thr_map.get(level, 15.0)

        d = _bbox_l1(pred_box, target_box)
        if d <= 0.0:
            return 1.0
        return 0.0 if d >= thr else max(0.0, 1.0 - float(d / thr))
    except Exception:
        return 0.0


# ======================= Point reward (unchanged, size-aware) ======================= #



def _adaptive_point_dist_threshold_px(target_box, size_bucket: str) -> float:
    """
    Returns a distance threshold (in pixels) for point matching that scales with
    the target bbox size and depends on the size_bucket.

    Uses a fraction of bbox diagonal, with bucket-specific clamps.
    """
    x1, y1, x2, y2 = [float(v) for v in target_box]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    diag = math.hypot(w, h)  # sqrt(w^2 + h^2)

    # Fractions of diagonal:
    # - XXS: more forgiving fraction (diag is tiny anyway)
    # - XS : medium
    # - S  : stricter fraction, but absolute sizes are bigger
    frac_map = {
        "xxs": 1.8,
        "xs":  1.2,
        "s":   0.8,
    }

    # Absolute clamps (px) so threshold isn't too small/large
    min_map = {
        "xxs": 12.0,
        "xs":  18.0,
        "s":   30.0,
    }
    max_map = {
        "xxs": 55.0,
        "xs":  90.0,
        "s":   140.0,
    }

    frac = frac_map.get(size_bucket, 1.0)
    thr = diag * frac
    thr = max(min_map.get(size_bucket, 15.0), min(max_map.get(size_bucket, 120.0), thr))
    return float(thr)

def point_weighted_reward(
    predict_str: str,
    ground_truth: str,
    size_bucket: str = None,         # "xxs" | "xs" | "s"
    dist_threshold: float = None,    # None => auto per bucket + bbox size
) -> float:
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

        # Keep your original constraint: both points must lie inside pred bbox
        if not (points_in_box(pred_pts[0], pred_box) and points_in_box(pred_pts[1], pred_box)):
            return 0.0

        def pair_distance(p1, p2):
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        d1 = pair_distance(pred_pts[0], gt_pts[0]) + pair_distance(pred_pts[1], gt_pts[1])
        d2 = pair_distance(pred_pts[0], gt_pts[1]) + pair_distance(pred_pts[1], gt_pts[0])
        best = min(d1, d2) / 2.0

        # derive bucket if needed
        if size_bucket is None:
            area = _bbox_area(target_box)
            area_ratio = area / (IMAGE_AREA + 1e-9)
            size_bucket = _size_bucket_from_area_ratio(area_ratio)

        # bucket-aware adaptive threshold if not explicitly provided
        if dist_threshold is None:
            dist_threshold = _adaptive_point_dist_threshold_px(target_box, size_bucket)

        if best >= float(dist_threshold):
            return 0.0

        bucket_mult_map = {"xxs": 1.5, "xs": 1.0, "s": 0.7}
        return 1.0 * bucket_mult_map.get(size_bucket, 1.0)

    except Exception:
        return 0.0


# ======================= Improvement reward (IoU + Center) ======================= #

def improvement_reward(
    predict_str: str,
    ground_truth: str,
    input_bbox,
    level: str = "medium",
    margin_iou: float = 0.01,
) -> float:
    """
    Encourage improvements MORE, discourage declines LESS.

    Uses two terms:
      - IoU improvement wrt target
      - Center-L1 improvement wrt target

    Both are normalized and passed through asym shaping.
    Center term normalization is done by TARGET bbox size (w+h) with a floor.
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

        tgt_box = gt.get("bbox_2d")
        pred_box = pred.get("bbox_2d")
        if not (isinstance(tgt_box, list) and len(tgt_box) == 4):
            return 0.0
        if not (isinstance(pred_box, list) and len(pred_box) == 4):
            return 0.0

        input_box = [float(v) for v in input_bbox]
        tgt_box   = [float(v) for v in tgt_box]
        pred_box  = [float(v) for v in pred_box]

        # --- IoU improvement (already normalized to [-1,1] and small) ---
        iou_in   = _bbox_iou(input_box, tgt_box)
        iou_pred = _bbox_iou(pred_box,  tgt_box)
        delta_iou = iou_pred - iou_in

        # IoU margins per level (optional)
        margin_iou_map = {"coarse": 0.01, "medium": 0.01, "fine": 0.005}
        margin_iou_use = margin_iou_map.get(level, margin_iou)

        # Make positive improvements count more than negatives
        pos_scale_iou_map = {"coarse": 1.5, "medium": 1.2, "fine": 1.0}
        neg_scale_iou = 0.25
        neg_clip_iou  = -0.15

        iou_term = _asym_delta(
            delta=float(delta_iou),
            margin=float(margin_iou_use),
            pos_scale=float(pos_scale_iou_map.get(level, 1.0)),
            neg_scale=float(neg_scale_iou),
            neg_clip=float(neg_clip_iou),
        )

        # --- Center improvement (normalized by TARGET bbox size, not full image) ---
        tgt_w = max(1.0, tgt_box[2] - tgt_box[0])
        tgt_h = max(1.0, tgt_box[3] - tgt_box[1])

        # IMPORTANT: normalize by target (w+h) with a floor for stability
        denom = max(200.0, tgt_w + tgt_h)  # tune floor: 150-300 are typical

        d_in   = _center_l1(input_box, tgt_box)
        d_pred = _center_l1(pred_box,  tgt_box)
        delta_ctr = (d_in - d_pred) / (denom + 1e-9)  # positive => got closer

        margin_center_px_map = {"coarse": 6.0, "medium": 4.0, "fine": 2.0}
        margin_ctr = float(margin_center_px_map.get(level, 4.0)) / (denom + 1e-9)

        pos_scale_ctr_map = {"coarse": 2.0, "medium": 1.2, "fine": 0.7}
        neg_scale_ctr = 0.20
        neg_clip_ctr  = -0.10

        ctr_term = _asym_delta(
            delta=float(delta_ctr),
            margin=float(margin_ctr),
            pos_scale=float(pos_scale_ctr_map.get(level, 1.0)),
            neg_scale=float(neg_scale_ctr),
            neg_clip=float(neg_clip_ctr),
        )

        # Combine: you can weight center more on coarse if you like
        w_iou_improve = {"coarse": 0.7, "medium": 0.8, "fine": 1.0}.get(level, 0.8)
        w_ctr_improve = {"coarse": 1.0, "medium": 0.7, "fine": 0.4}.get(level, 0.7)

        return float(w_iou_improve * iou_term + w_ctr_improve * ctr_term)

    except Exception:
        return 0.0


# ======================= Text reward ======================= #

def text_reward(predict_str: str, ground_truth: str) -> float:
    try:
        pred = _extract_json(predict_str)
        gt = _extract_json(ground_truth)
        if pred is None or gt is None:
            return 0.0

        output_text = str(pred.get("response", "")).strip()
        gt_response = str(gt.get("response", "")).strip().lower()
        if not gt_response:
            return 0.0

        if gt_response == "the object is found.":
            referring_keywords = ["is found", "is detected"]
            return 1.0 if any(k in output_text.lower() for k in referring_keywords) else 0.0

        if gt_response in ["a", "b", "c", "d", "A", "B", "C", "D"]:
            output = output_text.lower()
            option = gt_response.lower()
            pattern = rf"(?:\b|[\(\[\{{'\" ]){option}(?:\b|[\)\]\}}'\" ,.!?])"
            return 1.0 if re.search(pattern, output) else 0.0

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

def seg_iterative_compute_score_dense(
    predict_str: str,
    ground_truth: str,
    input_bbox=None,
    output_level: str = "medium",   # rec["output_level"]
    size_bucket: str = None         # rec["size_bucket"] (for points)
) -> float:
    """
    Main reward function to plug into VERL.

    Notes:
      - formatting rewards are kept small
      - bbox/points/text/decision/improve dominate
      - coverage is emphasized on coarse and fades by fine
    """
    print("------------------------------- predict_str -------------------------------")
    print(predict_str)
    thinking_format = iterative_thinking_format_reward(predict_str)
    seg_format = iterative_segmentation_format_reward(predict_str)

    iou = bbox_iou_reward(predict_str, ground_truth, level=output_level)
    box_l1 = bbox_l1_reward(predict_str, ground_truth, level=output_level)
    cov = bbox_coverage_reward(predict_str, ground_truth)
    points = point_weighted_reward(predict_str, ground_truth, size_bucket=size_bucket)
    text = text_reward(predict_str, ground_truth)
    decision = decision_reward(predict_str, ground_truth)
    improve = improvement_reward(
        predict_str, ground_truth,
        input_bbox=input_bbox,
        level=output_level,
    )

    # Level-dependent weights (coverage stronger early, IoU stronger later)
    w_cov = {"coarse": 2.0, "medium": 1.0, "fine": 0.3}.get(output_level, 1.0)

    # Your iou_reward is already level-thresholded; this weight is about relative importance
    w_iou = {"coarse": 1.2, "medium": 1.8, "fine": 2.2}.get(output_level, 1.8)
    w_l1  = {"coarse": 1.6, "medium": 2.0, "fine": 2.0}.get(output_level, 2.0)

    # Keep formatting rewards clearly smaller than task rewards
    w_fmt_think = 0.15
    w_fmt_seg   = 0.25

    # Encourage improvement shaping without dominating
    w_improve = {"coarse": 1.0, "medium": 0.8, "fine": 0.6}.get(output_level, 0.8)

    w_points = 1.0
    w_text     = {"coarse": 0.5, "medium": 0.7, "fine": 1.0}.get(output_level, 0.7)
    w_decision = {"coarse": 0.3, "medium": 1.0, "fine": 0.8}.get(output_level, 1.0)


    reward = (
        w_fmt_think * thinking_format
        + w_fmt_seg   * seg_format
        + w_iou       * iou
        + w_l1        * box_l1
        + w_cov       * cov
        + w_points    * points
        + w_text      * text
        + w_decision  * decision
        + w_improve   * improve
    )


    print("------------------------------- reward breakdown -------------------------------")
    print(
        f"output_level={output_level} | size_bucket={size_bucket}\n"
        f"  fmt_think : {thinking_format:.4f} (w={w_fmt_think}) -> {w_fmt_think*thinking_format:.4f}\n"
        f"  fmt_seg   : {seg_format:.4f} (w={w_fmt_seg}) -> {w_fmt_seg*seg_format:.4f}\n"
        f"  iou       : {iou:.4f} (w={w_iou}) -> {w_iou*iou:.4f}\n"
        f"  l1        : {box_l1:.4f} (w={w_l1}) -> {w_l1*box_l1:.4f}\n"
        f"  coverage  : {cov:.4f} (w={w_cov}) -> {w_cov*cov:.4f}\n"
        f"  points    : {points:.4f} (w={w_points}) -> {w_points*points:.4f}\n"
        f"  text      : {text:.4f} (w={w_text}) -> {w_text*text:.4f}\n"
        f"  decision  : {decision:.4f} (w={w_decision}) -> {w_decision*decision:.4f}\n"
        f"  improve   : {improve:.4f} (w={w_improve}) -> {w_improve*improve:.4f}\n"
        f"TOTAL: {reward:.4f}"
    )

    return reward
