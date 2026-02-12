import json
from datasets import Dataset, load_dataset
import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from loguru import logger
from PIL import Image

# ============================================================
# Global config
# ============================================================

# Area bucket thresholds (area_ratio = object_area / image_area)
XXS_TH = 0.00017
S_TH = 0.00055

# How many samples per (coarse->medium, medium->fine) transition
SAMPLES_PER_TRANSITION = 1

# Input-box level configs: how imperfect the input box is
# NOTE: no min_iou here
# bucket-dependent input configs (reduced aspect_jitter)
INPUT_LEVEL_CONFIG = {
    "xxs": {
        "coarse": {"scale_min": 9.0,  "scale_max": 15.0, "center_jitter": 0.30, "aspect_jitter": 0.20},
        "medium": {"scale_min": 5.0,  "scale_max": 9.0, "center_jitter": 0.22, "aspect_jitter": 0.15},
        "fine":   {"scale_min": 1.0,  "scale_max": 2.0,  "center_jitter": 0.12, "aspect_jitter": 0.10},
    },
    "xs": {
        "coarse": {"scale_min": 5.0,  "scale_max": 9.0,  "center_jitter": 0.25, "aspect_jitter": 0.20},
        "medium": {"scale_min": 3.0,  "scale_max": 5.0,  "center_jitter": 0.18, "aspect_jitter": 0.15},
        "fine":   {"scale_min": 1.0,  "scale_max": 1.8,  "center_jitter": 0.10, "aspect_jitter": 0.07},
    },
    "s": {
        "coarse": {"scale_min": 2.0,  "scale_max": 3.0,  "center_jitter": 0.20, "aspect_jitter": 0.25},
        "medium": {"scale_min": 1.3,  "scale_max": 2.0,  "center_jitter": 0.15, "aspect_jitter": 0.15},
        "fine":   {"scale_min": 1.05, "scale_max": 1.3,  "center_jitter": 0.10, "aspect_jitter": 0.10},
    },
}

# Target/output box config per area bucket + level:
# base_scale: how much to scale the GT side length
# scale_jitter: multiplicative jitter around base_scale
# aspect_jitter: how much to distort width vs height
TARGET_LEVEL_CONFIG = {
    "xxs": {
        # input coarse: 9–15
        "coarse": {"base_scale": 12.0, "scale_jitter": 0.25,      "aspect_jitter": 0.15},

        # input medium: 5–9
        "medium": {"base_scale": 7.0,  "scale_jitter": 0.285714,  "aspect_jitter": 0.10},

        # input fine: 1–2
        "fine":   {"base_scale": 1.5,  "scale_jitter": 0.333333,  "aspect_jitter": 0.05},
    },

    "xs": {
        # input coarse: 5–9
        "coarse": {"base_scale": 7.0,  "scale_jitter": 0.285714,  "aspect_jitter": 0.15},

        # input medium: 3–5
        "medium": {"base_scale": 4.0,  "scale_jitter": 0.25,      "aspect_jitter": 0.10},

        # input fine: 1–1.8
        "fine":   {"base_scale": 1.4,  "scale_jitter": 0.285714,  "aspect_jitter": 0.05},
    },

    "s": {
        # input coarse: 2–3
        "coarse": {"base_scale": 2.5,  "scale_jitter": 0.2,       "aspect_jitter": 0.15},

        # input medium: 1.3–2
        "medium": {"base_scale": 1.65, "scale_jitter": 0.212121,  "aspect_jitter": 0.10},

        # input fine: 1.05–1.3
        "fine":   {"base_scale": 1.175,"scale_jitter": 0.106383,  "aspect_jitter": 0.05},
    },
}

# ============================================================
# Basic geometry helpers
# ============================================================

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def bbox_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

# ============================================================
# Image / mask helpers (adapted from your code)
# ============================================================

def load_image_as_bytes(path, image_name, new_size):
    """
    new_size: (width, height)
    returns PIL.Image resized to new_size.
    """
    image_path = os.path.join(path, image_name)
    if os.path.exists(image_path) and os.path.isfile(image_path):
        try:
            image_np = cv2.imread(image_path)
            img = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
            img = img.resize(new_size)  # (width, height)
            return img
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    else:
        print("Path not exist:{}".format(image_path))
        return None

def get_mask_from_points(points, image_path):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    label_value = 1  # target

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
    cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)
    
    return mask, height, width

def get_bbox_from_mask(points, image_path, new_size, resize_box=False):
    """
    Compute bbox from polygon mask, then rescale to new_size.
    new_size: (new_height, new_width)
    returns (x_min, y_min, x_max, y_max) in resized coords.
    """
    mask, height, width = get_mask_from_points(points, image_path)
    
    y_coords, x_coords = np.nonzero(mask)  
    x_min = x_coords.min()  
    x_max = x_coords.max()  
    y_min = y_coords.min()  
    y_max = y_coords.max()
    
    new_h, new_w = new_size
    scale_x = new_w / width
    scale_y = new_h / height
    
    new_x_min = x_min * scale_x
    new_y_min = y_min * scale_y
    new_x_max = x_max * scale_x
    new_y_max = y_max * scale_y
    
    return [new_x_min, new_y_min, new_x_max, new_y_max]

def get_id_from_name(name):
    return name.split(".")[0]

# ============================================================
# RL geometry helpers (buckets + jitter)
# ============================================================

def get_size_bucket(area_ratio: float) -> str:
    if area_ratio < XXS_TH:
        return "xxs"
    elif area_ratio > S_TH:
        return "s"
    else:
        return "xs"

def sample_input_box_for_level(
    gt_box,
    img_w,
    img_h,
    level,
    area_ratio
):
    """
    Imperfect input bbox (center + aspect jitter), bucket-aware.

    You can pass either:
      - bucket: "xxs" | "xs" | "s"
      - OR area_ratio: object_area / image_area  (will be bucketized)

    NO min IoU condition – just returns a jittered box.
    """

    bucket = get_size_bucket(area_ratio)  # uses XXS_TH / S_TH

    cfg = INPUT_LEVEL_CONFIG[bucket][level]

    x1, y1, x2, y2 = gt_box
    gw = max(1.0, x2 - x1)
    gh = max(1.0, y2 - y1)
    gcx = 0.5 * (x1 + x2)
    gcy = 0.5 * (y1 + y2)

    scale = random.uniform(cfg["scale_min"], cfg["scale_max"])
    aspect = random.uniform(1.0 - cfg["aspect_jitter"], 1.0 + cfg["aspect_jitter"])

    w = gw * scale * aspect
    h = gh * scale / aspect

    dx = random.uniform(-cfg["center_jitter"], cfg["center_jitter"]) * gw
    dy = random.uniform(-cfg["center_jitter"], cfg["center_jitter"]) * gh
    cx = gcx + dx
    cy = gcy + dy

    x1_in = clamp(cx - 0.5 * w, 0, img_w - 1)
    y1_in = clamp(cy - 0.5 * h, 0, img_h - 1)
    x2_in = clamp(cx + 0.5 * w, x1_in + 1, img_w)
    y2_in = clamp(cy + 0.5 * h, y1_in + 1, img_h)

    return [x1_in, y1_in, x2_in, y2_in]

def sample_target_box_for_level(gt_box, level, area_ratio, img_w, img_h):
    """
    Output/target box:
    - fixed center at GT
    - bucket+level-dependent scale & jitters (from TARGET_LEVEL_CONFIG)
    - slight aspect jitter, no center jitter
    """
    x1, y1, x2, y2 = gt_box
    gw = max(1.0, x2 - x1)
    gh = max(1.0, y2 - y1)
    gcx = 0.5 * (x1 + x2)
    gcy = 0.5 * (y1 + y2)

    bucket = get_size_bucket(area_ratio)
    cfg = TARGET_LEVEL_CONFIG[bucket][level]

    base_scale = cfg["base_scale"]
    scale_jitter = cfg["scale_jitter"]
    aspect_jitter = cfg["aspect_jitter"]

    scale = random.uniform((1.0 - scale_jitter) * base_scale,
                           (1.0 + scale_jitter) * base_scale)
    aspect = random.uniform(1.0 - aspect_jitter, 1.0 + aspect_jitter)

    w = gw * scale * aspect
    h = gh * scale / aspect

    x1_out = gcx - 0.5 * w
    y1_out = gcy - 0.5 * h
    x2_out = gcx + 0.5 * w
    y2_out = gcy + 0.5 * h

    x1_out = clamp(x1_out, 0, img_w - 1)
    y1_out = clamp(y1_out, 0, img_h - 1)
    x2_out = clamp(x2_out, x1_out + 1, img_w)
    y2_out = clamp(y2_out, y1_out + 1, img_h)

    return [x1_out, y1_out, x2_out, y2_out]

def sample_points_from_bbox(box):
    """
    Two points inside the bbox: center + jittered point.
    """
    x_min, y_min, x_max, y_max = box
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    dx = (x_max - x_min) / 10.0
    dy = (y_max - y_min) / 10.0
    px = min(max(cx + random.uniform(-dx, dx), x_min), x_max)
    py = min(max(cy + random.uniform(-dy, dy), y_min), y_max)
    return [[cx, cy], [px, py]]

def build_answer_json(
    target_bbox,
    target_points,
    response,
    decision,
    reason,
):
    data = {
        "bbox_2d": [int(round(v)) for v in target_bbox],
        "points_2d": [[float(p[0]), float(p[1])] for p in target_points],
        "response": response,
        "decision": decision,
        "reason": reason,
    }
    return json.dumps(data, ensure_ascii=False)

def normalize_gt_response(ann):
    """
    What text goes into the 'response' field of the JSON.
    """
    q_type = ann["Q-type"]
    if q_type == "option":
        return "{}".format(ann["A"])
    elif q_type == "referring":
        return "The object is found."
    else:
        return "{}".format(ann["A"])

# ============================================================
# Main conversion function (JSON -> iterative RL parquet)
# ============================================================

def finers_hf2parquet_iterative(
    save_path,
    split,
    num_limit,
    all_image_path,
    new_size,   # (width, height)
    chunk_size,
    hf_dataset_name: str = "Jiazuo98/Finers-4k-benchmark",
):
    """
    Read Finers-4k from HuggingFace (hf_dataset_name, split),
    load images from all_image_path, and write iterative RL
    data to Parquet in streaming chunks.
    """
    logger.info(f"Resize image to: {new_size}")
    new_w, new_h = new_size
    image_area = float(new_w * new_h)

    ds = load_dataset(hf_dataset_name, split=split)

    records = []
    count = 0
    chunk_idx = 0

    os.makedirs(save_path, exist_ok=True)

    for row in tqdm(ds, desc=f"Processing {split}"):
        ann = row["annotations"]

        img_id = get_id_from_name(ann["image_path"])
        problem = ann["Q"]
        q_type = ann["Q-type"]
        options = ann.get("options")
        gt_response_text = normalize_gt_response(ann)

        # Load resized image
        image_data = load_image_as_bytes(all_image_path, ann["image_path"], new_size)
        if image_data is None:
            continue

        # GT bbox in resized coords
        gt_bbox = get_bbox_from_mask(
            ann["points"],
            os.path.join(all_image_path, ann["image_path"]),
            (new_size[1], new_size[0])  # (new_height, new_width)
        )

        obj_area = bbox_area(gt_bbox)
        area_ratio = obj_area / (image_area + 1e-9)
        bucket = get_size_bucket(area_ratio)

        # ------------------ INITIAL: None -> coarse ------------------
        output_level_init = "coarse"
        target_bbox_init = sample_target_box_for_level(
            gt_box=gt_bbox,
            level=output_level_init,
            area_ratio=area_ratio,
            img_w=new_w,
            img_h=new_h,
        )
        target_points_init = sample_points_from_bbox(target_bbox_init)

        answer_init = build_answer_json(
            target_bbox=target_bbox_init,
            target_points=target_points_init,
            response=gt_response_text,
            decision="refine",
            reason="Initial coarse localization around the object.",
        )

        rec_init = {
            "id": img_id,
            "image": image_data,
            "problem": problem,
            "type": q_type,
            "options": options if q_type == "option" else None,

            "stage": "initial",
            "input_level": None,
            "output_level": output_level_init,
            "size_bucket": bucket,

            "gt_bbox_resized": gt_bbox,
            "input_bbox": None,
            "target_bbox": target_bbox_init,
            "target_points": target_points_init,

            "gt_answer_text": gt_response_text,
            "answer": answer_init,
        }
        records.append(rec_init)
        count += 1

        if len(records) >= chunk_size:
            ds_chunk = Dataset.from_list(records)
            chunk_path = os.path.join(save_path, f"{split}_{chunk_idx}.parquet")
            ds_chunk.to_parquet(chunk_path)
            logger.info(f"Saved chunk: {chunk_path}")
            records = []
            chunk_idx += 1

        if num_limit is not None and count >= num_limit:
            break

        # ------------------ REFINE transitions ------------------
        transitions = [
            ("coarse", "medium"),
            ("medium", "fine"),
        ]

        for input_level, output_level in transitions:
            for _ in range(SAMPLES_PER_TRANSITION):
                input_bbox = sample_input_box_for_level(
                    gt_box=gt_bbox,
                    img_w=new_w,
                    img_h=new_h,
                    level=input_level,
                    area_ratio=area_ratio,
                )

                target_bbox = sample_target_box_for_level(
                    gt_box=gt_bbox,
                    level=output_level,
                    area_ratio=area_ratio,
                    img_w=new_w,
                    img_h=new_h,
                )
                target_points = sample_points_from_bbox(target_bbox)

                if output_level == "fine":
                    decision = "stop"
                    reason = "Refined to the fine level; no further refinement is needed."
                else:
                    decision = "refine"
                    reason = f"Refining from {input_level} level to {output_level} level."

                answer_json = build_answer_json(
                    target_bbox=target_bbox,
                    target_points=target_points,
                    response=gt_response_text,
                    decision=decision,
                    reason=reason,
                )

                rec = {
                    "id": img_id,
                    "image": image_data,
                    "problem": problem,
                    "type": q_type,
                    "options": options if q_type == "option" else None,

                    "stage": "refine",
                    "input_level": input_level,
                    "output_level": output_level,
                    "size_bucket": bucket,

                    "gt_bbox_resized": gt_bbox,
                    "input_bbox": input_bbox,
                    "target_bbox": target_bbox,
                    "target_points": target_points,

                    "gt_answer_text": gt_response_text,
                    "answer": answer_json,
                }
                records.append(rec)
                count += 1

                if len(records) >= chunk_size:
                    ds_chunk = Dataset.from_list(records)
                    chunk_path = os.path.join(save_path, f"{split}_{chunk_idx}.parquet")
                    ds_chunk.to_parquet(chunk_path)
                    logger.info(f"Saved chunk: {chunk_path}")
                    records = []
                    chunk_idx += 1

                if num_limit is not None and count >= num_limit:
                    break
            if num_limit is not None and count >= num_limit:
                break

        if num_limit is not None and count >= num_limit:
            break

    # flush any remaining records
    if records:
        ds_chunk = Dataset.from_list(records)
        chunk_path = os.path.join(save_path, f"{split}_{chunk_idx}.parquet")
        ds_chunk.to_parquet(chunk_path)
        logger.info(f"Saved final chunk: {chunk_path}")

    print(f"Iterative RL dataset saved to {save_path}")

# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    all_image_path = "/scratch/svaidy33/hf_cache/datasets/all_images/all_images/"
    save_path = "/scratch/svaidy33/hf_cache/datasets/all_parquet/finers_iterative_rl_1920x1080_1per_transition_scale/"

    new_size = (1920, 1080)  # (width, height)
    chunk_size = 20

    finers_hf2parquet_iterative(
        save_path=save_path,
        split="train",           # "validation" or "test" as needed
        num_limit=None,          # or an int for debugging
        all_image_path=all_image_path,
        new_size=new_size,
        chunk_size=chunk_size,
    )