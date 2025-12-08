import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pickle

RAW_JSON_PATH = Path("./experiments/coco/")
IMG_WIDTH = 960
IMG_HEIGHT = 540

# The 10 joints you mentioned, just for reference / debugging
JOINT_NAMES = [
    "head",
    "nose",
    "leftShoulder",
    "rightShoulder",
    "leftHip",
    "rightHip",
    "leftAnkle",
    "rightAnkle",
    "leftToe",
    "rightToe",
]

# For now we just map them to labels 0..9.
# If you later find the exact MHR-70 index mapping, you can plug that in here.
JOINT_LABELS = {name: idx for idx, name in enumerate(JOINT_NAMES)}

MHR70_MAP = {
    0: 69,   # head → neck
    1: 0,    # nose
    2: 5,    # left shoulder
    3: 6,    # right shoulder
    4: 9,    # left hip
    5: 10,   # right hip
    6: 13,   # left ankle
    7: 14,   # right ankle
    8: 15,   # left big toe tip
    9: 18,   # right big toe tip
}

def extract_raw_keypoints(skeleton: Dict[str, Any]) -> Optional[np.ndarray]:
    keypoint_names = [
        ("head", 0), ("nose", 2),
        ("leftShoulder", 12), ("rightShoulder", 6),
        ("leftHip", 24), ("rightHip", 18),
        ("leftAnkle", 28), ("rightAnkle", 22),
        ("leftFoot", 32), ("rightFoot", 30),
    ]
    coords = {}
    for person_id, kps in skeleton.items():
        kp = kps.get("keypoints")
        if kp is None:
            coords[person_id] = np.full((10, 2), np.nan)
            continue
        xy = []
        for _, idx in keypoint_names:
            xy.append(kp[idx])  # add x coord
            xy.append(kp[idx + 1])  # add y coord
        coords[person_id] = np.asarray(xy, dtype=float).reshape(10, 2)
    return coords

def build_save_bboxex_kps_all(output_path: Path):
    json_files = sorted(RAW_JSON_PATH.glob("*.json"))
    
    for json_path in json_files:

        stem_parts = json_path.stem.split("_")
        cam = vid = seg = None
        for part in stem_parts:
            if part.startswith("cam"):
                cam = int(part.replace("cam", ""))
            elif part.startswith("vid"):
                vid = int(part.replace("vid", ""))
            elif part.startswith("seg"):
                seg = int(part.replace("seg", ""))
        seg_name = f"{cam}{vid}{seg}"

        with open(json_path, "r") as file:
            raw_annotation = json.load(file)
        skeletons = raw_annotation.get("annotations", {}).get("skeletons", [])

        bboxes_kps = []
        for idx, skeleton in enumerate(skeletons):
            frame_coords = extract_raw_keypoints(skeleton)
            if frame_coords is None:
                continue
            bboxes, kps = build_bboxex_kps_single(frame_coords, IMG_WIDTH, IMG_HEIGHT)
            bboxes_kps.append({
                "bboxes": bboxes,
                "kps": kps,
            })
        
        # save to json
        if not output_path.exists():
            output_path.mkdir(parents=True)
        with open(output_path / f"{seg_name}.pkl", "wb") as file:
            pickle.dump(bboxes_kps, file)

def build_bboxex_kps_single(
    frame_coords,
    img_width,
    img_height,
    pad_ratio=0.15,
    min_valid_kps=3,
):
    """
    Build per-person bounding boxes and keypoint prompts for SAM-3D-Body
    from a dict of normalized 2D keypoints.

    Parameters
    ----------
    frame_coords : dict
        {person_id: (10, 2) ndarray} with x,y in [0,1] (normalized by imgW,imgH).
        NaNs are allowed and treated as missing keypoints.
    img_width : int
        Image width in pixels.
    img_height : int
        Image height in pixels.
    pad_ratio : float
        How much to expand the bbox, relative to its width/height.
    min_valid_kps : int
        Minimum number of valid keypoints required to keep a person.

    Returns
    -------
    bboxes : np.ndarray or None
        (num_person, 4) array of [x_min, y_min, x_max, y_max] in pixels,
        or None if no valid persons.
    keypoint_prompt : np.ndarray or None
        (num_person, 10, 3) array: [x_px, y_px, label].
        - x_px, y_px are pixel coordinates (still valid even if label is -2).
        - label in {0..9} for valid joints, -2 for invalid / missing joints.
        Returns None if no valid persons.

    Notes
    -----
    - This is designed to be passed into your modified `SAM3DBodyEstimator`
      where you convert from full-image pixels to crop space.
    - Labels -2 follow the SAM-3D-Body convention:
        label == -2 → invalid point (ignored by the prompt encoder)
    """

    person_ids = sorted(frame_coords.keys())
    all_bboxes = []
    all_keypoints = []

    for pid in person_ids:
        coords = np.asarray(frame_coords[pid], dtype=float)  # (10, 2)
        if coords.shape != (10, 2):
            raise ValueError(
                f"Expected (10,2) coords per person, got {coords.shape} for {pid}"
            )

        # Valid keypoints: both x and y finite (NaNs count as invalid)
        valid_mask = np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
        num_valid = int(valid_mask.sum())
        if num_valid < min_valid_kps:
            # Too few keypoints to build a stable bbox → skip this person
            continue

        # Clip normalized coords to [0,1] just in case
        coords_norm = np.clip(coords, 0.0, 1.0)

        # Convert to pixels
        xs = coords_norm[:, 0] * img_width
        ys = coords_norm[:, 1] * img_height
        xy_pix = np.stack([xs, ys], axis=-1)  # (10, 2)

        # Compute bbox from valid keypoints only
        xs_valid = xs[valid_mask]
        ys_valid = ys[valid_mask]

        x_min = xs_valid.min()
        x_max = xs_valid.max()
        y_min = ys_valid.min()
        y_max = ys_valid.max()

        # Add padding around bbox
        w = x_max - x_min
        h = y_max - y_min
        # If w or h is zero (all points at same place), give a minimum size
        if w <= 0:
            w = img_width * 0.02
        if h <= 0:
            h = img_height * 0.02

        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)

        pad_w = w * pad_ratio
        pad_h = h * pad_ratio

        x_min_p = max(0.0, cx - 0.5 * w - pad_w)
        x_max_p = min(float(img_width - 1), cx + 0.5 * w + pad_w)
        y_min_p = max(0.0, cy - 0.5 * h - pad_h)
        y_max_p = min(float(img_height - 1), cy + 0.5 * h + pad_h)

        bbox = np.array([x_min_p, y_min_p, x_max_p, y_max_p], dtype=np.float32)

        # Build keypoint [x_px, y_px, label] array
        labels = np.full((10,), -2.0, dtype=np.float32)  # -2 = invalid
        # Assign labels 0..9 ONLY to valid joints
        for j, name in enumerate(MHR70_MAP.keys()):
            if valid_mask[j]:
                labels[j] = float(MHR70_MAP[name])
        # Assign (0, 0) to invalid joints
        invalid_mask = (labels == -2)
        xy_pix[invalid_mask] = (0, 0)
        kps_with_labels = np.concatenate(
            [xy_pix.astype(np.float32), labels[:, None]], axis=-1
        )  # (10, 3)
        assert not np.any(np.isnan(kps_with_labels))

        all_bboxes.append(bbox)
        all_keypoints.append(kps_with_labels)

    if not all_bboxes:
        return None, None

    bboxes = np.stack(all_bboxes, axis=0)           # (num_person, 4)
    keypoint_prompt = np.stack(all_keypoints, axis=0)  # (num_person, 10, 3)

    return bboxes, keypoint_prompt

# def build_user_bboxes_and_keypoints(
#     frame_coords,
#     img_width,
#     img_height,
#     pad_ratio=0.15,
#     min_valid_kps=3,
# ):
#     """
#     frame_coords: {person_id: (10,2) ndarray} with normalized x,y in [0,1].
#     Returns:
#       user_bboxes: (num_user, 4) or None
#       user_kps:    (num_user, 10, 3) or None   [x_px, y_px, mhr_label]
#     """
#     if frame_coords is None or len(frame_coords) == 0:
#         return None, None

#     person_ids = sorted(frame_coords.keys())
#     all_bboxes = []
#     all_kps = []

#     for pid in person_ids:
#         coords = np.asarray(frame_coords[pid], dtype=float)  # (10,2)
#         if coords.shape != (10, 2):
#             raise ValueError(
#                 f"Expected (10,2) coords per person, got {coords.shape} for person {pid}"
#             )

#         # valid if both x,y finite
#         valid_mask = np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
#         num_valid = int(valid_mask.sum())
#         if num_valid < min_valid_kps:
#             # skip if too few
#             continue

#         # clip and convert to px
#         coords_norm = np.clip(coords, 0.0, 1.0)
#         xs = coords_norm[:, 0] * img_width
#         ys = coords_norm[:, 1] * img_height
#         xy_pix = np.stack([xs, ys], axis=-1).astype(np.float32)  # (10,2)

#         xs_valid = xs[valid_mask]
#         ys_valid = ys[valid_mask]

#         x_min = xs_valid.min()
#         x_max = xs_valid.max()
#         y_min = ys_valid.min()
#         y_max = ys_valid.max()

#         w = x_max - x_min
#         h = y_max - y_min
#         if w <= 0:
#             w = img_width * 0.02
#         if h <= 0:
#             h = img_height * 0.02

#         cx = 0.5 * (x_min + x_max)
#         cy = 0.5 * (y_min + y_max)

#         pad_w = w * pad_ratio
#         pad_h = h * pad_ratio

#         x_min_p = max(0.0, cx - 0.5 * w - pad_w)
#         x_max_p = min(float(img_width - 1), cx + 0.5 * w + pad_w)
#         y_min_p = max(0.0, cy - 0.5 * h - pad_h)
#         y_max_p = min(float(img_height - 1), cy + 0.5 * h + pad_h)

#         bbox = np.array([x_min_p, y_min_p, x_max_p, y_max_p], dtype=np.float32)

#         # [x, y, label]; default label -2 = invalid
#         labels = np.full((10,), -2.0, dtype=np.float32)
#         for j in range(10):
#             if valid_mask[j]:
#                 labels[j] = float(MHR70_MAP[j])

#         kps_with_labels = np.concatenate([xy_pix, labels[:, None]], axis=-1)  # (10,3)

#         all_bboxes.append(bbox)
#         all_kps.append(kps_with_labels)

#     if not all_bboxes:
#         return None, None

#     user_bboxes = np.stack(all_bboxes, axis=0)   # (num_user, 4)
#     user_kps = np.stack(all_kps, axis=0)         # (num_user, 10, 3)

#     return user_bboxes, user_kps


if __name__ == "__main__":
    build_save_bboxex_kps_all(Path("./experiments/bboxex_kps/"))