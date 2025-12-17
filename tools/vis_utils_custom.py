import cv2
import numpy as np
import torch
import os
import json
from ait_view import visualize_multi_person, read_vertices_and_faces
import pickle
import sys
import types

# Compatibility: allow loading pickles that reference numpy._core (NumPy 2.x)
if "numpy._core" not in sys.modules:
    np_core = types.ModuleType("numpy._core")
    np_core.numeric = np.core.numeric
    sys.modules["numpy._core"] = np_core
    sys.modules["numpy._core.numeric"] = np.core.numeric
    if hasattr(np.core, "multiarray"):
        sys.modules["numpy._core.multiarray"] = np.core.multiarray

def kp_check(filename, kps):
    assert isinstance(kps, np.ndarray), "Invalid keypoints type"
    assert kps.shape[1] == 10, "Invalid keypoints shape"

    labels = kps[..., -1]
    invalid_mask = (labels == -2)
    kps[invalid_mask] = 0
    kps[invalid_mask] = 1
    return kps

def plot_bboxes_kps(img, bboxes, kps):
    img_with_bboxes_kps = img.copy()
    if bboxes is not None and kps is not None:
        # one-person bbox and kp
        assert bboxes.shape[0] == kps.shape[0], "Number of bboxes and kps must be the same"
        for i in range(bboxes.shape[0]):
            bbox_person = bboxes[i, ...]
            kp_person = kps[i, ..., :2]
            # specify color for each person
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            cv2.rectangle(img_with_bboxes_kps, (int(bbox_person[0]), int(bbox_person[1])), (int(bbox_person[2]), int(bbox_person[3])), color, 2)
            for kp in kp_person :
                cv2.circle(img_with_bboxes_kps, (int(kp[0]), int(kp[1])), 3, color, -1)

    return img_with_bboxes_kps

def plot_bbox_test_image(img_folder, pkl_folder, output_folder):
    for img_file in os.listdir(img_folder):
        seg_name = img_file.split(".")[0].split("_")[0]
        kp_idx = int(img_file.split(".")[0].split("_")[1])

        pkl_file = os.path.join(pkl_folder, f"{seg_name}.pkl")
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        bboxes, kps = data[kp_idx]["bboxes"], data[kp_idx]["kps"]
        
        img = cv2.imread(os.path.join(img_folder, img_file))
        img_with_bboxes_kps = plot_bboxes_kps(img, bboxes, kps)
        cv2.imwrite(os.path.join(output_folder, img_file.replace(".jpg", f"_bbox_kps_{kp_idx}.jpg")), img_with_bboxes_kps)

def bbox_iou(box_a, box_b):
    """
    box_a, box_b: [x_min, y_min, x_max, y_max]
    returns scalar IoU
    """
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    denom = area_a + area_b - inter_area + 1e-6
    return inter_area / denom

# MHR70 keypoint indices for robust PnP (excluding head/neck)
# From MHR70_MAP: nose(0), shoulders(5,6), hips(9,10), ankles(13,14), toes(15,18)
PNP_ROBUST_INDICES = [0, 5, 6, 9, 10, 13, 14, 15, 18]
PNP_JOINT_NAMES = ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip", 
                   "left_ankle", "right_ankle", "left_toe", "right_toe"]

# Foot keypoint indices in MHR70 for ground plane fitting
# ankles(13,14) + big toe tips(15,18) 
FOOT_INDICES = [13, 14, 15, 18]
FOOT_NAMES = ["left_ankle", "right_ankle", "left_toe", "right_toe"]

# World coordinate system convention
# Your calibration uses Z-up (ground at Z=0), which is common in robotics/CV
# Change to [0., 1., 0.] if your world uses Y-up (common in graphics)
WORLD_UP = np.array([0., 0., 1.])  # Z-up: ground plane is Z=0


# =============================================================================
# Camera Extrinsics and Ray-Ground Intersection Functions
# =============================================================================

def load_camera_extrinsics(extrinsic_file):
    """
    Load camera extrinsics from JSON file.
    
    Expected format:
    {
        "R": [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]],  # or "rotation"
        "t": [tx, ty, tz],  # or "translation"
        // Optional: "rvec": [rx, ry, rz] (Rodrigues vector)
    }
    
    Convention: P_camera = R @ P_world + t
    
    Returns:
        R_cam: (3, 3) rotation matrix (world to camera)
        t_cam: (3,) translation vector (world to camera)
    """
    with open(extrinsic_file, "r") as f:
        data = json.load(f)
    
    # Try different key names
    if "R" in data:
        R_cam = np.array(data["R"], dtype=np.float64)
    elif "rotation" in data:
        R_cam = np.array(data["rotation"], dtype=np.float64)
    elif "rvec" in data:
        rvec = np.array(data["rvec"], dtype=np.float64)
        R_cam, _ = cv2.Rodrigues(rvec)
    else:
        raise KeyError("Could not find rotation in extrinsics file (expected 'R', 'rotation', or 'rvec')")
    
    if "t" in data:
        t_cam = np.array(data["t"], dtype=np.float64).flatten()
    elif "translation" in data:
        t_cam = np.array(data["translation"], dtype=np.float64).flatten()
    else:
        raise KeyError("Could not find translation in extrinsics file (expected 't' or 'translation')")
    
    return R_cam, t_cam


def get_ground_plane_in_camera(R_cam, t_cam, world_up=np.array([0., 0., 1.])):
    """
    Get the ground plane expressed in camera coordinates.
    
    Args:
        R_cam: (3, 3) rotation matrix (world to camera)
        t_cam: (3,) translation vector (world to camera)
        world_up: (3,) world up direction 
            - [0, 0, 1] for Z-up (ground at Z=0) - common in robotics/CV
            - [0, 1, 0] for Y-up (ground at Y=0) - common in graphics
        
    Returns:
        n_cam: (3,) plane normal in camera coords (points "up" from ground)
        d_cam: scalar such that n_cam · P_cam + d_cam = 0 for points on ground
    """
    # Transform world up direction to camera coords
    n_cam = R_cam @ world_up
    n_cam = n_cam / np.linalg.norm(n_cam)
    
    # A point on the ground in world coords is origin [0, 0, 0]
    # (works for both Y=0 and Z=0 ground planes)
    ground_point_world = np.array([0., 0., 0.])
    ground_point_cam = R_cam @ ground_point_world + t_cam
    
    # Plane equation: n · (P - P0) = 0  =>  n · P = n · P0
    # In form n · P + d = 0:  d = -n · P0
    d_cam = -np.dot(n_cam, ground_point_cam)
    
    return n_cam, d_cam


def backproject_to_ray(uv, K):
    """
    Back-project a 2D pixel to a 3D ray in camera coordinates.
    
    Args:
        uv: (2,) or (N, 2) pixel coordinates
        K: (3, 3) camera intrinsic matrix
        
    Returns:
        ray_dir: (3,) or (N, 3) normalized ray direction in camera coords
                 (ray starts at camera origin [0,0,0])
    """
    uv = np.atleast_2d(uv)
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Normalized image coordinates
    x = (uv[:, 0] - cx) / fx
    y = (uv[:, 1] - cy) / fy
    z = np.ones_like(x)
    
    ray_dir = np.stack([x, y, z], axis=-1)
    ray_dir = ray_dir / np.linalg.norm(ray_dir, axis=-1, keepdims=True)
    
    return ray_dir.squeeze()


def ray_plane_intersection(ray_origin, ray_dir, plane_normal, plane_d):
    """
    Find intersection of ray with plane.
    
    Ray: P = ray_origin + t * ray_dir
    Plane: plane_normal · P + plane_d = 0
    
    Args:
        ray_origin: (3,) or (N, 3) ray origin
        ray_dir: (3,) or (N, 3) ray direction (normalized)
        plane_normal: (3,) plane normal
        plane_d: scalar plane offset
        
    Returns:
        intersection: (3,) or (N, 3) intersection point
        t: scalar or (N,) parameter along ray (negative means behind camera)
    """
    ray_origin = np.atleast_2d(ray_origin)
    ray_dir = np.atleast_2d(ray_dir)
    
    # n · (O + t*D) + d = 0
    # n · O + t * (n · D) + d = 0
    # t = -(n · O + d) / (n · D)
    
    denom = ray_dir @ plane_normal  # (N,)
    numer = -(ray_origin @ plane_normal + plane_d)  # (N,)
    
    # Handle parallel rays (denom ≈ 0)
    t = np.where(np.abs(denom) > 1e-10, numer / denom, np.inf)
    
    intersection = ray_origin + t[:, None] * ray_dir
    
    return intersection.squeeze(), t.squeeze()


def project_2d_to_ground(uv_2d, K, R_cam, t_cam, world_up=np.array([0., 0., 1.])):
    """
    Project 2D pixel coordinates onto the ground plane.
    
    This is the key function: given a 2D foot keypoint, find where it is
    on the ground in 3D world coordinates.
    
    Args:
        uv_2d: (2,) or (N, 2) pixel coordinates
        K: (3, 3) camera intrinsic matrix
        R_cam: (3, 3) rotation (world to camera)
        t_cam: (3,) translation (world to camera)
        world_up: (3,) world up direction
            - [0, 0, 1] for Z-up (ground at Z=0)
            - [0, 1, 0] for Y-up (ground at Y=0)
        
    Returns:
        P_world: (3,) or (N, 3) 3D point on ground in world coordinates
        valid: bool or (N,) mask indicating valid intersections (not behind camera)
    """
    uv_2d = np.atleast_2d(uv_2d)
    
    # Get ground plane in camera coords
    n_cam, d_cam = get_ground_plane_in_camera(R_cam, t_cam, world_up)
    
    # Back-project to rays
    ray_dir = backproject_to_ray(uv_2d, K)
    ray_dir = np.atleast_2d(ray_dir)
    ray_origin = np.zeros((len(ray_dir), 3))  # camera origin
    
    # Intersect with ground plane
    P_cam, t_param = ray_plane_intersection(ray_origin, ray_dir, n_cam, d_cam)
    P_cam = np.atleast_2d(P_cam)
    t_param = np.atleast_1d(t_param)
    
    # Transform to world coordinates: P_world = R_cam^T @ (P_cam - t_cam)
    R_cam_inv = R_cam.T
    P_world = (R_cam_inv @ (P_cam - t_cam).T).T
    
    # Valid if intersection is in front of camera
    valid = t_param > 0
    
    return P_world.squeeze(), valid.squeeze()


def align_person_to_ground_extrinsics(
    J_3d_body, V_3d_body, J_2d, K, R_cam, t_cam, 
    foot_indices=FOOT_INDICES, world_up=np.array([0., 0., 1.]),
    verbose=False
):
    """
    Align a person's 3D pose to the ground using camera extrinsics.
    
    Strategy:
    1. Project 2D foot keypoints onto the ground plane → get ground-truth 3D foot positions
    2. Use Procrustes alignment (R, t, scale) to match body feet to ground-projected feet
    3. Apply the same transform to full body mesh
    
    Args:
        J_3d_body: (70, 3) 3D keypoints in body-canonical frame (from SAM3D)
        V_3d_body: (V, 3) vertices in body-canonical frame
        J_2d: (70, 2) 2D keypoints in image
        K: (3, 3) camera intrinsic matrix
        R_cam: (3, 3) camera rotation (world to camera)
        t_cam: (3,) camera translation (world to camera)
        foot_indices: list of foot keypoint indices
        world_up: (3,) world up direction
            - [0, 0, 1] for Z-up (ground at Z=0) - your calibration uses this
            - [0, 1, 0] for Y-up (ground at Y=0)
        verbose: whether to print debug info
        
    Returns:
        J_world: (70, 3) aligned keypoints in world coordinates
        V_world: (V, 3) aligned vertices in world coordinates  
        alignment_info: dict with alignment details
    """
    # Step 1: Get ground-truth foot positions by projecting 2D onto ground plane
    foot_2d = J_2d[foot_indices]  # (4, 2)
    foot_world_gt, foot_valid = project_2d_to_ground(foot_2d, K, R_cam, t_cam, world_up)
    foot_world_gt = np.atleast_2d(foot_world_gt)
    foot_valid = np.atleast_1d(foot_valid)
    
    if verbose:
        up_axis = np.argmax(np.abs(world_up))
        up_name = ['X', 'Y', 'Z'][up_axis]
        print(f"  Ground-truth foot positions (world, {up_name}-up):")
        for i, (name, pos, valid) in enumerate(zip(FOOT_NAMES, foot_world_gt, foot_valid)):
            status = "OK" if valid else "BEHIND_CAMERA"
            print(f"    {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] ({status})")
    
    if not foot_valid.any():
        print("  WARNING: All foot projections behind camera!")
        return None, None, {"success": False, "reason": "all_feet_behind_camera"}
    
    # Step 2: Get foot positions in body frame
    foot_body = J_3d_body[foot_indices]  # (4, 3)
    
    # Step 3: Solve for rigid transform (R, t, s) that maps body feet to world feet
    # Using Procrustes alignment with scale
    
    # Use only valid feet
    valid_foot_body = foot_body[foot_valid]
    valid_foot_world = foot_world_gt[foot_valid]
    
    if len(valid_foot_body) < 2:
        print("  WARNING: Need at least 2 valid feet for alignment")
        return None, None, {"success": False, "reason": "insufficient_feet"}
    
    # Procrustes: find R, t, s such that s * R @ body + t ≈ world
    # Center both point sets
    centroid_body = valid_foot_body.mean(axis=0)
    centroid_world = valid_foot_world.mean(axis=0)
    
    body_centered = valid_foot_body - centroid_body
    world_centered = valid_foot_world - centroid_world
    
    # Compute scale
    scale_body = np.sqrt((body_centered ** 2).sum())
    scale_world = np.sqrt((world_centered ** 2).sum())
    s = scale_world / (scale_body + 1e-10)
    
    # Compute rotation using SVD
    H = body_centered.T @ world_centered
    U, _, Vt = np.linalg.svd(H)
    R_align = Vt.T @ U.T
    
    # Handle reflection
    if np.linalg.det(R_align) < 0:
        Vt[-1, :] *= -1
        R_align = Vt.T @ U.T
    
    # Compute translation
    t_align = centroid_world - s * R_align @ centroid_body
    
    # Step 4: Apply transform to all joints and vertices
    J_world = s * (R_align @ J_3d_body.T).T + t_align
    V_world = s * (R_align @ V_3d_body.T).T + t_align
    
    # Step 5: Verify alignment
    foot_world_aligned = J_world[foot_indices]
    foot_errors = np.linalg.norm(foot_world_aligned[foot_valid] - valid_foot_world, axis=1)
    
    alignment_info = {
        "success": True,
        "scale": s,
        "R_align": R_align,
        "t_align": t_align,
        "foot_errors": foot_errors,
        "mean_foot_error": foot_errors.mean(),
        "max_foot_error": foot_errors.max(),
        "num_valid_feet": foot_valid.sum(),
        "foot_world_gt": foot_world_gt,
        "foot_world_aligned": foot_world_aligned,
    }
    
    if verbose:
        # Determine up axis (Z=2 for Z-up, Y=1 for Y-up)
        up_axis = np.argmax(np.abs(world_up))
        up_name = ['X', 'Y', 'Z'][up_axis]
        print(f"  Alignment result:")
        print(f"    Scale factor: {s:.4f}")
        print(f"    Mean foot error: {foot_errors.mean()*100:.2f} cm")
        print(f"    Max foot error: {foot_errors.max()*100:.2f} cm")
        print(f"    Aligned foot {up_name} (should be ~0): {foot_world_aligned[:,up_axis].mean():.4f}")
    
    return J_world, V_world, alignment_info


def sanity_check_2d(J_3d, J_2d, R, t, K, joint_names=None):
    """
    Check reprojection error for PnP result.
    
    Args:
        J_3d: (N, 3) 3D keypoints used for PnP
        J_2d: (N, 2) 2D keypoints used for PnP  
        R: (3, 3) rotation matrix from PnP
        t: (3,) translation vector from PnP
        K: (3, 3) camera intrinsic matrix
        joint_names: optional list of joint names for detailed output
        
    Returns:
        mean_err: mean reprojection error in pixels
        max_err: max reprojection error in pixels
        per_joint_errors: (N,) array of per-joint errors
    """
    # === Reproject all joints ===
    J_cam = (R @ J_3d.T).T + t   # (N,3)

    # Project with true intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X = J_cam[:, 0]
    Y = J_cam[:, 1]
    Z = J_cam[:, 2].clip(min=1e-6)

    x_proj = fx * (X / Z) + cx
    y_proj = fy * (Y / Z) + cy
    proj_2d = np.stack([x_proj, y_proj], axis=-1)

    # === Compare ===
    diff = proj_2d - J_2d
    per_joint_errors = np.linalg.norm(diff, axis=-1)   # pixel error per joint
    mean_err = per_joint_errors.mean()
    max_err = per_joint_errors.max()

    return mean_err, max_err, per_joint_errors


def solve_pnp_person(output, K, dist_coeffs, person_idx=0, verbose=True):
    """
    Solve PnP for a single person using robust body keypoints.
    
    Args:
        output: dict with 'pred_keypoints_3d' and 'pred_keypoints_2d'
        K: (3, 3) camera intrinsic matrix (should match 2D keypoint resolution)
        dist_coeffs: distortion coefficients
        person_idx: index for logging
        verbose: whether to print detailed PnP quality info
        
    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        pnp_quality: dict with quality metrics
    """
    J_3d_full = output["pred_keypoints_3d"]
    J_2d_full = output["pred_keypoints_2d"]  # in image coords matching K
    
    # Use only robust body keypoints (excluding head)
    J_3d = J_3d_full[PNP_ROBUST_INDICES].astype(np.float32)
    J_2d = J_2d_full[PNP_ROBUST_INDICES].astype(np.float32)
    
    # solve pnp
    success, rvec, tvec = cv2.solvePnP(
        objectPoints=J_3d,
        imagePoints=J_2d,
        cameraMatrix=K,
        distCoeffs=dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    
    pnp_quality = {
        "success": success,
        "mean_err": np.inf,
        "max_err": np.inf,
        "per_joint_errors": None,
        "worst_joint": None,
    }
    
    if not success:
        print(f"[Person {person_idx}] PnP FAILED")
        return np.eye(3), np.zeros(3), pnp_quality
    
    R, _ = cv2.Rodrigues(rvec)   # (3,3)
    t = tvec.reshape(3)          # (3,)

    # 2D sanity check on the robust keypoints used
    mean_err, max_err, per_joint_errors = sanity_check_2d(J_3d, J_2d, R, t, K, PNP_JOINT_NAMES)
    
    pnp_quality["mean_err"] = mean_err
    pnp_quality["max_err"] = max_err
    pnp_quality["per_joint_errors"] = per_joint_errors
    
    # Find worst joint
    worst_idx = np.argmax(per_joint_errors)
    pnp_quality["worst_joint"] = (PNP_JOINT_NAMES[worst_idx], per_joint_errors[worst_idx])
    
    if verbose:
        status = "OK" if mean_err < 5.0 else "WARNING" if mean_err < 10.0 else "BAD"
        print(f"[Person {person_idx}] PnP {status}: mean={mean_err:.2f}px, max={max_err:.2f}px, "
              f"worst={PNP_JOINT_NAMES[worst_idx]}({per_joint_errors[worst_idx]:.2f}px)")
        
        if verbose and mean_err > 5.0:
            # Print per-joint breakdown for problematic cases
            print(f"  Per-joint errors:")
            for name, err in zip(PNP_JOINT_NAMES, per_joint_errors):
                marker = " ***" if err > 5.0 else ""
                print(f"    {name}: {err:.2f}px{marker}")
    
    return R, t, pnp_quality

def adjust_K(K, scale_factor):
    K_resized = np.array([[K[0,0]*scale_factor, 0,           K[0,2]*scale_factor],
             [0,           K[1,1]*scale_factor, K[1,2]*scale_factor],
             [0,           0,         1]])
    return K_resized

def solve_pnp_all(pkl_folder, output_folder, intrinsic_folder, verbose=True, 
                   alignment_mode="per_person", ransac_threshold=0.10,
                   extrinsic_folder=None):
    """
    Solve PnP for all people in all frames, align to ground, and transform to world coords.
    
    Args:
        pkl_folder: folder containing per-frame pickle files with SAM3D outputs
        output_folder: folder to save transformed outputs
        intrinsic_folder: folder containing camera intrinsic JSON files
        verbose: whether to print detailed PnP quality info
        alignment_mode: how to align people to ground plane
            - "plane": fit plane to all foot points (original method, sensitive to outliers)
            - "ransac": RANSAC-based plane fitting (more robust to outliers)
            - "per_person": shift each person so their lowest foot is at Y=0 (recommended for noisy data)
            - "extrinsics": use camera extrinsics to project feet onto known ground plane (BEST if you have extrinsics)
        ransac_threshold: inlier threshold for RANSAC mode (meters)
        extrinsic_folder: folder containing camera extrinsic JSON files (required for "extrinsics" mode)
        
    Returns:
        all_quality_metrics: dict mapping img_id -> list of per-person PnP quality dicts
    """
    os.makedirs(output_folder, exist_ok=True)
    all_quality_metrics = {}
    
    print(f"\nAlignment mode: {alignment_mode}")
    if alignment_mode == "ransac":
        print(f"RANSAC threshold: {ransac_threshold}m")
    if alignment_mode == "extrinsics":
        if extrinsic_folder is None:
            raise ValueError("extrinsic_folder is required for alignment_mode='extrinsics'")
        print(f"Using camera extrinsics from: {extrinsic_folder}")

    for pkl_file in sorted(os.listdir(pkl_folder)):
        if not pkl_file.endswith(".pkl"):
            continue
            
        img_id = pkl_file.split(".")[0]
        cam_num = int(img_id.split("_")[0][0])
        intrinsic_file = os.path.join(intrinsic_folder, f"intrinsic_{cam_num}.json")
        
        with open(intrinsic_file, "r") as f:
            intrinsic_data = json.load(f)
            K = np.array(intrinsic_data["intrinsic"])
            dist_coeffs = np.array(intrinsic_data["distortion_coefficients"])
            # Scale K to match 0.5x resolution images fed to SAM3D
            K = adjust_K(K, 0.5)
        
        pkl_file_path = os.path.join(pkl_folder, pkl_file)
        data = pickle.load(open(pkl_file_path, "rb"))
        outputs = data["outputs"]
        data_copy = data.copy()
        
        print(f"\n{'='*60}")
        print(f"Processing: {img_id} ({len(outputs)} people)")
        print(f"{'='*60}")

        # --- 1) PnP for each person, store camera-frame joints/verts ---
        frame_J_cam = []
        frame_V_cam = []
        frame_pnp_quality = []
        
        for idx, person_output in enumerate(outputs):
            R, t, pnp_quality = solve_pnp_person(
                person_output, K, dist_coeffs, 
                person_idx=idx, verbose=verbose
            )
            frame_pnp_quality.append(pnp_quality)

            J_3d = person_output["pred_keypoints_3d"]
            V_3d = person_output["pred_vertices"]

            J_cam = (R @ J_3d.T).T + t
            V_cam = (R @ V_3d.T).T + t

            frame_J_cam.append(J_cam)
            frame_V_cam.append(V_cam)

        all_quality_metrics[img_id] = frame_pnp_quality
        
        # Summary for this frame
        mean_errs = [q["mean_err"] for q in frame_pnp_quality if q["success"]]
        if mean_errs:
            print(f"\n[{img_id}] PnP Summary: avg_mean_err={np.mean(mean_errs):.2f}px, "
                  f"worst_person_mean_err={np.max(mean_errs):.2f}px")

        # --- 2) Collect all foot points for analysis ---
        all_foot_points = []
        foot_person_ids = []
        for person_idx, J_cam in enumerate(frame_J_cam):
            foot_pts = filter_foot_points(J_cam)
            all_foot_points.append(foot_pts)
            foot_person_ids.extend([person_idx] * len(foot_pts))
        all_foot_points = np.concatenate(all_foot_points, axis=0)
        foot_person_ids = np.array(foot_person_ids)

        print(f"\n[{img_id}] Foot points before alignment:")
        print(f"  Total foot points: {len(all_foot_points)}")
        print(f"  Z range (camera frame, depth): {all_foot_points[:,2].min():.4f} to {all_foot_points[:,2].max():.4f}")
        print(f"  Z spread: {all_foot_points[:,2].max() - all_foot_points[:,2].min():.4f}m")

        # --- 3) Apply alignment based on mode ---
        if alignment_mode == "plane":
            # Original method: fit plane to ALL foot points
            n, d, p0 = fit_plane(all_foot_points)
            R_world = make_world_transform(n)
            
            dists = np.abs(all_foot_points @ n + d)
            print(f"\n[{img_id}] Plane fitting (all points):")
            print(f"  Plane normal: [{n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f}]")
            print(f"  RMS distance: {np.sqrt((dists**2).mean()):.4f}m")
            print(f"  Max distance: {dists.max():.4f}m")
            
            # Apply same transform to all
            print(f"\n[{img_id}] After world transform:")
            for idx, (J_cam, V_cam) in enumerate(zip(frame_J_cam, frame_V_cam)):
                V_world = (R_world @ (V_cam - p0).T).T
                J_world = (R_world @ (J_cam - p0).T).T
                
                foot_J_world = J_world[FOOT_INDICES]
                print(f"  Person {idx} foot Y: min={foot_J_world[:,1].min():.4f}, max={foot_J_world[:,1].max():.4f}")
                
                data_copy["outputs"][idx]["pred_vertices"] = V_world.astype(np.float32)
                data_copy["outputs"][idx]["pred_keypoints_3d_world"] = J_world.astype(np.float32)
                data_copy["outputs"][idx]["pred_cam_t"] = np.zeros(3, dtype=np.float32)
                data_copy["outputs"][idx]["focal_length"] = K[0, 0]

        elif alignment_mode == "ransac":
            # RANSAC: more robust to outliers
            n, d, p0, inlier_mask = fit_plane_ransac(all_foot_points, threshold=ransac_threshold)
            R_world = make_world_transform(n)
            
            num_inliers = inlier_mask.sum()
            dists = np.abs(all_foot_points @ n + d)
            print(f"\n[{img_id}] RANSAC plane fitting:")
            print(f"  Inliers: {num_inliers}/{len(all_foot_points)} ({100*num_inliers/len(all_foot_points):.1f}%)")
            print(f"  Plane normal: [{n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f}]")
            print(f"  RMS distance (inliers): {np.sqrt((dists[inlier_mask]**2).mean()):.4f}m")
            print(f"  Max distance (inliers): {dists[inlier_mask].max():.4f}m")
            
            # Show which people are inliers/outliers
            print(f"  Per-person inlier status:")
            for person_idx in range(len(outputs)):
                person_mask = foot_person_ids == person_idx
                person_inliers = inlier_mask[person_mask].sum()
                person_total = person_mask.sum()
                status = "INLIER" if person_inliers == person_total else "OUTLIER" if person_inliers == 0 else "PARTIAL"
                print(f"    Person {person_idx}: {person_inliers}/{person_total} inlier points ({status})")
            
            # Apply same transform to all
            print(f"\n[{img_id}] After world transform:")
            for idx, (J_cam, V_cam) in enumerate(zip(frame_J_cam, frame_V_cam)):
                V_world = (R_world @ (V_cam - p0).T).T
                J_world = (R_world @ (J_cam - p0).T).T
                
                foot_J_world = J_world[FOOT_INDICES]
                print(f"  Person {idx} foot Y: min={foot_J_world[:,1].min():.4f}, max={foot_J_world[:,1].max():.4f}")
                
                data_copy["outputs"][idx]["pred_vertices"] = V_world.astype(np.float32)
                data_copy["outputs"][idx]["pred_keypoints_3d_world"] = J_world.astype(np.float32)
                data_copy["outputs"][idx]["pred_cam_t"] = np.zeros(3, dtype=np.float32)
                data_copy["outputs"][idx]["focal_length"] = K[0, 0]

        elif alignment_mode == "per_person":
            # Per-person: shift each person so their lowest foot is at Y=0
            # First, still compute a common plane orientation (for consistent world up)
            n, d, p0 = fit_plane(all_foot_points)
            R_world = make_world_transform(n)
            
            print(f"\n[{img_id}] Per-person ground contact adjustment:")
            print(f"  Using common plane orientation: [{n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f}]")
            
            for idx, (J_cam, V_cam) in enumerate(zip(frame_J_cam, frame_V_cam)):
                # First apply rotation to get consistent world orientation
                V_world = (R_world @ (V_cam - p0).T).T
                J_world = (R_world @ (J_cam - p0).T).T
                
                # Then shift this person so their lowest foot is at Y=0
                foot_J_world = J_world[FOOT_INDICES]
                min_foot_y = foot_J_world[:, 1].min()
                
                V_world[:, 1] -= min_foot_y
                J_world[:, 1] -= min_foot_y
                
                # Now foot Y should be >= 0
                foot_J_world_adjusted = J_world[FOOT_INDICES]
                print(f"  Person {idx}: shifted by {-min_foot_y:.4f}m, "
                      f"foot Y now: {foot_J_world_adjusted[:,1].min():.4f} to {foot_J_world_adjusted[:,1].max():.4f}")
                
                data_copy["outputs"][idx]["pred_vertices"] = V_world.astype(np.float32)
                data_copy["outputs"][idx]["pred_keypoints_3d_world"] = J_world.astype(np.float32)
                data_copy["outputs"][idx]["pred_cam_t"] = np.zeros(3, dtype=np.float32)
                data_copy["outputs"][idx]["focal_length"] = K[0, 0]
                data_copy["outputs"][idx]["ground_shift"] = float(-min_foot_y)  # store for reference

        elif alignment_mode == "extrinsics":
            # Use camera extrinsics to project feet onto known ground plane
            # This is the BEST method if you have calibrated extrinsics!
            
            # Load extrinsics for this camera
            extrinsic_file = os.path.join(extrinsic_folder, f"extrinsic_{cam_num}.json")
            if not os.path.exists(extrinsic_file):
                print(f"  WARNING: Extrinsic file not found: {extrinsic_file}")
                print(f"  Falling back to per_person mode for this frame")
                # Fall back to per_person for this frame
                n, d, p0 = fit_plane(all_foot_points)
                R_world = make_world_transform(n)
                for idx, (J_cam, V_cam) in enumerate(zip(frame_J_cam, frame_V_cam)):
                    V_world = (R_world @ (V_cam - p0).T).T
                    J_world = (R_world @ (J_cam - p0).T).T
                    foot_J_world = J_world[FOOT_INDICES]
                    min_foot_y = foot_J_world[:, 1].min()
                    V_world[:, 1] -= min_foot_y
                    J_world[:, 1] -= min_foot_y
                    data_copy["outputs"][idx]["pred_vertices"] = V_world.astype(np.float32)
                    data_copy["outputs"][idx]["pred_keypoints_3d_world"] = J_world.astype(np.float32)
                    data_copy["outputs"][idx]["pred_cam_t"] = np.zeros(3, dtype=np.float32)
                    data_copy["outputs"][idx]["focal_length"] = K[0, 0]
            else:
                R_cam, t_cam = load_camera_extrinsics(extrinsic_file)
                
                print(f"\n[{img_id}] Extrinsics-based ground alignment:")
                print(f"  World up direction: {WORLD_UP} ({'Z-up' if WORLD_UP[2] > 0.5 else 'Y-up'})")
                print(f"  Camera position (world): {(-R_cam.T @ t_cam)}")
                
                # Get ground plane in camera coords for reference
                n_cam, d_cam = get_ground_plane_in_camera(R_cam, t_cam, world_up=WORLD_UP)
                print(f"  Ground plane normal (camera): [{n_cam[0]:.4f}, {n_cam[1]:.4f}, {n_cam[2]:.4f}]")
                
                # Determine which axis is "up" in world coords for fallback
                up_axis = np.argmax(np.abs(WORLD_UP))  # 2 for Z-up, 1 for Y-up
                
                successful_alignments = 0
                for idx, person_output in enumerate(outputs):
                    J_3d_body = person_output["pred_keypoints_3d"]
                    V_3d_body = person_output["pred_vertices"]
                    J_2d = person_output["pred_keypoints_2d"]
                    
                    print(f"\n  Person {idx}:")
                    J_world, V_world, align_info = align_person_to_ground_extrinsics(
                        J_3d_body, V_3d_body, J_2d, K, R_cam, t_cam,
                        foot_indices=FOOT_INDICES, world_up=WORLD_UP, verbose=verbose
                    )
                    
                    if J_world is None:
                        print(f"    FAILED: {align_info.get('reason', 'unknown')}")
                        # Fall back: use PnP result with per-person shift
                        J_cam, V_cam = frame_J_cam[idx], frame_V_cam[idx]
                        # Transform to world using extrinsics
                        R_cam_inv = R_cam.T
                        J_world = (R_cam_inv @ (J_cam - t_cam).T).T
                        V_world = (R_cam_inv @ (V_cam - t_cam).T).T
                        # Shift to ground along the up axis
                        min_foot_up = J_world[FOOT_INDICES, up_axis].min()
                        J_world[:, up_axis] -= min_foot_up
                        V_world[:, up_axis] -= min_foot_up
                        print(f"    Fallback: shifted by {-min_foot_up:.4f}m along axis {up_axis}")
                    else:
                        successful_alignments += 1
                        data_copy["outputs"][idx]["alignment_scale"] = float(align_info["scale"])
                        data_copy["outputs"][idx]["alignment_foot_error"] = float(align_info["mean_foot_error"])
                    
                    data_copy["outputs"][idx]["pred_vertices"] = V_world.astype(np.float32)
                    data_copy["outputs"][idx]["pred_keypoints_3d_world"] = J_world.astype(np.float32)
                    data_copy["outputs"][idx]["pred_cam_t"] = np.zeros(3, dtype=np.float32)
                    data_copy["outputs"][idx]["focal_length"] = K[0, 0]
                
                print(f"\n  Summary: {successful_alignments}/{len(outputs)} people aligned successfully")
        
        else:
            raise ValueError(f"Unknown alignment_mode: {alignment_mode}")

        vertices, faces = read_vertices_and_faces(data_copy)
        visualize_multi_person(vertices=vertices, faces=faces)
        
        with open(os.path.join(output_folder, f"{img_id}.pkl"), "wb") as f:
            pickle.dump(data_copy, f)

    return all_quality_metrics

def plot_calibration_image(J_cam, padding=20):
    Jcam_cm = J_cam * 100
    # Build a white canvas that tightly fits all joints with a small padding margin.
    min_xy = np.floor(Jcam_cm[:, :2].min(axis=0)).astype(int)
    max_xy = np.ceil(Jcam_cm[:, :2].max(axis=0)).astype(int)
    width = max(int(max_xy[0] - min_xy[0] + 2 * padding), 1)
    height = max(int(max_xy[1] - min_xy[1] + 2 * padding), 1)

    img_with_joints = np.full((height, width, 3), 255, dtype=np.uint8)
    shift = np.array([padding - min_xy[0], padding - min_xy[1]])

    for i in range(Jcam_cm.shape[0]):
        pt = Jcam_cm[i, :2] + shift
        cv2.circle(img_with_joints, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
    return img_with_joints

def filter_foot_points(J_cam):
    """Extract foot keypoints for ground plane fitting."""
    return J_cam[FOOT_INDICES]


def fit_plane_ransac(points, threshold=0.05, max_iterations=1000):
    """
    RANSAC-based plane fitting - more robust to outliers.
    
    Args:
        points: (M, 3) array of 3D points
        threshold: inlier distance threshold in meters
        max_iterations: max RANSAC iterations
        
    Returns:
        n: (3,) plane normal
        d: plane offset
        p0: (3,) point on the plane (centroid of inliers)
        inlier_mask: (M,) boolean mask of inliers
    """
    best_n, best_d, best_inliers = None, None, None
    best_num_inliers = 0
    
    M = len(points)
    if M < 3:
        # Fall back to regular fit
        n, d, p0 = fit_plane(points)
        return n, d, p0, np.ones(M, dtype=bool)
    
    for _ in range(max_iterations):
        # Sample 3 random points
        idx = np.random.choice(M, 3, replace=False)
        p1, p2, p3 = points[idx]
        
        # Compute plane from 3 points
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < 1e-10:
            continue
        n = n / norm
        d = -np.dot(n, p1)
        
        # Count inliers
        dists = np.abs(points @ n + d)
        inlier_mask = dists < threshold
        num_inliers = inlier_mask.sum()
        
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_n = n
            best_d = d
            best_inliers = inlier_mask
    
    if best_n is None:
        # Fallback
        n, d, p0 = fit_plane(points)
        return n, d, p0, np.ones(M, dtype=bool)
    
    # Refit using only inliers
    inlier_points = points[best_inliers]
    n, d, p0 = fit_plane(inlier_points)
    
    # Recompute inlier mask with final plane
    dists = np.abs(points @ n + d)
    final_inlier_mask = dists < threshold
    
    return n, d, p0, final_inlier_mask


def fit_plane(points):
    """
    points: (M,3)
    returns: normal n (3,), offset d, and a point on the plane p0 (3,)
    """
    # Center the data
    centroid = points.mean(axis=0)
    X = points - centroid

    # SVD on covariance
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    n = vh[-1]                # normal = last singular vector
    n = n / np.linalg.norm(n)

    # plane equation: n^T (X - centroid) = 0 -> n^T X + d = 0
    d = -np.dot(n, centroid)
    return n, d, centroid


def make_world_transform(n):
    """
    n: plane normal in camera coords (unit vector)
    returns: R_world (3,3) mapping camera -> world
             such that R_world @ n ≈ [0, 1, 0]
    """
    up = np.array([0., 1., 0.])  # world up

    # axis to rotate around = n x up
    axis = np.cross(n, up)
    norm_axis = np.linalg.norm(axis)

    if norm_axis < 1e-6:
        # n already aligned with up or down
        if np.dot(n, up) > 0:
            R_world = np.eye(3)
        else:
            # flip 180 degrees
            R_world = np.diag([1, -1, -1])
        return R_world

    axis = axis / norm_axis
    angle = np.arccos(np.clip(np.dot(n, up), -1.0, 1.0))

    # Rodrigues formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R_world = (
        np.eye(3)
        + np.sin(angle) * K
        + (1 - np.cos(angle)) * (K @ K)
    )
    return R_world


def main():
    pkl_folder = "./experiments/inputs/pickles/detections"
    output_pkl_folder = "./experiments/inputs/pickles/calibration"
    intrinsic_folder = "./experiments/inputs/intrinsics"
    extrinsic_folder = "./experiments/inputs/extrinsics"  # <-- Add your extrinsics folder path

    # Alignment mode options:
    # - "plane": fit plane to all foot points (original, sensitive to outliers)
    # - "ransac": RANSAC plane fitting (robust to outliers, but still a common plane)
    # - "per_person": shift each person individually so their lowest foot is at Y=0 (best for noisy data)
    # - "extrinsics": use camera extrinsics to project feet onto known ground (BEST if you have extrinsics!)
    alignment_mode = "extrinsics"  # <-- Change this to try different modes
    
    # Run PnP + ground alignment for all frames
    all_quality_metrics = solve_pnp_all(
        pkl_folder, output_pkl_folder, intrinsic_folder, 
        verbose=True,
        alignment_mode=alignment_mode,
        ransac_threshold=0.10,  # 10cm threshold for RANSAC mode
        extrinsic_folder=extrinsic_folder,  # required for "extrinsics" mode
    )
    
    # Print overall summary
    print("\n" + "="*60)
    print("OVERALL PnP QUALITY SUMMARY")
    print("="*60)
    
    all_mean_errs = []
    problematic_frames = []
    
    for img_id, person_qualities in all_quality_metrics.items():
        frame_mean_errs = [q["mean_err"] for q in person_qualities if q["success"] and q["mean_err"] < np.inf]
        if frame_mean_errs:
            all_mean_errs.extend(frame_mean_errs)
            max_err = max(frame_mean_errs)
            if max_err > 10.0:
                problematic_frames.append((img_id, max_err))
    
    if all_mean_errs:
        print(f"Total persons processed: {len(all_mean_errs)}")
        print(f"Overall mean reprojection error: {np.mean(all_mean_errs):.2f}px")
        print(f"Overall max reprojection error: {np.max(all_mean_errs):.2f}px")
        print(f"Persons with error < 5px: {sum(1 for e in all_mean_errs if e < 5.0)} / {len(all_mean_errs)}")
        print(f"Persons with error 5-10px: {sum(1 for e in all_mean_errs if 5.0 <= e < 10.0)} / {len(all_mean_errs)}")
        print(f"Persons with error > 10px: {sum(1 for e in all_mean_errs if e >= 10.0)} / {len(all_mean_errs)}")
        
        if problematic_frames:
            print(f"\nProblematic frames (max person error > 10px):")
            for img_id, max_err in sorted(problematic_frames, key=lambda x: -x[1]):
                print(f"  {img_id}: {max_err:.2f}px")

if __name__ == "__main__":
    main()
    # img_folder = "experiments/inputs/images_check"
    # pkl_folder = "experiments/inputs/bboxes_kps_refined"
    # output_folder = "experiments/outputs/images_check"
    # plot_bbox_test_image(img_folder, pkl_folder, output_folder)
