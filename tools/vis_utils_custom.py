import cv2
import numpy as np
import pickle
import torch
import os
import json

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

def sanity_check_2d(J_3d, J_2d, R, t, K):
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
    errors = np.linalg.norm(diff, axis=-1)   # pixel error per joint
    mean_err = errors.mean()
    max_err = errors.max()

    # print("Reprojection sanity check:")
    # print(f"  Mean error: {mean_err:.3f} px")
    # print(f"  Max error:  {max_err:.3f} px")
    return mean_err, max_err

def solve_pnp_person(output, K, dist_coeffs):
    """
    kps: (N, 10, 3)
    returns (N, 3)
    """
    J_3d = output["pred_keypoints_3d"]
    J_2d = output["pred_keypoints_2d"]  # in original image coords  
    # idxs = [0, 5, 6, 9, 10, 13, 14]  # example: nose, shoulders, hips, ankles
    obj_points = J_3d.astype(np.float32)  # (N,3)
    img_points = J_2d.astype(np.float32)  # (N,2)
    
    # solve pnp
    success, rvec, tvec = cv2.solvePnP(
        objectPoints=obj_points,
        imagePoints=img_points,
        cameraMatrix=K,
        distCoeffs=dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        print(f"Failed to solve pnp for person")
    
    R, _ = cv2.Rodrigues(rvec)   # (3,3)
    t = tvec.reshape(3)          # (3,)

    # 2d sanity check
    # mean_err, max_err = sanity_check_2d(J_3d, J_2d, R, t, K)
    # if mean_err > 1.0 or max_err > 2.0:
    #     print(f"2d sanity check failed")
    return R, t

def adjust_K(K, scale_factor):
    K_resized = np.array([[K[0,0]*scale_factor, 0,           K[0,2]*scale_factor],
             [0,           K[1,1]*scale_factor, K[1,2]*scale_factor],
             [0,           0,         1]])
    return K_resized

def solve_pnp_all(pkl_folder, output_folder, intrinsic_folder):
    """
    pkl_folder: str
    output_folder: str
    intrinsic_folder: str
    returns None
    """
    frame_coords_all = []
    for pkl_file in os.listdir(pkl_folder):
        img_id = pkl_file.split(".")[0]
        cam_num = int(img_id.split("_")[0][0])
        intrinsic_file = os.path.join(intrinsic_folder, f"intrinsic_{cam_num}.json")
        with open(intrinsic_file, "r") as f:
            data = json.load(f)
            K = np.array(data["intrinsic"])
            dist_coeffs = np.array(data["distortion_coefficients"])
            K = adjust_K(K, 0.5)
        
        pkl_file_path = os.path.join(pkl_folder, pkl_file)
        with open(pkl_file_path, "rb") as f:
            data = pickle.load(f)
        outputs = data["outputs"]
        data_copy = data.copy()

        # --- 1) PnP for each person, store camera-frame joints/verts ---
        frame_J_cam = []
        frame_V_cam = []
        for idx, person_output in enumerate(outputs):
            R, t = solve_pnp_person(person_output, K, dist_coeffs)

            J_3d = person_output["pred_keypoints_3d"]
            V_3d = person_output["pred_vertices"]

            J_cam = (R @ J_3d.T).T + t
            V_cam = (R @ V_3d.T).T + t

            frame_J_cam.append(J_cam)
            frame_V_cam.append(V_cam)

        # --- 2) Fit ONE ground plane using all foot joints in this frame ---
        all_foot_points = []
        for J_cam in frame_J_cam:
            all_foot_points.append(filter_foot_points(J_cam))  # uses MHR70 indices
        all_foot_points = np.concatenate(all_foot_points, axis=0)

        print(img_id, "world foot Y min/max:",
              all_foot_points[:,1].min(),
              all_foot_points[:,1].max())

        n, d, p0 = fit_plane(all_foot_points)
        R_world = make_world_transform(n)  # camera -> world

        # (optional) print RMS to check:
        dists = np.abs(all_foot_points @ n + d)
        print(img_id, "RMS distance to plane:", np.sqrt((dists**2).mean()))

        # --- 3) Apply this SAME world transform to all people ---
        frame_coords_2d = []
        for idx, (J_cam, V_cam) in enumerate(zip(frame_J_cam, frame_V_cam)):
            V_world = (R_world @ (V_cam - p0).T).T  # (V,3)
            J_world = (R_world @ (J_cam - p0).T).T  # (J,3)

            data_copy["outputs"][idx]["pred_vertices"] = V_world.astype(np.float32)
            # renderer will treat these as "in a shared world/camera frame"
            data_copy["outputs"][idx]["pred_cam_t"] = np.zeros(3, dtype=np.float32)
            data_copy["outputs"][idx]["focal_length"] = K[0, 0]

            frame_coords_2d.append(J_world)

        with open(os.path.join(output_folder, f"{img_id}.pkl"), "wb") as f:
            pickle.dump(data_copy, f)
            # cv2.imwrite(os.path.join(output_folder, pkl_file.replace(".pkl", f"_pnp.jpg")), img)
        frame_coords_all.append({img_id: np.concatenate(frame_coords_2d, axis=0)})
    return frame_coords_all

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
    foot_indices = [13, 14, 15, 18]  # adjust as you like
    return J_cam[foot_indices]

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
    # check pickle file
    # pickle_file = "experiments/bboxex_kps/428.pkl"
    # with open(pickle_file, "rb") as f:
    #     data = pickle.load(f)
    # print(data)
    pkl_folder = "experiments/inputs/pickles/detections"
    output_pkl_folder = "experiments/inputs/pickles/calibration"
    output_img_folder = "experiments/outputs/intrinsic_check_img"
    intrinsic_folder = "experiments/inputs/intrinsics"

    frame_coords_all = solve_pnp_all(pkl_folder, output_pkl_folder, intrinsic_folder)
    for img_info in frame_coords_all:
        img_id = list(img_info.keys())[0]
        J_cam = img_info[img_id]

        img_with_joints = plot_calibration_image(J_cam)
        cv2.imwrite(os.path.join(output_img_folder, f"{img_id}_pnp.jpg"), img_with_joints)

if __name__ == "__main__":
    main()
    # img_folder = "experiments/inputs/images_check"
    # pkl_folder = "experiments/inputs/bboxes_kps_refined"
    # output_folder = "experiments/outputs/images_check"
    # plot_bbox_test_image(img_folder, pkl_folder, output_folder)
