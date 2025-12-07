import cv2
import numpy as np

def kp_check(filename, kps):
    assert isinstance(kps, np.ndarray), "Invalid keypoints type"
    assert kps.shape[1] == 10, "Invalid keypoints shape"

    if np.any(kps < 0):
        print(f"Warning: Keypoints < 0 found in {filename}")
    if np.any(kps > 1):
        print(f"Warning: Keypoints > 1 found in {filename}")
    kps[kps < 0] = 0
    kps[kps > 1] = 1
    return kps

def plot_bboxes_kps(img, bboxes, kps):
    img_with_bboxes_kps = img.copy()
    if bboxes is not None:
        for bbox in bboxes:
            cv2.rectangle(img_with_bboxes_kps, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
    if kps is not None:
        for kp in kps:
            cv2.circle(img_with_bboxes_kps, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)
    return img_with_bboxes_kps