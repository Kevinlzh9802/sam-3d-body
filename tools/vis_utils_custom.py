import cv2
import numpy as np
import pickle
import torch
import os

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


def main():
    # check pickle file
    pickle_file = "experiments/bboxex_kps/428.pkl"
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    print(data)

if __name__ == "__main__":
    # main()
    img_folder = "experiments/inputs/images_check"
    pkl_folder = "experiments/inputs/bboxes_kps_refined"
    output_folder = "experiments/outputs/images_check"
    plot_bbox_test_image(img_folder, pkl_folder, output_folder)