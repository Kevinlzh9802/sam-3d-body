# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import os
from collections import defaultdict

import pyrootutils
import pickle

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

import numpy as np
import torch
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
# from tools.vis_utils import visualize_sample, visualize_sample_together
from tqdm import tqdm
from tools.vis_utils_custom import plot_bboxes_kps, kp_check, bbox_iou
from tools.build_kp_bbox import build_user_bboxes_and_keypoints
import cv2


def main(args):
    if args.output_folder == "":
        output_folder = os.path.join("./output", os.path.basename(args.image_folder))
    else:
        output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)

    # Use command-line args or environment variables
    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")

    # Initialize sam-3d-body model and other optional modules
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(
        args.checkpoint_path, device=device, mhr_path=mhr_path
    )

    human_detector, human_segmentor, fov_estimator = None, None, None
    if args.detector_name:
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(
            name=args.detector_name, device=device, path=detector_path
        )
    if len(segmentor_path):
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=args.segmentor_name, device=device, path=segmentor_path
        )
    if args.fov_name:
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=args.fov_name, device=device, path=fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    image_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".webp",
    ]
    images_list = []
    for root, _, files in os.walk(args.image_folder):
        for file_name in files:
            if os.path.splitext(file_name)[1].lower() in image_extensions:
                image_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(image_path, args.image_folder)
                images_list.append((image_path, rel_path))
    images_list = sorted(images_list, key=lambda x: x[1])
    images_by_folder = defaultdict(list)
    for image_path, rel_path in images_list:
        folder_name = rel_path.split(os.sep)[0]
        images_by_folder[folder_name].append((image_path, rel_path))

    with tqdm(total=len(images_list)) as pbar:
        for folder_name in sorted(images_by_folder.keys()):

            folder_code = int(folder_name)
            # folder code: 228, 429, ...
            # filter out folders not in the interval
            if folder_code < args.segs_interval[0] or folder_code > args.segs_interval[1]:
                continue
            print(f"Processing folder: {folder_name}")

            # sort images by index
            folder_images = sorted(images_by_folder[folder_name], key=lambda x: x[1])

            # load bboxes and kps
            bboxes_kps_data = None
            if len(args.bbox_kp_folder):
                keypoint_path = os.path.join(args.bbox_kp_folder, f"{folder_name}.pkl")
                with open(keypoint_path, "rb") as kp_f:
                    bboxes_kps_data = pickle.load(kp_f)

            # process images
            for _, (image_path, rel_path) in enumerate(folder_images):
                rel_path_no_ext = os.path.splitext(rel_path)[0]
                idx = int(rel_path_no_ext.split("_")[1])
                try:
                    bboxes = bboxes_kps_data[idx]["bboxes"]
                    kps = bboxes_kps_data[idx]["kps"]
                    # kps = kp_check(rel_path_no_ext, kps)
                except:
                    print(f"Warning: No bboxes and kps found for {rel_path_no_ext}")
                    bboxes, kps = None, None

                # outputs = estimator.process_one_image(
                #     image_path,
                #     bboxes=bboxes,
                #     bbox_thr=args.bbox_thresh,
                #     use_mask=args.use_mask,
                #     inference_type="body",              # since we now manually prompt
                #     keypoint_prompt=kps,    # <--- NEW
                # )

                outputs = run_mixed_sam3d_inference(
                    estimator,
                    image_path,
                    user_bboxes=bboxes,
                    user_kps=kps,
                    bbox_thr=0.5,
                    nms_thr=0.3,
                    use_mask=False,
                    inference_type="body",      # then your process_one_image will call run_keypoint_prompt
                    iou_merge_thr=0.5,
                )
                pkl_path = os.path.join(output_folder, f"{rel_path_no_ext}.pkl")
                pkl_dir = os.path.dirname(pkl_path)
                if pkl_dir:
                    os.makedirs(pkl_dir, exist_ok=True)
                with open(pkl_path, "wb") as f:
                    pickle.dump(
                        {
                            "image_path": image_path,
                            "image_name": os.path.basename(image_path),
                            "outputs": outputs,
                            "faces": estimator.faces,
                        },
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                pbar.update(1)
        # print(type(outputs))
        # print(type(estimator.faces))

        # print(outputs)
        # print(estimator.faces)
        # img = cv2.imread(image_path)
        # rend_img = visualize_sample_together(img, outputs, estimator.faces)
        # cv2.imwrite(
        #     f"{output_folder}/{os.path.basename(image_path)[:-4]}.jpg",
        #     rend_img.astype(np.uint8),
        # )

def run_mixed_sam3d_inference(
    estimator,
    img,
    user_bboxes=None,
    user_kps=None,
    # frame_coords=None,
    bbox_thr=0.5,
    nms_thr=0.3,
    use_mask=False,
    inference_type="body",
    iou_merge_thr=0.5,
):
    """
    Mixed mode:
    - uses estimator.detector to find all people
    - injects / overrides some with your bboxes+keypoints
    - calls estimator.process_one_image once with merged inputs

    Parameters
    ----------
    estimator : SAM3DBodyEstimator
    img : np.ndarray or str
        Image array (H,W,3) or path. If path, it must be readable with cv2.
    frame_coords : dict or None
        {person_id: (10,2) normalized coords in [0,1]} for the subset you want to prompt.
    iou_merge_thr : float
        If IoU(user_box, det_box) >= this, they are treated as the *same* person and the
        detector box is replaced by your box & keypoints. Otherwise, the user_box is added
        as an extra person.

    Returns
    -------
    outputs : whatever estimator.process_one_image returns
    """

    # --- get image size ---
    import cv2

    if isinstance(img, str):
        im = cv2.imread(img)
        if im is None:
            raise RuntimeError(f"Failed to read image from path: {img}")
        img_h, img_w = im.shape[:2]
    else:
        im = img
        img_h, img_w = im.shape[:2]

    # --- build user bboxes + keypoints (may be None) ---
    # user_bboxes, user_kps = build_user_bboxes_and_keypoints(
    #     frame_coords,
    #     img_width=img_w,
    #     img_height=img_h,
    #     pad_ratio=pad_ratio,
    #     min_valid_kps=min_valid_kps,
    # )

    # --- get detector boxes (if available) ---
    if estimator.detector is not None:
        det_boxes = estimator.detector.run_human_detection(
            im,
            det_cat_id=0,
            bbox_thr=bbox_thr,
            nms_thr=nms_thr,
            default_to_full_image=False,
        )
        if det_boxes is None:
            det_boxes = np.zeros((0, 4), dtype=np.float32)
        else:
            det_boxes = det_boxes.reshape(-1, 4).astype(np.float32)
    else:
        det_boxes = np.zeros((0, 4), dtype=np.float32)

    # --- if no user coords and no detector, just fall back to original API ---
    if (user_bboxes is None or len(user_bboxes) == 0) and len(det_boxes) == 0:
        # no hints and no detector -> single box over image (estimator's default)
        return estimator.process_one_image(
            img,
            bboxes=None,
            bbox_thr=bbox_thr,
            nms_thr=nms_thr,
            use_mask=use_mask,
            inference_type=inference_type,
            keypoint_prompt=None,
        )

    # --- start with detector boxes; no prompts for them yet ---
    combined_boxes = []
    combined_kps = []

    # For detector-only persons: fill invalid labels (-2) so they are ignored by prompt encoder
    if len(det_boxes) > 0:
        dummy_kps = np.zeros((10, 3), dtype=np.float32)
        dummy_kps[:, 2] = -2.0  # all invalid
        for b in det_boxes:
            combined_boxes.append(b)
            combined_kps.append(dummy_kps.copy())

    # --- merge / add user boxes ---
    if user_bboxes is not None and len(user_bboxes) > 0:
        for i in range(len(user_bboxes)):
            u_box = user_bboxes[i]
            u_kps = user_kps[i]

            if len(det_boxes) == 0:
                # no detector boxes; just append all user boxes
                combined_boxes.append(u_box)
                combined_kps.append(u_kps)
                continue

            # compute IoU with all detector boxes (in the current combined list)
            ious = np.array([bbox_iou(u_box, db) for db in det_boxes], dtype=np.float32)
            max_idx = int(ious.argmax()) if len(ious) > 0 else -1
            max_iou = float(ious[max_idx]) if len(ious) > 0 else 0.0

            if max_iou >= iou_merge_thr:
                # treat as the same person; override that slot with user box + keypoints
                combined_boxes[max_idx] = u_box
                combined_kps[max_idx] = u_kps
            else:
                # new person; append
                combined_boxes.append(u_box)
                combined_kps.append(u_kps)

    combined_boxes = np.stack(combined_boxes, axis=0).astype(np.float32)  # (N,4)
    combined_kps = np.stack(combined_kps, axis=0).astype(np.float32)      # (N,10,3)

    # --- finally call the estimator with merged inputs ---
    outputs = estimator.process_one_image(
        img,
        bboxes=combined_boxes,
        bbox_thr=bbox_thr,    # detector threshold still used only if bboxes=None
        nms_thr=nms_thr,
        use_mask=use_mask,
        inference_type=inference_type,
        keypoint_prompt=combined_kps,
    )
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM 3D Body Demo - Single Image Human Mesh Recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                python demo.py --image_folder ./images --checkpoint_path ./checkpoints/model.ckpt

                Environment Variables:
                SAM3D_MHR_PATH: Path to MHR asset
                SAM3D_DETECTOR_PATH: Path to human detection model folder
                SAM3D_SEGMENTOR_PATH: Path to human segmentation model folder
                SAM3D_FOV_PATH: Path to fov estimation model folder
                """,
    )
    parser.add_argument(
        "--image_folder",
        required=True,
        type=str,
        help="Path to folder containing input images",
    )
    parser.add_argument(
        "--output_folder",
        default="",
        type=str,
        help="Path to output folder (default: ./output/<image_folder_name>)",
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to SAM 3D Body model checkpoint",
    )
    parser.add_argument(
        "--detector_name",
        default="vitdet",
        type=str,
        help="Human detection model for demo (Default `vitdet`, add your favorite detector if needed).",
    )
    parser.add_argument(
        "--segmentor_name",
        default="sam2",
        type=str,
        help="Human segmentation model for demo (Default `sam2`, add your favorite segmentor if needed).",
    )
    parser.add_argument(
        "--fov_name",
        default="moge2",
        type=str,
        help="FOV estimation model for demo (Default `moge2`, add your favorite fov estimator if needed).",
    )
    parser.add_argument(
        "--detector_path",
        default="",
        type=str,
        help="Path to human detection model folder (or set SAM3D_DETECTOR_PATH)",
    )
    parser.add_argument(
        "--segmentor_path",
        default="",
        type=str,
        help="Path to human segmentation model folder (or set SAM3D_SEGMENTOR_PATH)",
    )
    parser.add_argument(
        "--fov_path",
        default="",
        type=str,
        help="Path to fov estimation model folder (or set SAM3D_FOV_PATH)",
    )
    parser.add_argument(
        "--mhr_path",
        default="",
        type=str,
        help="Path to MoHR/assets folder (or set SAM3D_mhr_path)",
    )
    parser.add_argument(
        "--bbox_thresh",
        default=0.8,
        type=float,
        help="Bounding box detection threshold",
    )
    parser.add_argument(
        "--use_mask",
        action="store_true",
        default=False,
        help="Use mask-conditioned prediction (segmentation mask is automatically generated from bbox)",
    )
    parser.add_argument(
        "--bbox_kp_folder",
        default="",
        type=str,
        help="Optional folder containing per-image bboxes and keypoints as pickle (image.jpg -> image.pkl).",
    )
    parser.add_argument(
        "--segs_interval",
        default=[400, 500],
        type=list,
        help="Optional list of segments to process, default is [400, 500].",
    )
    args = parser.parse_args()

    main(args)
