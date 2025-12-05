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
            folder_images = sorted(images_by_folder[folder_name], key=lambda x: x[1])
            bboxes_kps_data = None
            if len(args.bbox_kp_folder):
                keypoint_path = os.path.join(args.keypoint_json_folder, f"{folder_name}.pkl")
                with open(keypoint_path, "rb") as kp_f:
                    bboxes_kps_data = pickle.load(kp_f)
            for idx, (image_path, rel_path) in enumerate(folder_images):
                #TODO: idx may not be the same as image order. Deal with this.
                if bboxes_kps_data is not None:
                    bboxes = bboxes_kps_data[idx]["bboxes"]
                    kps = bboxes_kps_data[idx]["kps"]
                else:
                    bboxes, kps = None, None

                outputs = estimator.process_one_image(
                    image_path,
                    bboxes=bboxes,
                    bbox_thr=args.bbox_thresh,
                    use_mask=args.use_mask,
                    inference_type="body",              # since we now manually prompt
                    keypoint_prompt=kps,    # <--- NEW
                )
                pkl_path = os.path.join(output_folder, f"{rel_path}.pkl")
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
    args = parser.parse_args()

    main(args)
