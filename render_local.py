import argparse
import os
import pickle
import cv2
import numpy as np

from tools.vis_utils import visualize_sample_together  # uses Renderer internally
from tools.vis_utils import visualize_world_floor, visualize_world_floor_interactive

def load_image(entry, images_dir=None):
    candidates = []
    if images_dir:
        candidates.append(os.path.join(images_dir, entry["image_name"]))
    candidates.append(entry.get("image_path") or "")
    for p in candidates:
        if p and os.path.exists(p):
            img = cv2.imread(p)
            if img is not None:
                return img
    raise FileNotFoundError(
        f"Could not find image for {entry.get('image_name')} in {candidates}"
    )

def main(args):
    os.makedirs(args.output_folder, exist_ok=True)
    pkl_files = sorted(
        f for f in os.listdir(args.pickle_folder) if f.lower().endswith(".pkl")
    )
    for fname in pkl_files:
        pkl_path = os.path.join(args.pickle_folder, fname)
        with open(pkl_path, "rb") as f:
            entry = pickle.load(f)
        img = load_image(entry, args.images_dir)
        outputs = entry["outputs"]
        faces = entry["faces"]
        # rend_img = visualize_sample_together(img, outputs, faces)
        rend_img = visualize_world_floor(outputs, faces, render_res=(800, 800))
        # rend_img = visualize_world_floor_interactive(outputs, faces)
        out_name = os.path.splitext(fname)[0] + "_render.jpg"
        cv2.imwrite(os.path.join(args.output_folder, out_name), rend_img.astype(np.uint8))
        print(f"Rendered {fname} -> {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render SAM-3D-Body pickles locally"
    )
    parser.add_argument("--pickle_folder", required=True, help="Folder with .pkl files")
    parser.add_argument(
        "--images_dir",
        required=False,
        default=None,
        help="Folder containing the original images (use if image_path inside pkl is not valid locally)",
    )
    parser.add_argument(
        "--output_folder",
        required=True,
        help="Where to save rendered images",
    )
    args = parser.parse_args()
    main(args)