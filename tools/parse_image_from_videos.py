import os
import cv2
from tqdm import tqdm

def simplify_vid_name(video_name):
    cam_num = video_name.split("-")[0][-1]
    vid_num = video_name.split("-")[1][-1]
    seg_num = video_name.split("-")[2][-1]
    return f"{cam_num}{vid_num}{seg_num}"

def parse_image_from_videos(video_path, output_path):
    video_name = os.path.basename(video_path).replace(".mp4", "")
    simplified_video_name = simplify_vid_name(video_name)
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_path, f"{simplified_video_name}_{frame_count:06d}.jpg"), frame)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")
    video_capture.release()
    print(f"Processed {frame_count} frames from {video_name}")

def parse_image_from_videos_folder(video_folder, output_folder):
    video_files = sorted(os.listdir(video_folder))
    for video_file in tqdm(video_files) :
        video_path = os.path.join(video_folder, video_file)
        output_path = os.path.join(output_folder, simplify_vid_name(video_file))
        os.makedirs(output_path, exist_ok=True)
        parse_image_from_videos(video_path, output_path)

if __name__ == "__main__":
    video_folder = "./experiments/video_segments"
    output_folder = "./experiments/images_raw"
    parse_image_from_videos_folder(video_folder, output_folder)
