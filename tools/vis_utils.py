# Copyright (c) Meta Platforms, Inc. and affiliates.
import numpy as np
import cv2
from sam_3d_body.visualization.renderer import Renderer
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
import pyrender
import trimesh

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

visualizer = SkeletonVisualizer(line_width=2, radius=5)
visualizer.set_pose_meta(mhr70_pose_info)


def visualize_sample(img_cv2, outputs, faces):
    img_keypoints = img_cv2.copy()
    img_mesh = img_cv2.copy()

    rend_img = []
    for pid, person_output in enumerate(outputs):
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img1 = visualizer.draw_skeleton(img_keypoints.copy(), keypoints_2d)

        img1 = cv2.rectangle(
            img1,
            (int(person_output["bbox"][0]), int(person_output["bbox"][1])),
            (int(person_output["bbox"][2]), int(person_output["bbox"][3])),
            (0, 255, 0),
            2,
        )

        if "lhand_bbox" in person_output:
            img1 = cv2.rectangle(
                img1,
                (
                    int(person_output["lhand_bbox"][0]),
                    int(person_output["lhand_bbox"][1]),
                ),
                (
                    int(person_output["lhand_bbox"][2]),
                    int(person_output["lhand_bbox"][3]),
                ),
                (255, 0, 0),
                2,
            )

        if "rhand_bbox" in person_output:
            img1 = cv2.rectangle(
                img1,
                (
                    int(person_output["rhand_bbox"][0]),
                    int(person_output["rhand_bbox"][1]),
                ),
                (
                    int(person_output["rhand_bbox"][2]),
                    int(person_output["rhand_bbox"][3]),
                ),
                (0, 0, 255),
                2,
            )

        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)
        img2 = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                img_mesh.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        )

        white_img = np.ones_like(img_cv2) * 255
        img3 = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                white_img,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                side_view=True,
            )
            * 255
        )

        cur_img = np.concatenate([img_cv2, img1, img2, img3], axis=1)
        rend_img.append(cur_img)

    return rend_img

def visualize_sample_together(img_cv2, outputs, faces):
    # Render everything together
    img_keypoints = img_cv2.copy()
    img_mesh = img_cv2.copy()

    # First, sort by depth, furthest to closest
    all_depths = np.stack([tmp['pred_cam_t'] for tmp in outputs], axis=0)[:, 2]
    outputs_sorted = [outputs[idx] for idx in np.argsort(-all_depths)]

    # Then, draw all keypoints.
    for pid, person_output in enumerate(outputs_sorted):
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img_keypoints = visualizer.draw_skeleton(img_keypoints, keypoints_2d)

    # Then, put all meshes together as one super mesh
    all_pred_vertices = []
    all_faces = []
    for pid, person_output in enumerate(outputs_sorted):
        all_pred_vertices.append(person_output["pred_vertices"] + person_output["pred_cam_t"])
        all_faces.append(faces + len(person_output["pred_vertices"]) * pid)
    all_pred_vertices = np.concatenate(all_pred_vertices, axis=0)
    all_faces = np.concatenate(all_faces, axis=0)

    # Pull out a fake translation; take the closest two
    fake_pred_cam_t = (np.max(all_pred_vertices[-2*18439:], axis=0) + np.min(all_pred_vertices[-2*18439:], axis=0)) / 2
    all_pred_vertices = all_pred_vertices - fake_pred_cam_t
    
    # Render front view
    renderer = Renderer(focal_length=person_output["focal_length"], faces=all_faces)
    img_mesh = (
        renderer(
            all_pred_vertices,
            fake_pred_cam_t,
            img_mesh,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
        )
        * 255
    )

    # Render side view
    white_img = np.ones_like(img_cv2) * 255
    img_mesh_side = (
        renderer(
            all_pred_vertices,
            fake_pred_cam_t,
            white_img,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
            side_view=True,
        )
        * 255
    )

    cur_img = np.concatenate([img_cv2, img_keypoints, img_mesh, img_mesh_side], axis=1)

    return cur_img


def visualize_world_floor(outputs, faces, render_res=(800, 800)):
    """
    Render all people using pred_vertices as already world-aligned coordinates.
    Ignores per-person pred_cam_t; uses a fixed virtual camera.
    """
    # Gather vertices from all people
    V_list = [o["pred_vertices"] for o in outputs]

    # Use any focal length; we stored K[0,0] into each output
    focal_length = float(outputs[0]["focal_length"])
    renderer = Renderer(focal_length=focal_length, faces=faces)

    # We treat V_list as already in camera/world coords -> use zero translations
    cam_t_list = [np.zeros(3, dtype=np.float32) for _ in V_list]

    rgba = renderer.render_rgba_multiple(
        V_list[5:8],
        cam_t_list[5:8],
        rot_axis=[1, 0, 0],
        rot_angle=0,
        scene_bg_color=(1, 1, 1),
        render_res=list(render_res),
        focal_length=focal_length,
    )
    # rgba: H x W x 4 in [0,1]
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    return rgb


def visualize_world_floor_interactive(outputs, faces, floor_height=0.0, floor_size=5.0):
    """
    Launch an interactive 3D viewer with all people placed using pred_vertices
    in a shared (world) frame. Keeps visualize_sample_together untouched.
    """
    verts = []
    all_faces = []
    offset = 0
    for o in outputs:
        v = o["pred_vertices"]
        verts.append(v)
        all_faces.append(faces + offset)
        offset += v.shape[0]
    verts = np.concatenate(verts, axis=0)
    all_faces = np.concatenate(all_faces, axis=0)

    # People mesh
    people_mesh = trimesh.Trimesh(verts, all_faces, process=False)
    people_node = pyrender.Mesh.from_trimesh(people_mesh, smooth=True)

    # Simple floor plane for reference
    plane = trimesh.creation.box(extents=(floor_size, 0.01, floor_size))
    plane.apply_translation([0, floor_height - 0.005, 0])
    plane_node = pyrender.Mesh.from_trimesh(plane, smooth=False)

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=(0.3, 0.3, 0.3))
    scene.add(people_node)
    scene.add(plane_node)

    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(60.0))
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = np.array([0, 1.5, 4.0])
    scene.add(camera, pose=cam_pose)

    # Compatibility for NumPy 2.0 where np.infty was removed.
    if not hasattr(np, "infty"):
        np.infty = np.inf

    try:
        pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)
    except Exception as e:
        print(
            f"pyrender.Viewer failed (likely due to missing onscreen GL): {e}. "
            "Falling back to offscreen render."
        )
        r = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=800)
        color, _ = r.render(scene)
        r.delete()
        return color
