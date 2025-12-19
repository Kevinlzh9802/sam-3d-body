import numpy as np
from aitviewer.renderables.meshes import Meshes
from aitviewer.viewer import Viewer
import sys
import types
import os
import pickle
import json

# Compatibility: allow loading pickles that reference numpy._core (NumPy 2.x)
if "numpy._core" not in sys.modules:
    np_core = types.ModuleType("numpy._core")
    np_core.numeric = np.core.numeric
    sys.modules["numpy._core"] = np_core
    sys.modules["numpy._core.numeric"] = np.core.numeric
    if hasattr(np.core, "multiarray"):
        sys.modules["numpy._core.multiarray"] = np.core.multiarray

# Import alignment functions from vis_utils_custom
from vis_utils_custom import (
    solve_pnp_person,
    align_person_to_ground_extrinsics,
    adjust_K,
    load_camera_extrinsics,
    FOOT_INDICES,
    WORLD_UP,
)

def visualize_mesh_sequence(vertices_list, faces_list, colors=None):
    """
    Visualize mesh sequences for multiple people in aitviewer.
    
    Args:
        vertices_list: List of vertex arrays, one per person
                      Each array shape: (N_frames, N_verts, 3)
        faces_list: List of face arrays, one per person
                   Each array shape: (N_faces, 3)
        colors: Optional list of RGB colors for each person, each color as (R, G, B) 
                with values in [0, 1]. If None, uses default colors.
    """
    # Create viewer
    v = Viewer()
    
    # Default colors if not provided
    if colors is None:
        colors = [
            (0.8, 0.2, 0.2),  # Red
            (0.2, 0.8, 0.2),  # Green
            (0.2, 0.2, 0.8),  # Blue
            (0.8, 0.8, 0.2),  # Yellow
            (0.8, 0.2, 0.8),  # Magenta
            (0.2, 0.8, 0.8),  # Cyan
        ]
    
    # Add each person to the scene
    for i, (verts, faces) in enumerate(zip(vertices_list, faces_list)):
        # Ensure vertices are numpy arrays with correct dtype
        verts = np.asarray(verts, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)
        
        # Ensure vertices have correct shape (N_frames, N_verts, 3)
        if verts.ndim == 2:
            # Static mesh: (N_verts, 3) -> (1, N_verts, 3)
            verts = verts[np.newaxis, ...]
        elif verts.ndim != 3:
            raise ValueError(f"Vertices must be 2D or 3D, got shape {verts.shape}")
        
        # Ensure faces are 2D
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"Faces must be shape (N_faces, 3), got {faces.shape}")
        
        n_frames, n_verts, _ = verts.shape
        n_faces = faces.shape[0]
        
        print(f"Person {i+1}: {n_frames} frames, {n_verts} vertices, {n_faces} faces")
        print(f"  Vertices shape: {verts.shape}")
        print(f"  Faces shape: {faces.shape}")
        print(f"  Vertex range: [{verts.min():.3f}, {verts.max():.3f}]")
        print(f"  Face indices range: [{faces.min()}, {faces.max()}]")
        
        # Verify face indices are valid
        if faces.max() >= n_verts:
            raise ValueError(
                f"Face indices out of bounds for Person {i+1}: "
                f"max index {faces.max()} >= num vertices {n_verts}"
            )
        
        # Get color for this person
        color = np.array(colors[i % len(colors)], dtype=np.float32)
        
        try:
            # Create mesh renderable
            mesh = Meshes(
                vertices=verts,
                faces=faces,
                name=f"Person_{i+1}"
            )
            
            # Add to viewer
            v.scene.add(mesh)
            print(f"  ✓ Successfully added Person {i+1}\n")
            
        except Exception as e:
            print(f"  ✗ Error creating mesh for Person {i+1}: {e}\n")
            raise
    
    print(f"Total: {len(vertices_list)} people loaded")
    print("Press SPACE to play/pause the animation")
    print("Use arrow keys to step through frames")
    
    # Set floor plane to XY (for Z-up world coordinate system)
    v.scene.floor.plane = "xy"
    v.scene.floor.side_length = 20  # 20 meters
    
    # Run viewer
    v.run()


def load_sequence_from_pickles(pkl_folder, segment_id=None):
    """
    Load mesh sequences from pickle files in a folder (no alignment).
    
    Args:
        pkl_folder: Path to folder containing pickle files
        segment_id: Optional segment ID to filter (e.g., "428"). If None, loads all.
        
    Returns:
        vertices_list: List of vertex arrays per person, each (N_frames, N_verts, 3)
        faces: Faces array (N_faces, 3) - same for all people/frames
        frame_ids: List of frame identifiers
    """
    # Get all pickle files
    pkl_files = sorted([f for f in os.listdir(pkl_folder) if f.endswith(".pkl")])
    
    # Filter by segment if specified
    if segment_id is not None:
        pkl_files = [f for f in pkl_files if f.startswith(str(segment_id))]
    
    if not pkl_files:
        raise ValueError(f"No pickle files found in {pkl_folder}" + 
                        (f" with segment_id={segment_id}" if segment_id else ""))
    
    print(f"Found {len(pkl_files)} pickle files")
    
    # Load first file to get faces and number of people
    first_file = os.path.join(pkl_folder, pkl_files[0])
    with open(first_file, "rb") as f:
        first_data = pickle.load(f)
    
    faces = first_data["faces"]
    n_people_first = len(first_data["outputs"])
    print(f"First frame has {n_people_first} people, {faces.shape[0]} faces")
    
    # Collect vertices per person across all frames
    # Note: This assumes consistent person ordering across frames
    # For proper tracking, you'd need person IDs or re-identification
    person_vertices = {i: [] for i in range(n_people_first)}
    frame_ids = []
    
    for pkl_file in pkl_files:
        pkl_path = os.path.join(pkl_folder, pkl_file)
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        
        outputs = data["outputs"]
        frame_id = pkl_file.replace(".pkl", "")
        frame_ids.append(frame_id)
        
        # Get vertices for each person in this frame
        for person_idx, person_output in enumerate(outputs):
            if person_idx < n_people_first:
                verts = person_output["pred_vertices"]
                person_vertices[person_idx].append(verts)
        
        # If this frame has more people than the first, skip extras
        # If fewer, we'll have missing frames for some people
    
    # Convert to arrays: (N_frames, N_verts, 3)
    vertices_list = []
    for person_idx in range(n_people_first):
        verts_sequence = person_vertices[person_idx]
        if len(verts_sequence) > 0:
            verts_array = np.stack(verts_sequence, axis=0)
            vertices_list.append(verts_array)
            print(f"Person {person_idx}: {verts_array.shape[0]} frames, {verts_array.shape[1]} vertices")
    
    return vertices_list, faces, frame_ids


def load_sequence_with_extrinsics_alignment(
    pkl_folder,
    intrinsic_folder,
    extrinsic_folder,
    segment_id=None,
    image_scale=2.0,
    extrinsics_scale=0.01,
    verbose=False
):
    """
    Load mesh sequences from pickle files with extrinsics-based ground alignment.
    
    This uses PnP to estimate body pose and camera extrinsics to project feet
    onto the known ground plane, aligning all people to a common world frame.
    
    Args:
        pkl_folder: Path to folder containing pickle files from SAM3D
        intrinsic_folder: Path to folder containing camera intrinsic JSON files
        extrinsic_folder: Path to folder containing camera extrinsic JSON files
        segment_id: Optional segment ID to filter (e.g., "428")
        image_scale: Scale factor from SAM3D resolution to extrinsics resolution
        extrinsics_scale: Scale factor for extrinsics translation (e.g., 0.01 for cm->m)
        verbose: Print detailed alignment info
        
    Returns:
        vertices_list: List of aligned vertex arrays per person, each (N_frames, N_verts, 3)
        faces: Faces array (N_faces, 3) - same for all people/frames
        frame_ids: List of frame identifiers
    """
    # Get all pickle files
    pkl_files = sorted([f for f in os.listdir(pkl_folder) if f.endswith(".pkl")])
    
    # Filter by segment if specified
    if segment_id is not None:
        pkl_files = [f for f in pkl_files if f.startswith(str(segment_id))]
    
    if not pkl_files:
        raise ValueError(f"No pickle files found in {pkl_folder}" + 
                        (f" with segment_id={segment_id}" if segment_id else ""))
    
    print(f"Found {len(pkl_files)} pickle files")
    print(f"Alignment: extrinsics-based (IMAGE_SCALE={image_scale}, EXTRINSICS_SCALE={extrinsics_scale})")
    
    # Load first file to get faces and number of people
    first_file = os.path.join(pkl_folder, pkl_files[0])
    with open(first_file, "rb") as f:
        first_data = pickle.load(f)
    
    faces = first_data["faces"]
    n_people_first = len(first_data["outputs"])
    print(f"First frame has {n_people_first} people, {faces.shape[0]} faces")
    
    # Collect aligned vertices per person across all frames
    person_vertices = {i: [] for i in range(n_people_first)}
    frame_ids = []
    
    for pkl_file in pkl_files:
        pkl_path = os.path.join(pkl_folder, pkl_file)
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        
        outputs = data["outputs"]
        frame_id = pkl_file.replace(".pkl", "")
        frame_ids.append(frame_id)
        
        # Get camera number from frame_id (e.g., "428_000000" -> 4)
        cam_num = int(frame_id.split("_")[0][0])
        
        # Load intrinsics
        intrinsic_file = os.path.join(intrinsic_folder, f"intrinsic_{cam_num}.json")
        with open(intrinsic_file, "r") as f:
            intrinsic_data = json.load(f)
            K = np.array(intrinsic_data["intrinsic"])
            dist_coeffs = np.array(intrinsic_data["distortion_coefficients"])
            # Scale K to match SAM3D resolution (0.5x of original)
            K_scaled = adjust_K(K, 0.5)
        
        # Load extrinsics
        extrinsic_file = os.path.join(extrinsic_folder, f"extrinsic_{cam_num}_zh.json")
        R_cam_ext, t_cam_ext = load_camera_extrinsics(extrinsic_file)
        t_cam_ext = t_cam_ext.flatten()
        
        if verbose:
            print(f"\n{'='*40}")
            print(f"Frame: {frame_id} ({len(outputs)} people)")
        
        # Process each person with PnP + extrinsics alignment
        for person_idx, person_output in enumerate(outputs):
            if person_idx >= n_people_first:
                continue  # Skip if more people than first frame
            
            # Run PnP
            R_pnp, t_pnp, pnp_quality = solve_pnp_person(
                person_output, K_scaled, dist_coeffs,
                person_idx=person_idx, verbose=verbose
            )
            
            # Get body data
            J_3d_body = person_output["pred_keypoints_3d"]
            V_3d_body = person_output["pred_vertices"]
            J_2d = person_output["pred_keypoints_2d"]
            
            # Align using extrinsics
            J_world, V_world, align_info = align_person_to_ground_extrinsics(
                J_3d_body, V_3d_body, J_2d, K_scaled, R_cam_ext, t_cam_ext,
                R_pnp=R_pnp, t_pnp=t_pnp,
                foot_indices=FOOT_INDICES, world_up=WORLD_UP,
                image_scale=image_scale, extrinsics_scale=extrinsics_scale,
                verbose=verbose
            )
            
            if V_world is None:
                # Fallback: use PnP result with per-person shift
                if verbose:
                    print(f"    Person {person_idx}: Fallback alignment")
                J_cam = (R_pnp @ J_3d_body.T).T + t_pnp
                V_cam = (R_pnp @ V_3d_body.T).T + t_pnp
                # Transform to world
                t_cam_m = t_cam_ext * extrinsics_scale
                R_cam_inv = R_cam_ext.T
                V_world = (R_cam_inv @ (V_cam - t_cam_m).T).T
                # Shift to ground along up axis
                up_axis = np.argmax(np.abs(WORLD_UP))
                min_foot_up = V_world[FOOT_INDICES, up_axis].min()
                V_world[:, up_axis] -= min_foot_up
            
            person_vertices[person_idx].append(V_world)
    
    # Convert to arrays: (N_frames, N_verts, 3)
    vertices_list = []
    for person_idx in range(n_people_first):
        verts_sequence = person_vertices[person_idx]
        if len(verts_sequence) > 0:
            verts_array = np.stack(verts_sequence, axis=0)
            vertices_list.append(verts_array)
            print(f"Person {person_idx}: {verts_array.shape[0]} frames, {verts_array.shape[1]} vertices")
    
    return vertices_list, faces, frame_ids


# Example usage:
if __name__ == "__main__":
    # Paths
    pkl_folder = "./experiments/inputs/pickles/sequences"
    intrinsic_folder = "./experiments/inputs/intrinsics"
    extrinsic_folder = "./experiments/inputs/extrinsics"
    
    # Optional: filter by segment ID (e.g., "428" for files like "428_000000.pkl")
    segment_id = "428"  # Set to None to load all
    
    # Alignment parameters (matching vis_utils_custom.py)
    IMAGE_SCALE = 2.0        # SAM3D at 960x540, extrinsics at 1920x1080
    EXTRINSICS_SCALE = 0.01  # Extrinsics translation in cm -> meters
    
    print(f"Loading sequences from: {pkl_folder}")
    if segment_id:
        print(f"Filtering by segment: {segment_id}")
    
    # Load sequences with extrinsics-based alignment
    vertices_list, faces, frame_ids = load_sequence_with_extrinsics_alignment(
        pkl_folder=pkl_folder,
        intrinsic_folder=intrinsic_folder,
        extrinsic_folder=extrinsic_folder,
        segment_id=segment_id,
        image_scale=IMAGE_SCALE,
        extrinsics_scale=EXTRINSICS_SCALE,
        verbose=False  # Set to True for detailed output
    )
    
    print(f"\nLoaded {len(vertices_list)} people across {len(frame_ids)} frames")
    print(f"Frame range: {frame_ids[0]} to {frame_ids[-1]}")
    
    # Create faces list (same faces for all people)
    faces_list = [faces] * len(vertices_list)
    
    # Visualize the sequences
    visualize_mesh_sequence(
        vertices_list=vertices_list,
        faces_list=faces_list,
    )