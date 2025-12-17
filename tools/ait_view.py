# # file: view_multi_mesh.py
import numpy as np
from aitviewer.viewer import Viewer
# from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.renderables.meshes import Meshes
import os
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

def visualize_multi_person(vertices, faces, id_range=None):
    """
    Visualize multiple people in aitviewer.
    
    Args:
        vertices_list: List of numpy arrays, each of shape (N_verts, 3) or (T, N_verts, 3)
        faces_list: List of numpy arrays, each of shape (N_faces, 3)
        colors: Optional list of RGB colors for each person, each color as (R, G, B) 
                with values in [0, 1]. If None, uses default colors.
    """
    # Create viewer
    v = Viewer()
    # Add each person to the scene
    for k in range(vertices.shape[0]):  
        if id_range is None or (k >= id_range[0] and k <= id_range[1]):
            vertices_k = vertices[k, ...]
            mesh = Meshes(vertices=vertices_k, faces=faces, name=f"Person_{k}")
            v.scene.add(mesh)
    
    v.run()
    return v

def read_vertices_and_faces(data):
    outputs = data["outputs"]
    verts_list = []
    for o in outputs:
        V = np.asarray(o["pred_vertices"], dtype=np.float32)
        verts_list.append(V)  # (V, 3) for this person
    vertices = np.stack(verts_list, axis=0)  # (P, V, 3)
    faces = data["faces"]
    return vertices, faces

def print_foot_coords(vertices, id_range=None):
    for k in range(vertices.shape[0]):
        if id_range is not None and k < id_range[0] or k > id_range[1]:
            continue
        vertices_k = vertices[k, ...]
        print(f"Person {k}:")
        print(f"Left foot: {vertices_k[13, :]}")
        print(f"Right foot: {vertices_k[14, :]}")
        print(f"Left ankle: {vertices_k[15, :]}")
        print(f"Right ankle: {vertices_k[18, :]}")

# Example usage:
if __name__ == "__main__":
    pkl_path = "./experiments/inputs/pickles/calibration/428_000000.pkl"
    data = pickle.load(open(pkl_path, "rb"))
    vertices, faces = read_vertices_and_faces(data)
    id_range = (0, 25)

    # Visualize
    print_foot_coords(vertices=vertices, id_range=(0,1))
    visualize_multi_person(vertices=vertices, faces=faces, id_range=(0,25))
    c = 9