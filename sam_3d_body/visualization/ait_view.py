# file: view_multi_mesh.py
import numpy as np
from aitviewer.viewer import Viewer
from aitviewer.renderables.smpl import SMPLSequence
import pickle

def main():
    data = np.load("./experiments/inputs/npz/428_000000.npz")
    pickle_data = pickle.load(open("./experiments/inputs/pickles/calibration/428_000000.pkl", "rb"))
    vertices = data["vertices"]  # (T=1, P, V, 3)
    T, P, V, _ = vertices.shape

    # If you have faces from SMPL:
    # faces = np.load("smpl_faces.npy")  # (F,3)
    # For now I'll assume faces come from somewhere in your repo:
    faces = pickle_data["faces"]

    v = Viewer()

    # Add each person as a MeshSequence with T=1
    for p in range(P):
        verts_p = vertices[:, p, :, :]  # (1, V, 3)
        ms = SMPLSequence(verts_p, faces, name=f"person_{p}")
        v.add(ms)

    v.run()

if __name__ == "__main__":
    main()