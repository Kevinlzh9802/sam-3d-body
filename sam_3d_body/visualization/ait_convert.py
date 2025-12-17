import os
import pickle
import numpy as np

def export_frame_to_npz(pkl_path, npz_out_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    outputs = data["outputs"]
    verts_list = []
    for o in outputs:
        V = np.asarray(o["pred_vertices"], dtype=np.float32)
        verts_list.append(V[None, ...])  # (1, V, 3) for this person

    # Stack persons: (P, 1, V, 3) -> (1, P, V, 3)
    verts = np.stack(verts_list, axis=0)  # (P, 1, V, 3)
    verts = np.transpose(verts, (1, 0, 2, 3))  # (1, P, V, 3)

    np.savez(npz_out_path, vertices=verts)

if __name__ == "__main__":
    pkl_path = "./experiments/inputs/pickles/calibration/428_000000.pkl"
    npz_out = "./experiments/inputs/npz/428_000000.npz"
    export_frame_to_npz(pkl_path, npz_out)
    print("Exported", npz_out)