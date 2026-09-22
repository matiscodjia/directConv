import os

import numpy as np

from directConv.convolver import compute

pairs = {}
for dirpath, _, files in os.walk("data/"):
    for fname in files:
        if fname.startswith("vid_"):
            key = fname[len("vid_") :]  # suffixe commun = clé d'appariement
            pairs.setdefault(key, {})["video"] = os.path.join(dirpath, fname)
        elif fname.startswith("fil_"):
            key = fname[len("fil_") :]
            pairs.setdefault(key, {})["filter"] = os.path.join(dirpath, fname)

for key, p in pairs.items():
    assert "video" in p and "filter" in p, f"paire incomplete pour {key}: {p}"
    video_path, filter_path = p["video"], p["filter"]
    vid, filt = np.load(video_path), np.load(filter_path)
    print(f"Video : {vid.shape} - Filter : {filt.shape}")
    convolved = compute(sequence=vid, conv_filter=filt)
    final_path = os.path.join("data/", f"output_{key}")
    np.save(final_path, convolved)
