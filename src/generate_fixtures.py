import numpy as np

np.random.seed(42)

BASELINE = dict(N=1, C=3, K=2, HW=128)
KH, KW = 3, 3

SWEEPS = {
    "N": [1, 2, 4, 8, 16, 32],
    "C": [1, 3, 8, 16],
    "K": [1, 2, 4, 8, 16],
    "HW": [32, 64, 128, 256, 720],
}


def make_case(N: int, C: int, K: int, HW: int):
    video = np.random.rand(N, C, HW, HW).astype(np.float32)
    filt = np.random.rand(K, C, KH, KW).astype(np.float32)
    return video, filt


for axis, values in SWEEPS.items():
    for v in values:
        params = {**BASELINE, axis: v}
        video, filt = make_case(**params)
        tag = f"{axis}={v}"
        np.save(f"vid_{axis}_{tag}.npy", video)
        np.save(f"fil_{axis}_{tag}.npy", filt)
