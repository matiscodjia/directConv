import os

import numpy as np

REF_DIR = "data/output_numpy"
FERRITE_DIR = "data/ferrite"
TOLERANCE = 1e-5

results = []
for fname in os.listdir(REF_DIR):
    ref_path = os.path.join(REF_DIR, fname)
    ferrite_path = os.path.join(FERRITE_DIR, fname)
    if not os.path.exists(ferrite_path):
        results.append((fname, "MISSING", None))
        continue

    ref = np.load(ref_path)
    out = np.load(ferrite_path)

    if ref.shape != out.shape:
        results.append((fname, "SHAPE_MISMATCH", (ref.shape, out.shape)))
        continue

    rel_err = np.max(np.abs(ref - out)) / np.max(np.abs(ref))
    verdict = "PASS" if rel_err < TOLERANCE else "FAIL"
    results.append((fname, verdict, rel_err))

for fname, verdict, detail in results:
    print(f"{verdict:15} {fname:40} {detail}")

n_fail = sum(1 for _, v, _ in results if v != "PASS")
print(f"\n{len(results) - n_fail}/{len(results)} passed")
