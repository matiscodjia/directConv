import csv
import json
import os
import traceback
import tracemalloc
from datetime import datetime, timezone
from time import perf_counter

import numpy as np

from directConv.convolver import compute
from directConv.filters import CONV_FILTER

# sequence (n, 3, 720, 1280)

# dtypes de filtres à comparer
FILTER_DTYPES = [np.int16, np.int32, np.float16, np.float32, np.float64]


def make_filter(dtype):
    """Reconstruit CONV_FILTER dans le dtype demandé."""
    return CONV_FILTER.astype(dtype)


def measure(sequence, conv_filter):
    """Retourne (durée en s, pic mémoire en Mo) d'un appel à compute."""
    tracemalloc.start()
    start = perf_counter()
    compute(sequence, conv_filter=conv_filter)
    duration = perf_counter() - start
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return duration, peak / 1e6


class CSVAppender:
    """Écrit les lignes une à une dans un CSV et flush à chaque écriture.

    Chaque ligne est persistée dès qu'elle est calculée : la RAM ne grossit
    pas avec le nombre de résultats, et un plantage en cours de benchmark
    laisse le CSV avec tout ce qui a été mesuré jusque-là.
    """

    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        # On repart d'un fichier propre à chaque exécution.
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def append(self, row):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())


class ErrorLogAppender:
    """Journalise les erreurs au format JSON Lines (un objet JSON par ligne).

    Même philosophie que CSVAppender : chaque erreur est écrite puis flushée
    immédiatement sur le disque, donc rien n'est affiché sur le terminal et le
    fichier reste exploitable même si le programme est interrompu ensuite.

    Attention : cela ne rattrape QUE les exceptions Python. Un OOM-kill de l'OS
    (SIGKILL) ne passe par aucun except et ne peut donc pas être journalisé.
    """

    def __init__(self, path):
        self.path = path
        # On repart d'un journal propre à chaque exécution.
        open(path, "w").close()

    def log(self, context, exc):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": context,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())


def bench_by_sequence_length(n_iters=10, max_images=3):
    """Temps et RAM en fonction du nombre d'images (filtre float32 par défaut)."""
    fieldnames = ["sequence_length", "mean_duration_s", "mean_peak_mem_mb"]
    appender = CSVAppender("time_and_mem_per_sequence_length.csv", fieldnames)
    errors = ErrorLogAppender("bench_by_sequence_length_errors.jsonl")

    for j in range(max_images):
        n_images = j + 1
        try:
            durations, peaks = [], []
            for i in range(n_iters):
                random_sequence = np.random.randint(
                    low=0, high=255, size=(n_images, 3, 720, 1280)
                )
                duration, peak = measure(random_sequence, CONV_FILTER)
                durations.append(duration)
                peaks.append(peak)
            row = {
                "sequence_length": n_images,
                "mean_duration_s": np.mean(durations),
                "mean_peak_mem_mb": np.mean(peaks),
            }
            appender.append(row)
        except Exception as exc:  # noqa: BLE001
            errors.log({"benchmark": "sequence_length", "sequence_length": n_images}, exc)
            continue


def bench_by_filter_dtype(n_iters=10, n_images=3):
    """Temps et RAM en fonction du dtype (filtre ET séquence, nombre d'images fixe)."""
    fieldnames = ["filter_dtype", "mean_duration_s", "mean_peak_mem_mb"]
    appender = CSVAppender("time_and_mem_per_filter_dtype.csv", fieldnames)
    errors = ErrorLogAppender("bench_by_filter_dtype_errors.jsonl")

    for dtype in FILTER_DTYPES:
        name = np.dtype(dtype).name
        try:
            conv_filter = make_filter(dtype)
            durations, peaks = [], []
            for i in range(n_iters):
                random_sequence = np.random.randint(
                    low=0, high=255, size=(n_images, 3, 720, 1280)
                ).astype(dtype)
                duration, peak = measure(random_sequence, conv_filter)
                durations.append(duration)
                peaks.append(peak)
            row = {
                "filter_dtype": name,
                "mean_duration_s": np.mean(durations),
                "mean_peak_mem_mb": np.mean(peaks),
            }
            appender.append(row)
        except Exception as exc:  # noqa: BLE001
            errors.log({"benchmark": "filter_dtype", "filter_dtype": name}, exc)
            continue


if __name__ == "__main__":
    bench_by_sequence_length(n_iters=10, max_images=200)
    bench_by_filter_dtype(n_iters=10, n_images=200)
