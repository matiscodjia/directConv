import imageio.v3 as iio
import numpy as np


def load_nframes(n_frames: int | None = None) -> np.ndarray:
    frames: np.ndarray = iio.imread(uri="imageio:cockatoo.mp4", index=None)
    frames = frames.transpose(0, 3, 1, 2)
    assert frames.shape[1] in (3, 4), (
        f"Canal C attendu en position C de valeur 3 (RGB) ou 4 (RGBA), "
        f"Résultat obtenu : {frames.shape[1]}, vérifier l'ordre des axes"
    )
    if n_frames is None:
        print(f"Tenseur chargé : {frames.shape}")
        return frames
    if n_frames <= 0:
        raise ValueError(
            f"Le nombre de frames doit être au moins 1 valeur entrée: {n_frames}"
        )
    frames = frames[:n_frames]

    print(f"Tenseur chargé : {frames.shape}")
    print(f"Tenseur de type {(frames.dtype)}")
    return frames


MAX_DOWNLOAD_BYTES = 300_000_000


def download_video(url: str, cache_dir: str = "data/videos") -> str:
    """Downloads `url` into `cache_dir` (once) and returns the local path."""
    import os
    import shutil
    import urllib.parse
    import urllib.request

    name = os.path.basename(urllib.parse.urlparse(url).path) or "video.mp4"
    if not os.path.splitext(name)[1]:
        name += ".mp4"
    path = os.path.join(cache_dir, name)
    if os.path.exists(path):
        return path
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Téléchargement de {url} -> {path}")
    # Many hosts answer 403 to urllib's default User-Agent.
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    partial = path + ".part"
    try:
        with urllib.request.urlopen(request, timeout=30) as response, open(partial, "wb") as out:
            size = int(response.headers.get("Content-Length") or 0)
            if size > MAX_DOWNLOAD_BYTES:
                raise ValueError(f"video is {size / 1e6:.0f} MB, the limit is {MAX_DOWNLOAD_BYTES / 1e6:.0f} MB")
            shutil.copyfileobj(response, out)
        os.replace(partial, path)
    finally:
        if os.path.exists(partial):
            os.remove(partial)
    return path


def resolve_source(source: str) -> str:
    """A local path or `imageio:` resource is returned as is, a URL is downloaded."""
    if source.startswith(("http://", "https://")):
        return download_video(source)
    return source


def video_fps(source: str, default: float = 24.0) -> float:
    try:
        return float(iio.immeta(resolve_source(source), plugin="FFMPEG")["fps"])
    except Exception:
        return default


def _fit(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Center-crops an (H, W, 3) frame to the aspect of `size` = (H, W), then resizes."""
    from PIL import Image

    h, w = size
    img = Image.fromarray(frame)
    src_w, src_h = img.size
    scale = max(w / src_w, h / src_h)
    crop_w, crop_h = min(src_w, round(w / scale)), min(src_h, round(h / scale))
    left, top = (src_w - crop_w) // 2, (src_h - crop_h) // 2
    box = (left, top, left + crop_w, top + crop_h)
    return np.asarray(img.resize((w, h), Image.Resampling.BILINEAR, box=box))


def decode_frames(source: str, n_frames: int | None = None) -> list[np.ndarray]:
    """The first `n_frames` frames of a video as (H, W, 3) uint8 arrays.

    `source` is a local path, an `imageio:` resource, or an http(s) URL (which is
    downloaded first, see `download_video`). Frames are decoded one by one and
    decoding stops after `n_frames`, so a long video is never fully read just to
    keep its first frames.
    """
    from itertools import islice

    if n_frames is not None and n_frames <= 0:
        raise ValueError(f"n_frames must be at least 1, got {n_frames}")
    return list(islice(iio.imiter(resolve_source(source)), n_frames))


def iter_frames(source: str, n_frames: int | None = None):
    """Like `decode_frames`, but yields one (H, W, 3) frame at a time, so a caller that only needs a
    resized copy never holds the full-size video in memory."""
    from itertools import islice

    return islice(iio.imiter(resolve_source(source)), n_frames)


def fit_frames(frames: list[np.ndarray], size: tuple[int, int]) -> np.ndarray:
    """(N, H, W, 3) uint8: every frame center-cropped to the aspect of `size` = (H, W), then resized."""
    return np.stack([_fit(f, size) for f in frames])


def load_video(
    source: str, n_frames: int | None = None, size: tuple[int, int] | None = None
) -> np.ndarray:
    """Loads a video as an (N, C, H, W) uint8 array. `size` = (H, W) crops and resizes every frame,
    to land on a shape the Rust build has compiled in."""
    frames = decode_frames(source, n_frames)
    stacked = fit_frames(frames, size) if size is not None else np.stack(frames)
    return stacked.transpose(0, 3, 1, 2)
