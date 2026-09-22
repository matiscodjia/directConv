from typing import cast

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage


def display(sequence, convolved):
    filtered_sequences: dict[str, np.ndarray] = {
        "No filter": sequence.transpose(0, 2, 3, 1),
        "Sobel filter": convolved[:, :, :, 0],
        "Gaussian filter": convolved[:, :, :, 1],
    }

    fig, raw_axes = plt.subplots(1, len(filtered_sequences), figsize=(15, 5))
    axes = cast(list[Axes], np.atleast_1d(raw_axes).ravel().tolist())

    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("Convolution Tensorielle")

    ims: list[tuple[AxesImage, np.ndarray]] = []
    for ax, (titre, seq) in zip(axes, filtered_sequences.items()):
        im = ax.imshow(seq[0], cmap="magma", vmin=seq.min(), vmax=seq.max())
        ax.set_title(titre)
        ax.axis("off")
        ims.append((im, seq))

    def update(frame_idx: int) -> list[AxesImage]:
        for im, seq in ims:
            im.set_array(seq[frame_idx])
        return [im for im, _ in ims]

    n_frames = min(len(seq) for _, seq in ims)
    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=33, repeat=True
    )
    _keep_alive = ani

    plt.show()
