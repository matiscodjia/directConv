import numpy as np

sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_filter = sobel_filter[np.newaxis, ...]
SOBEL_FILTER = np.repeat(sobel_filter, 3, axis=0)

gaussian_filter = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) * 1 / 16
gaussian_filter = gaussian_filter[np.newaxis, ...]
GAUSSIAN_FILTER = np.repeat(gaussian_filter, 3, axis=0)

CONV_FILTER = np.stack([SOBEL_FILTER, GAUSSIAN_FILTER], axis=0)
