import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from directConv.filters import CONV_FILTER

conv_shape = (3, 3)


def compute(sequence, conv_shape=conv_shape, conv_filter=CONV_FILTER):
    sequence, conv_filter = sequence.astype(np.float32), conv_filter.astype(np.float32)
    view = sliding_window_view(sequence, window_shape=conv_shape, axis=(2, 3))
    view = view.transpose(0, 2, 3, 1, 4, 5)
    return np.tensordot(view, conv_filter, axes=([3, 4, 5], [1, 2, 3]))
