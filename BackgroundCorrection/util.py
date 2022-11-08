from typing import Tuple, Optional

import numpy as np


class DDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def apply_limits(data: np.ndarray, selection_range: Optional[Tuple[float, float]] = None, selection=None):
    if selection is None:
        selection_min, selection_max = selection_range

        selection = (data >= selection_min) & (data <= selection_max)

    return data[selection], selection


def normalize_area(x, y):
    area = np.abs(np.trapz(x=x, y=y))

    return y / area


def normalize_sum(y):
    y_sum = np.abs(np.sum(y))

    return y / y_sum


def normalize_max(y):
    y_max = np.max(y)

    return y / y_max


def ground(y):
    y_min = np.min(y)

    return y - y_min
