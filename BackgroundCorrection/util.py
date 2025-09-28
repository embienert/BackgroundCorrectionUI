from typing import Tuple, Optional
import itertools

import numpy as np


class DDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    @staticmethod
    def prettify_tree(tree: dict, path: tuple = ()) -> str:
        output = ""
        for key, value in tree.items():
            if isinstance(value, dict):
                output += DDict.prettify_tree(value, path + (key, ))
            else:
                prefix = "".join([str(subpath.strip()) + "." for subpath in path if subpath.strip()])
                output += f"{prefix}{key}: {value}\n"

        return output

    def prettify(self):
        return self.prettify_tree(self)


def dictify(d):
    cpy = {}
    for key, value in d.items():
        if isinstance(value, DDict):
            cpy[key] = dictify(value)
        else:
            cpy[key] = value
    return cpy


def apply_limits(data: np.ndarray, selection_range: Optional[Tuple[float, float]] = None, selection=None):
    if selection is None:
        selection_min, selection_max = selection_range

        selection = (data >= selection_min) & (data <= selection_max)

    return data[selection], selection


def ranges(iterable):
    for _, group in itertools.groupby(enumerate(iterable), lambda pair: pair[1] - pair[0]):
        group = list(group)
        yield group[0][1], group[-1][1]


def normalize_area(x, y):
    area = np.abs(np.trapz(x=x, y=y))

    return y / area


def normalize_sum(y):
    y_sum = np.abs(np.sum(y))

    return y / y_sum


def normalize_max(y):
    y_max = np.max(y)

    return y / y_max


def ground(y, only_negative=False):
    y_min = np.min(y)

    if only_negative:
        if y_min < 0:
            return y - y_min
        return y
    return y - y_min
