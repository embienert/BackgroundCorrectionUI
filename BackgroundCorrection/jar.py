import numpy as np

from BackgroundCorrection.util import apply_limits
from BackgroundCorrection.reader import read, DataFile
import BackgroundCorrection.algorithm as algorithm

from typing import Tuple


def load_jar(filename: str, head_rows: int, jar_selection_range: Tuple[float, float], x_selection):
    jar_file = read(filename, head_rows)
    jar_intensity = jar_file.ys[0]

    jar_x_ranged, _ = apply_limits(jar_file.x, selection=x_selection)
    jar_x_ranged2, jar_selection = apply_limits(jar_x_ranged, selection_range=jar_selection_range)

    jar_y_ranged, _ = apply_limits(jar_intensity, selection=x_selection)
    jar_intensity_ranged, _ = apply_limits(jar_y_ranged, selection=jar_selection)

    jar_file.x = jar_x_ranged
    jar_file.ys = np.array([jar_y_ranged])

    jar_file.x_ranged = jar_x_ranged2
    jar_file.ys_ranged = np.array([jar_intensity_ranged])
    jar_file.range_selection = jar_selection

    return jar_file


def jar_correct(jar_file: DataFile, intensity: np.ndarray, **opt):
    jar_intensity = jar_file.ys[0]
    jar_selection = jar_file.range_selection

    jar_ranged_corrected = jar_file.ys_background_corrected

    if jar_ranged_corrected.size == 0:
        jar_corrected, jar_baseline = algorithm.correct(jar_file.ys[0], **opt)

        jar_ranged_corrected = jar_corrected[jar_selection]

        jar_file.ys_background_corrected = np.array([jar_ranged_corrected])
        jar_file.ys_background_baseline = np.array([jar_baseline])

    data_corrected, data_baseline = algorithm.correct(intensity, **opt)
    data_ranged_corrected = data_corrected[jar_selection]

    scaling_factor, _, _, _ = np.linalg.lstsq(jar_ranged_corrected.reshape(-1, 1), data_ranged_corrected, rcond=None)
    jar_intensity_scaled = scaling_factor * jar_intensity

    return intensity - jar_intensity_scaled, jar_intensity_scaled, scaling_factor
