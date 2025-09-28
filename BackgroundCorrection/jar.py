import warnings

import numpy as np
from matplotlib import pyplot as plt

from BackgroundCorrection.util import apply_limits, ground
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


def jar_correct(jar_file: DataFile, intensity: np.ndarray,
                lstsq=True, lstsq_shifted=False, linear=False, use_bkg=False, **opt):
    if not lstsq and not lstsq_shifted and not linear:
        warnings.warn("No method set for JAR-correction. Using default `lstsq`.")
        lstsq = True

    jar_intensity = jar_file.ys[0]
    jar_selection = jar_file.range_selection

    jar_ranged_corrected = jar_file.ys_background_corrected

    if jar_ranged_corrected.size == 0:
        jar_corrected, jar_baseline = algorithm.correct(jar_file.ys[0], **opt)

        jar_ranged_corrected = jar_corrected[jar_selection]
        jar_ranged_baseline = jar_baseline[jar_selection]

        jar_file.ys_background_corrected = np.array([jar_ranged_corrected])
        jar_file.ys_background_baseline = np.array([jar_ranged_baseline])

    data_corrected, data_baseline = algorithm.correct(intensity, **opt)
    data_ranged_corrected = data_corrected[jar_selection]

    if use_bkg:
        jar_reference = jar_ranged_corrected.reshape(-1, 1)
        data_reference = data_ranged_corrected
    else:
        jar_reference = jar_intensity[jar_selection]
        data_reference = intensity[jar_selection]

    if lstsq or lstsq_shifted:
        scaling_factor, _, _, _ = np.linalg.lstsq(jar_reference, data_reference, rcond=None)
    else:
        # Linear scaling
        scaling_factor = np.min(data_reference / jar_reference)
    jar_intensity_scaled = scaling_factor * jar_intensity

    intensity_jar_corrected = intensity - jar_intensity_scaled

    shift = 0
    if lstsq_shifted:
        shift = -np.min(intensity_jar_corrected)
        intensity_jar_corrected = ground(intensity_jar_corrected, only_negative=True)

    return intensity_jar_corrected, jar_intensity_scaled, scaling_factor, shift
