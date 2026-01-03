import warnings

import numpy as np
from scipy.optimize import minimize

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


def cl_objective(params, y_fit, y_ref):
    scalar, offset = params

    diff = scalar * y_fit + offset - y_ref

    return np.sum(diff ** 2)


def cl_constraint(params, y_fit, y_ref):
    scalar, offset = params

    return y_ref - (scalar * y_fit + offset)


CL_BOUNDS = [(0, None), (None, None)]


def jar_correct(jar_file: DataFile, intensity: np.ndarray,
                lstsq=True, lstsq_shifted=False, linear=False, advanced=False, use_bkg=False, **opt):
    if not lstsq and not lstsq_shifted and not linear and not advanced:
        warnings.warn("No method set for JAR-correction. Using default `advanced`.")
        advanced = True

    jar_intensity = jar_file.ys[0]
    jar_selection = jar_file.range_selection

    jar_ranged_corrected = jar_file.ys_background_corrected
    jar_ranged_baseline = jar_file.ys_background_baseline

    if jar_ranged_corrected.size == 0:
        jar_corrected, jar_baseline = algorithm.correct(jar_file.ys[0], **opt)

        jar_ranged_corrected = jar_corrected[jar_selection]
        jar_ranged_baseline = jar_baseline[jar_selection]

        jar_file.ys_background_corrected = np.array([jar_ranged_corrected])
        jar_file.ys_background_baseline = np.array([jar_ranged_baseline])

    data_corrected, data_baseline = algorithm.correct(intensity, **opt)
    data_ranged_corrected = data_corrected[jar_selection]

    if use_bkg:
        jar_reference = jar_ranged_corrected
        data_reference = data_ranged_corrected
    else:
        jar_reference = jar_intensity[jar_selection]
        data_reference = intensity[jar_selection]

    if lstsq or lstsq_shifted:
        scaling_factor, _, _, _ = np.linalg.lstsq(jar_reference.reshape(-1, 1), data_reference, rcond=None)
        offset = 0
    elif advanced:
        # Constrained linear scalar+offset solver
        jar = jar_reference.reshape(-1,)

        constraints = {
            "type": "ineq",
            "fun": cl_constraint,
            "args": (jar, data_reference)
        }

        initial = [1.0, np.min(data_reference - jar)]

        result = minimize(
            cl_objective,
            x0=initial,
            args=(jar, data_reference),
            bounds=CL_BOUNDS,
            constraints=constraints,
            options={
                "maxiter": 1_000,
            },
        )

        scaling_factor, offset = result.x

        # print(result.x)
        # print(initial)
        # print(result.nit)
        # print(cl_objective(initial, jar, data_reference))
        # print(result.fun)
    else:
        # Linear scaling
        data_jar_ratio = data_reference.reshape(-1, 1) / jar_reference.reshape(-1, 1)
        data_jar_ratio_positives = data_jar_ratio[data_jar_ratio > 0]

        if data_jar_ratio_positives.any():
            scaling_factor = np.min(data_jar_ratio_positives)
        else:
            scaling_factor = np.max(data_jar_ratio)
        offset = 0
    jar_intensity_scaled = scaling_factor * jar_intensity + offset

    intensity_jar_corrected = intensity - jar_intensity_scaled

    if lstsq_shifted:
        offset = -np.min(intensity_jar_corrected)
        intensity_jar_corrected = ground(intensity_jar_corrected, only_negative=True)

    return intensity_jar_corrected, jar_intensity_scaled, scaling_factor, offset, data_corrected, data_baseline
