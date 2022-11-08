import os.path

import pandas as pd
import numpy as np

from BackgroundCorrection.util import apply_limits
from BackgroundCorrection.writer import write_dat
from BackgroundCorrection.reader import read, DataFile
import BackgroundCorrection.algorithm as algorithm

from typing import Tuple


def load_jar(filename: str, head_rows: int, jar_selection_range: Tuple[float, float], x_selection):
    jar_file = read(filename, head_rows)
    jar_intensity = jar_file.ys[0]

    jar_x_ranged, _ = apply_limits(jar_file.x, selection=x_selection)
    jar_x_ranged, jar_selection = apply_limits(jar_x_ranged, selection_range=jar_selection_range)

    jar_y_ranged, _ = apply_limits(jar_intensity, selection=x_selection)
    jar_intensity_ranged, _ = apply_limits(jar_y_ranged, selection=jar_selection)

    jar_file.x_ranged = jar_x_ranged
    jar_file.ys_ranged = np.array([jar_intensity_ranged])
    jar_file.range_selection = jar_selection

    return jar_file


def jar_correct(jar_file: DataFile, intensity: np.ndarray, **opt):
    jar_intensity = jar_file.ys[0]
    jar_selection = jar_file.range_selection

    data_ranged = intensity[jar_selection]

    jar_ranged_corrected = jar_file.ys_background_corrected

    if jar_ranged_corrected.size == 0:
        jar_ranged_corrected, jar_ranged_baseline = algorithm.correct(jar_file.ys_ranged[0], **opt)

        jar_file.ys_background_corrected = np.array([jar_ranged_corrected])
        jar_file.ys_background_baseline = np.array([jar_ranged_baseline])

    data_ranged_corrected, data_ranged_baseline = algorithm.correct(data_ranged, **opt)

    scaling_factor, _, _, _ = np.linalg.lstsq(jar_ranged_corrected.reshape(-1, 1), data_ranged_corrected, rcond=None)
    jar_intensity_scaled = scaling_factor * jar_intensity

    return intensity - jar_intensity_scaled, jar_intensity_scaled, scaling_factor


def write_jar(orig_filename: str, jar_x, jar_intensity, head, sep):
    out_filename = '.'.join(orig_filename.split('.')[:-1]) + ".dat"
    out_basename = os.path.basename(out_filename)
    out_dir = os.path.dirname(out_filename)

    jar_out_dir = os.path.join(out_dir, "out", "jar")
    jar_out_path = os.path.join(jar_out_dir, out_basename)

    if not os.path.exists(jar_out_dir):
        os.mkdir(jar_out_dir)

    out_data = pd.concat([pd.DataFrame(jar_x), pd.DataFrame(jar_intensity)], axis=1)
    write_dat(out_data, jar_out_path, head=head, sep=sep)
