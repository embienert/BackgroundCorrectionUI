import os.path

import pandas as pd
import numpy as np

from BackgroundCorrection.util import apply_wave_range
from BackgroundCorrection.writer import write_dat
from BackgroundCorrection.reader import read
import BackgroundCorrection.algorithm as algorithm

from typing import Tuple


def load_jar(filename: str, head_rows: int, jar_selection_range: Tuple[float, float], x_selection):
    jar_read, _ = read(filename, head_rows)

    range_min, range_max = jar_selection_range

    jar_intensity, _ = apply_wave_range(jar_read, jar_read.columns[1], selection=x_selection)
    jar_intensity_ranged, jar_selection = apply_wave_range(jar_read, jar_read.columns, wave_min=range_min, wave_max=range_max)


    return jar_intensity.to_numpy(), jar_intensity_ranged.to_numpy(), jar_selection


def jar_correct(jar_intensity: np.ndarray, jar_intensity_ranged: np.ndarray, jar_selection, intensity: np.ndarray, **opt):
    data_ranged = intensity[jar_selection]

    jar_ranged_corrected, jar_ranged_baseline = algorithm.correct(jar_intensity_ranged, **opt)
    data_ranged_corrected, data_ranged_baseline = algorithm.correct(data_ranged, **opt)

    scaling_factor = np.linalg.lstsq(jar_ranged_corrected.reshape(-1, 1), data_ranged_corrected)
    jar_intensity_scaled = scaling_factor * jar_intensity

    return intensity - jar_intensity_scaled


def write_jar(orig_filename: str, jar_x, jar_intensity, head, sep):
    out_filename = '.'.join(orig_filename.split('.')[:-1]) + ".dat"
    out_basename = os.path.basename(out_filename)
    out_dir = os.path.dirname(out_filename)

    jar_out_dir = os.path.join(out_dir, "out", "jar")
    jar_out_path = os.path.join(jar_out_dir, out_basename)

    if not os.path.exists(jar_out_dir):
        os.mkdir(jar_out_dir)

    out_data = pd.concat([pd.DataFrame(jar_x), pd.DataFrame(jar_intensity)], axis=1)
    write_dat(out_data, jar_out_path, head=head, sep=sep, include_head=True)
