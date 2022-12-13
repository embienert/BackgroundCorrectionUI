import multiprocessing
import numpy as np

from BackgroundCorrection import jar, algorithm
from BackgroundCorrection.util import apply_limits, ground, normalize_area, normalize_sum, normalize_max


class ProcessingResult:
    def __init__(self):
        self.y_ranged = np.array([])

        self.y_jar_corrected = np.array([])
        self.y_jar_scaled = np.array([])
        self.jar_scaling_factor = np.nan

        self.y_background_corrected = np.array([])
        self.y_background_baseline = np.array([])

        self.y_grounded = np.array([])
        self.y_normalized = np.array([])

        self.y_result = np.array([])

        self.label = ""


def process_parallel(intensity, x_ranged, range_selection, jar_file, settings, bkg_params, label):
    result = ProcessingResult()
    result.label = label

    # Apply x-limits to intensity values
    intensity_ranged, _ = apply_limits(intensity, selection=range_selection)
    result.y_ranged = intensity_ranged

    # Apply jar-correction to intensity
    intensity_pre_bkg = intensity_ranged
    if settings["jar"]["enable"]:
        intensity_jar_corrected, jar_intensity_scaled, jar_scaling_factor = jar.jar_correct(jar_file,
                                                                                            intensity_ranged,
                                                                                            **bkg_params)

        result.y_jar_corrected = intensity_jar_corrected
        result.y_jar_scaled = jar_intensity_scaled
        result.jar_scaling_factor = jar_scaling_factor

        intensity_pre_bkg = intensity_jar_corrected

    # Perform background correction on prepared intensity data
    intensity_pre_ground = intensity_pre_bkg

    if settings["baseline"]["enable"]:
        intensity_corrected, baseline = algorithm.correct(intensity_pre_bkg, **bkg_params)
        result.y_background_corrected = intensity_corrected
        result.y_background_baseline = baseline

        intensity_pre_ground = intensity_corrected

    # Ground processed intensity data
    intensity_pre_norm = intensity_pre_ground
    if settings["normalization"]["ground"]:
        intensity_grounded = ground(intensity_pre_ground)
        result.y_grounded = intensity_grounded

        intensity_pre_norm = intensity_grounded

    # Normalize processed intensity data
    intensity_final = intensity_pre_norm
    if settings["normalization"]["area"]:
        intensity_normalized = normalize_area(x=x_ranged, y=intensity_pre_norm)
        result.y_normalized = intensity_normalized

        intensity_final = intensity_normalized
    elif settings["normalization"]["sum"]:
        intensity_normalized = normalize_sum(intensity_pre_norm)
        result.y_normalized = intensity_normalized

        intensity_final = intensity_normalized
    elif settings["normalization"]["max"]:
        intensity_normalized = normalize_max(intensity_pre_norm)
        result.y_normalized = intensity_normalized

        intensity_final = intensity_normalized

    result.y_result = intensity_final

    # print(f"Processed {label}")

    return result
