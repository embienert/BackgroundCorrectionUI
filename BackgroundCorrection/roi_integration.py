import numpy as np
from scipy.optimize import lsq_linear
import os


def get_area(x, intensities, roi_min, roi_max):
    selection = (x >= roi_min) & (x <= roi_max)

    x_ranged = x[selection]
    intensities_ranged = intensities[selection]

    y_area = np.trapz(x=x_ranged, y=intensities_ranged)

    return y_area


def normalize_linear(roi_areas) -> (np.ndarray, float):
    sums = np.sum(roi_areas, axis=0)
    linear_scale = 1 / np.mean(sums)

    roi_areas_scaled = roi_areas * linear_scale
    mean_error = np.mean(np.ones(roi_areas.shape[1]) - roi_areas_scaled)

    return roi_areas_scaled, mean_error


def normalize(roi_areas):
    roi_areas = normalize_linear(roi_areas)

    target_y = np.ones(roi_areas.shape[1])

    # TODO: Different bounds for variables?
    # roi_scales, error_sq_sum, _, _ = np.linalg.lstsq(roi_areas.T, target_y, rcond=None)
    optimizeResult = lsq_linear(roi_areas.T, target_y, bounds=(0.5, 2))
    roi_scales = optimizeResult.x
    errors = optimizeResult.fun

    print("Scaling factors:", roi_scales)

    roi_areas_scaled = (roi_areas.T * roi_scales).T
    # mean_error = np.mean(np.sqrt(error_sq_sum / target_y.shape[0]))
    mean_error = np.mean(errors)

    print("Error:", mean_error)

    return roi_areas_scaled, mean_error


def export_rois(rois_values, filenames, rois_ranges, out_dir, name: str = ""):
    header = ",".join(["filename",
                       *[f"roi_{i}[{str(float(start)) + '_to_' + str(float(stop))}]" for i, (start, stop, color) in
                         enumerate(rois_ranges)]])
    export_data = np.concatenate((np.array(filenames, dtype=object).reshape(1, -1), rois_values)).T

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    np.savetxt(os.path.join(out_dir, name + "_rois.csv"), export_data, delimiter=",", fmt="%s", header=header, comments="")
