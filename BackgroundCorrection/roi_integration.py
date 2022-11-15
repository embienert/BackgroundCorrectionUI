import numpy as np
import os


def get_area(x, intensities, roi_min, roi_max):
    selection = (x >= roi_min) & (x <= roi_max)

    x_ranged = x[selection]
    intensities_ranged = intensities[selection]

    y_area = np.trapz(x=x_ranged, y=intensities_ranged)

    return y_area


def normalize(roi_areas):
    target_y = np.ones(roi_areas.shape[1])
    roi_scales, error_sq_sum, _, _ = np.linalg.lstsq(roi_areas.T, target_y, rcond=None)

    roi_areas_scaled = (roi_areas.T * roi_scales).T
    mean_error = np.sqrt(error_sq_sum / target_y.shape[0])

    return roi_areas_scaled, mean_error


def export_rois(rois_values, filenames, rois_ranges, out_dir, name: str = ""):
    header = ",".join(["filename",
                       *[f"roi_{i}[{str(float(start)) + '_to_' + str(float(stop))}]" for i, (start, stop, color) in
                         enumerate(rois_ranges)]])
    export_data = np.concatenate((np.array(filenames, dtype=object).reshape(1, -1), rois_values)).T

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    np.savetxt(os.path.join(out_dir, name + "_rois.csv"), export_data, delimiter=",", fmt="%s", header=header, comments="")


