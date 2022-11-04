import numpy as np


def get_area(x, intensities, roi_min, roi_max):
    selection = x >= roi_min & x <= roi_max

    x_ranged = x[selection]
    intensities_ranged = intensities[selection]

    y_sum = np.sum(intensities_ranged, axis=1)
    y_area = np.trapz(x=x_ranged, y=y_sum)

    return y_area
