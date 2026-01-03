import os.path
import json

BASE_PATH = os.path.dirname(os.path.realpath(__file__))

defaults = {
    "parallel": {
        "enable": True,
        "cores": "auto"
    },
    "io": {
        "out_dir": "bkgcorr",
        "debug": True,
        "debug_dir": "debug",
        "dat_file_sep": '\t',
        "head_row_count": 0,
        "header": {
            "baseline_params": True,
            "jar_params": True,
            "roi_params": True,
            "normalization": True,
            "additional": [
                "plot.time_step",
                "plot.x_unit",
                "plot.y_unit",
            ]
        }
    },
    "data": {
        "range_start": -(2 ** 32 - 1),
        "range_stop": 2 ** 32 - 1,
        "unit_in": 1,
        "unit_out": 1
    },
    "baseline": {
        "enable": True,
        "algorithm": 1,
        "itermax": 500,
        "lam": 1e5,
        "ratio": 7e-4,
        "out_dir": "",
        "plot": {
            "enable": True,
            "original": True,
            "baseline": True,
            "corrected": True,
            "corrected_normalized": False,
            "test_datasets": [  # Indices of dataset you want to plot
                10
            ]
        }
    },
    "jar": {
        "enable": False,
        "method": {
            "lstsq": False,
            "lstsq_shifted": False,  # Same as lstsq, but shifts the result to guarantee non-negative values only
            "linear": False,  # Uses biggest linear scaling factor that does not cause negative values
            "advanced": True,  # Solve constrained linear scaling+offset problem
            "use_bkg": False  # Calculate jar scaling based on background-corrected probe and jar intensities
        },
        "range_start": -(2 ** 32 - 1),
        "range_stop": 2 ** 32 - 1,
        "reference_file_head_row_count": 0,
        "out_dir": "jarcorr",
        "plot": {
            "enable": True,
            "jar_original": False,
            "jar_ranged": True,
            "jar_baseline": False,
            "jar_corrected": False,
            "jar_scaled": True,
            "intensity_original": True,
            "intensity_corrected": True
        }
    },
    "normalization": {
        "area": True,
        "sum": False,
        "max": False,
        "ground": False,
    },
    "rois": {
        "enable": False,
        "selection": {
            "area": True,
            "max": False
        },
        "ranges": [
        ],
        "out_dir": "",
        "normalize": {
            "sum": True,
            "sum_linear": False,
            "max_total": False,  # Previously max
            "max": False,
        },
        "plot": {
            "enable": True,
            "ratio": [6, 2]
        }
    },
    "plot": {
        "enable": True,
        "time_step": 1,
        "display_time_range_start": 0,
        "display_time_range_stop": 2 ** 32 - 1,
        "flip_x_data": False,
        "flip_y_data": False,
        "flip_x_ticks": False,
        "flip_y_ticks": False,
        "x_unit": "",
        "y_unit": "s",
        "heatmap": "hot",
        "colorbar": False
    }
}
_NOT_ENABLED_DEFAULT = "N/A"

from BackgroundCorrection.util import DDict


def write_settings(settings: dict, filename=None):
    if not filename:
        filename = os.path.join(BASE_PATH, "settings.json")

    with open(filename, "w+") as out_stream:
        json.dump(settings, out_stream, indent=4)


def load_settings(filename=None) -> DDict:
    if not filename:
        filename = os.path.join(BASE_PATH, "settings.json")

    if not os.path.exists(filename):
        write_settings(defaults, filename)

        return _overwrite_settings(defaults.copy(), defaults)

    with open(filename, "r") as in_stream:
        file_settings = json.load(in_stream)

    settings = _overwrite_settings(defaults.copy(), file_settings)
    write_settings(settings, filename)

    return settings


def _overwrite_settings(settings: dict, other_settings: dict) -> DDict:
    for key, value in settings.items():
        if isinstance(value, dict):
            if key in other_settings.keys() and isinstance(other_settings[key], dict):
                # Recursive call
                settings[key] = _overwrite_settings(value, other_settings[key])
            else:
                settings[key] = _overwrite_settings(value, value)

        if key in other_settings.keys() and not isinstance(other_settings[key], dict):
            settings[key] = other_settings[key]

    return DDict(settings)


def option_to_str(settings: DDict, key: str):
    if "enable" in settings.keys() and "enable" not in key and not settings["enable"]:
        return _NOT_ENABLED_DEFAULT

    key_split = key.split(".", maxsplit=1)

    if len(key_split) == 1:
        base_key = key_split[0]
        select = settings[base_key]

        if isinstance(select, str):
            select = f"\"{select}\""

        return str(select)

    base_key, residual_key = key_split
    select = settings[base_key]

    return key + "=" + option_to_str(select, residual_key)
