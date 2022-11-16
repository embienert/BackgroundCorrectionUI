import os.path
import json

from BackgroundCorrection.util import DDict

defaults = {
    "io": {
        "out_dir": "out",
        "dat_file_sep": '\t',
        "head_row_count": 0
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
            "corrected_normalized": False
        }
    },
    "jar": {
        "enable": False,
        "range_start": -(2 ** 32 - 1),
        "range_stop": 2 ** 32 - 1,
        "reference_file_head_row_count": 0,
        "out_dir": "jar",
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
    "rois": {
        "enable": False,
        "ranges": [
        ],
        "out_dir": "",
        "plot": {
            "enable": True,
            "time_step": 1,
            "flip_y": False,
            "heatmap": "hot"
        }
    },
    "normalization": {
        "area": True,
        "sum": False,
        "max": False,
        "ground": False,
    }
}


def write_settings(settings: dict, filename: str = "settings.json"):
    with open(filename, "w+") as out_stream:
        json.dump(settings, out_stream, indent=4)


def load_settings(filename: str = "settings.json") -> DDict:
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

        if key in other_settings.keys() and not isinstance(other_settings[key], dict):
            settings[key] = other_settings[key]

    return DDict(settings)
