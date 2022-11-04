import os.path
import json

from BackgroundCorrection.util import DDict

defaults = {
    "io": {
        "dat_file_sep": '\t',
        "include_head": True,
        "head_row_count": 4
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
        "lambda": 1e5,
        "ratio": 7e-4,
        "out_dir": "/out",
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
        "out_dir": "/out/jar",
        "plot": {
            "enable": True,
            "jar_original": False,
            "jar_ranged": True,
            "jar_baseline": False,
            "jar_corrected": False,
            "jar_scaled": True,
            "intensity_corrected": True
        }
    },
    "rois": {
        "enable": False,
        "ranges": [
            []
        ],
        "out_dir": "/out",
        "plot": {
            "enable": True,
            "flip_y": False,
            "heatmap": "hot"
        }
    },
    "normalization": {
        "area": True,
        "max": False,
        "ground": False,
    }
}


def write_settings(settings: dict, filename: str = "settings.json"):
    with open(filename, "w+") as out_stream:
        json.dump(settings, out_stream)


def load_settings(filename: str = "settings.json"):
    if not os.path.exists(filename):
        write_settings(defaults, filename)

        return defaults

    with open(filename, "r") as in_stream:
        file_settings = json.load(in_stream)

    settings = _overwrite_settings(defaults.copy(), file_settings)
    write_settings(settings, filename)


def _overwrite_settings(settings: dict, file_settings: dict):
    for key, value in settings.items():
        if isinstance(value, dict):
            if key in file_settings.keys() and isinstance(file_settings[key], dict):
                # Recursive call
                settings[key] = _overwrite_settings(value, file_settings[key])

        if key in file_settings.keys():
            settings[key] = file_settings[key]

    return DDict(settings)

