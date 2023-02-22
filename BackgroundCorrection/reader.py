import spc_spectra as spc
import pandas as pd
import numpy as np

import pyspectra.readers.read_spc

from typing import List
import os.path


class DataFile:
    def __init__(self, filename: str, content: np.ndarray, head: List[str]):
        self.filename: str = filename
        self.basename: str = os.path.basename(filename)

        self.data: np.ndarray = content
        self.head: List[str] = head
        self.times: List[float] = []
        self.time_unit = ""

        self.x = np.copy(self.data[0]) if self.data.size != 0 else np.array([])
        self.ys = np.copy(self.data[1:]) if self.data.size != 0 else np.array([])

        self.x_ranged = np.array([])
        self.ys_ranged = np.array([])
        self.range_selection = np.array([])

        self.ys_jar_corrected = np.array([])
        self.jar_scaling_factors = np.array([])

        self.ys_background_corrected = np.array([])
        self.ys_background_baseline = np.array([])

        self.ys_normalized = np.array([])
        self.ys_grounded = np.array([])

        self.x_result = np.array([])
        self.ys_result = np.array([])

    def labels(self):
        basename = os.path.basename(self.filename)

        return [f"{basename}[{i+1}]" for i in range(self.ys.shape[0])]

    def set_time_step(self, time_step, unit: str):
        self.times = np.arange(0, self.ys_result.shape[0] * time_step, time_step)
        self.time_unit = unit

    def write_dat(self, out_dir: str, sep: str):
        head = self.head if self.head is not None else []
        if self.times:
            # Add another "header" row containing time column labels
            head = [*head, "x" + sep + sep.join(map(lambda t: f"{t}{self.time_unit}", self.times))]
        else:
            # Add another "header" row containing column labels (numbers 1..#Enries)
            # head = [*head, "x" + sep + sep.join(map(str, np.arange(0, self.ys_result.shape[0], 1) + 1))]
            pass

        data = np.concatenate((self.x_result.reshape(1, -1), self.ys_result)).T

        # Join all columns to rows
        output_list = map(lambda y: sep.join(map(lambda x: "%2.7e" % x, y)), data)
        body = '\n'.join(output_list)

        data = '\n'.join(head) + '\n' + body
        with open(os.path.join(out_dir, self.filename), "w+") as out_stream:
            out_stream.write(data)

        # TODO: Change to logging
        print(f"Wrote data to {out_dir}")

    def extend_head_legacy(self, script_version: str, **params):
        head_extension = [
            f"BackgroundCorrection.py (Version {script_version})",
            "".join([f", {param_name} = {param_value}" for param_name, param_value in params.items()])[2:]
        ]

        self.head = [*head_extension, *self.head]

    def extend_head(self, head_extension: List[str]):
        self.head = [*head_extension, *self.head]

    def file(self) -> str:
        return str(os.path.basename(self.filename))


def read_raman(filename: str, head_rows: int) -> (np.ndarray, List[str], List[str]):
    with open(filename, 'r') as in_stream:
        file_content = in_stream.read()

    lines_raw = file_content.split('\n')
    lines = np.vstack(list(map(lambda row: np.array([float(elem) for elem in row]),
                               map(lambda row: row.split(), lines_raw[head_rows:-1])))).T
    header = lines_raw[:head_rows]

    return lines, header


def read_spc(filename: str) -> (np.ndarray, List[str], List[str]):
    # This function was taken from the pyspectra library to be modified to this programs needs
    # TODO: Rewrite to work without DataFrame

    out = pd.DataFrame()

    f = spc.File(filename)  # Read file
    if f.dat_fmt.endswith('-xy'):
        for s in f.sub:
            x = s.x
            y = s.y

            out["RamanShift (cm-1)"] = x
            out[str(round(s.subtime))] = y
    else:
        for s in f.sub:
            x = f.x
            y = s.y

            out["RamanShift (cm-1)"] = x
            out[str(round(s.subtime))] = y

    return out.to_numpy(dtype=np.float64).T, []


def read(filename: str, head_rows: int) -> DataFile:
    file_ext = filename.split('.')[-1]

    if file_ext in ["txt", "raman", "xy", "dat"]:
        file_content, header = read_raman(filename, head_rows=head_rows)
    # elif file_ext in ["xy", "dat"]:
    #     file_content, header, columns = read_chi(filename, head_rows=head_rows)
    elif file_ext in ["spc"]:
        file_content, header = read_spc(filename)
    else:
        raise NotImplementedError(f"The filetype {file_ext} is not supported (yet).")

    # TODO: Check file validity

    return DataFile(filename, file_content, header)


def read_many(filenames: List[str], head_rows: int) -> List[DataFile]:
    dataFiles = [read(filename, head_rows) for filename in filenames]

    return dataFiles
