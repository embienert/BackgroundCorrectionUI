from typing import List

import pyspectra as spc
import pandas as pd
import numpy as np


def read_raman(filename: str, head_rows: int):
    with open(filename, 'r') as in_stream:
        file_content = in_stream.read()

    lines_raw = file_content.split('\n')
    lines = np.array(list(map(float,
                              map(lambda row: row.split(), lines_raw[head_rows:-1]))),
                     dtype=object)
    header = lines_raw[:head_rows]

    nr_columns = len(lines[0])
    try:
        column_names = list(set(lines_raw[0].split()[1:]))

        if nr_columns != len(column_names):
            raise IndexError
    except IndexError:
        column_names = ['x', *[f"I_{y_index}" for y_index in range(nr_columns-1)]]

    return lines, header, column_names


def read_chi(filename: str, head_rows: int):
    with open(filename, 'r') as in_stream:
        file_content = in_stream.read()

    lines_raw = file_content.split('\n')
    lines = np.array(list(map(float,
                              map(lambda row: row.split(), lines_raw[head_rows:-1]))),
                     dtype=object)
    header = lines_raw[:head_rows]

    return lines, header, []


def read_spc(filename: str):
    # This function was taken from the pyspectra library to be modified to this programs needs
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

    return out.to_numpy(dtype=np.float64), [], out.columns


def read(filename: str, head_rows: int):
    file_ext = filename.split('.')[-1]

    if file_ext in ["xy", "dat"]:
        file_content, header, columns = read_chi(filename, head_rows=head_rows)
    elif file_ext in ["txt", "raman"]:
        file_content, header, columns = read_raman(filename, head_rows=head_rows)
    elif file_ext in ["spc"]:
        file_content, header, columns = read_spc(filename)
    else:
        raise NotImplementedError(f"The filetype {file_ext} is not supported (yet).")

    return file_content, header, columns


def read_many(filenames: List[str], head_rows: int):
    file_contents = []
    headers = []
    columns = []

    for filename in filenames:
        file_content, header, data_columns = read(filename, head_rows)

        file_contents.append(file_content)
        headers.append(header)
        columns.append(data_columns)

    return file_contents, headers, columns


# TODO: File IO with DataFile class instead of single variables?
