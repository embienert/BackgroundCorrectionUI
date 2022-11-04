import numpy as np
import pandas as pd

from typing import List


def write_dat(data: np.ndarray, filename: str, head: List[str], columns: List[str], sep: str, include_head=True):
    head = head if include_head else ""
    head = head if head is not None else ""

    # Join all columns to rows
    output_list = map(lambda y: sep.join(map(lambda x: "%2.7e" % x, y)), data)
    body = '\n'.join(output_list)

    columns_row = sep.join(columns) if columns != [] else ""

    data_enc = bytes('\n'.join(head) + '\n' + columns_row + body)
    with open(filename, "wb+") as out_stream:
        out_stream.write(data_enc)


def extend_head(head: List[str], script_version: str, **params):
    head_extension = [
        f"BackgroundCorrection.py (Version {script_version})",
        "".join([f", {param_name} = {param_value}" for param_name, param_value in params.items()])
    ]

    return [*head_extension, *head]
