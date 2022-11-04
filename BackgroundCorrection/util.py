import pandas as pd


class DDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def apply_wave_range(df: pd.DataFrame, column_name: str,
                     wave_min=None, wave_max=None,
                     selection=None):
    if selection is None:
        selection = df[column_name] >= wave_min & df[column_name] <= wave_max

    return df[column_name].loc[selection], selection
