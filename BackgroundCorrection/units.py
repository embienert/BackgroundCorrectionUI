import numpy as np


def deg_to_rad(val: float) -> float:
    return (2 * np.pi / 360) * val


def rad_to_deg(val: float) -> float:
    return val / (2 * np.pi / 360)



def q_nm_to_q_A(val: float) -> float:
    return val / 10


def q_A_to_q_nm(val: float) -> float:
    return val * 10



def two_theta_to_q_A(val: float, lam) -> float:
    val_rad = deg_to_rad(val)
    return (4 * np.pi * np.sin(val_rad / 2)) / lam


def q_A_to_two_theta(val: float, lam) -> float:
    ret_rad = 2 * np.arcsin(lam * val / (4 * np.pi))
    return rad_to_deg(ret_rad)



def convert_x(val: float, from_unit: int, to_unit: int) -> float:
    try:
        if int(from_unit) == int(to_unit):
            return val
    except ValueError:
        raise ValueError("from_unit and to_unit must be of type Integer!")

    to_q_A = [q_nm_to_q_A, lambda x: x, two_theta_to_q_A]
    from_q_A = [q_A_to_q_nm, lambda x: x, q_A_to_two_theta]

    to_q_A_func = to_q_A[from_unit]
    from_q_A_func = from_q_A[to_unit]

    q_A_val = to_q_A_func(val)
    return from_q_A_func(q_A_val)


def unit_x_str(to_unit: int) -> str:
    unit_strs = ["q [1/nm]", "q [1/A]", "q [2θ]"]

    return unit_strs[to_unit]
