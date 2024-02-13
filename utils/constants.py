import numpy as np

# terrain, terr_idx, run_idx, win_idx, time, <sensor_channels>
ch_cols = dict(zip(("terrain", "terr_idx", "run_idx", "win_idx", "time"), range(5)))


class HuskyConstants:
    # Data from Reina2016 + Manufacturer specs
    # https://doi.org/10.1080/00423114.2016.1203961
    ugv_mass = 70
    ugv_wr = 0.13
    ugv_wb = 0.6
    ugv_wl = 0.55
    ugv_Bs = 1.1
    motor_Kt = 0.044
    motor_Ke = 0.141 / np.pi
    motor_I_nl = 1.35
    motor_R = 0.46
    motor_L = 0.22e-3
    gear_ratio = 78.71
    gear_eta = 0.7
    motor_I_bias = 0.59

    # Gravity constant
    g = 9.81
