import numpy as np

N_rs = np.int32(100)
N_thetas = 30
N_phis = 180
phis = np.linspace(0.0, 2*np.pi, num = N_phis)

delta_theta = np.float64(np.pi / N_thetas)
delta_phi = np.float64(2*np.pi / N_phis)

N_timesteps = np.int(1)

r_max = np.float64(200.0)
alpha = np.float64(25)
L = alpha * r_max / 2

Temp_Env = "sin_squared"
N = np.int32(12)

delta_t = 0.1
omega = 0.057 # equivalent to lambda = 800 nm

c_au = 137.036