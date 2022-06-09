import numpy as np

N = np.int32(400)
N_thetas = 100
N_phis = 200

N_timesteps = np.int(500)

r_max = np.float64(200.0)
alpha = np.float64(25)
L = alpha * r_max / 2

# Partial waves maximum number
L_max = 30

exp_of_phis = 