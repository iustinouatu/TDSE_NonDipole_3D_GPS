import numpy as np
from matplotlib import pyplot as plt

import gl

from sympy.physics.hydrogen import Psi_nlm
from sympy import *
from sympy.abc import r, phi, theta, Z 

rs = np.linspace(    0.0,    gl.r_max,   num = gl.N_rs - 1) # radial coordinate r
thetas = np.linspace(0.0,    np.pi,      num = gl.N_thetas) # polar angle
phis = np.linspace(  0.0,    2*np.pi,    num = gl.N_phis) # azimuthal angle

rs, phis, thetas = np.meshgrid(rs, phis, thetas, indexing='ij')

f = lambdify( [r, phi, theta], Psi_nlm(1, 0, 0, r, phi, theta, Z=1), "numpy")
print(f(rs, phis, thetas).shape)
# print(np.unique(f(rs, phis, thetas)).shape)

wavef_grid_values = f(rs, phis, thetas) # shape (gl.N_rs - 1, gl.N_phis, gl.N_thetas)
wavef_grid_values_new = np.zeros( (wavef_grid_values.shape[0], wavef_grid_values.shape[2], wavef_grid_values.shape[1]) , dtype=np.complex128)

for i in range(wavef_grid_values.shape[0]):
    wavef_grid_values_new[i, :, :] = wavef_grid_values[i, :, :].T


np.save("Initial_PSI_n1_l0_m0_r_0to{}_theta_0to{}_phi_0to{}_3Dgrid.npy".format(gl.r_max, np.pi, 2*np.pi), wavef_grid_values_new)