import numpy as np
import scipy
from matplotlib import pyplot as plt

import gl

from sympy.physics.hydrogen import Psi_nlm
from sympy import *
from sympy.abc import r, phi, theta, Z 

rs = np.linspace(    0.0,    gl.r_max,   num = gl.N) # radial coordinate r
thetas = np.linspace(0.0,    np.pi,      num = gl.N_thetas) # polar angle
phis = np.linspace(  0.0,    2*np.pi,    num = gl.N_phis) # azimuthal angle

rs, phis, thetas = np.meshgrid(rs, phis, thetas, indexing='ij')

f = lambdify( [r, phi, theta], Psi_nlm(1, 0, 0, r, phi, theta, Z=1), "numpy")
print(f(rs, phis, thetas).shape)
print(np.unique(f(rs, phis, thetas)).shape)

np.save("Initial_PSI_n1_l0_m0_r_0to{}_theta_0to{}_phi_0to{}_3Dgrid.npy".format(gl.r_max, np.pi, 2*np.pi), f(rs, phis, thetas))