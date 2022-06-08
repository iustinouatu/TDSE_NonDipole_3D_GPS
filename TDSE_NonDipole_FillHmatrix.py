import numpy as np
import scipy
from matplotlib import pyplot as plt

from scipy.special import roots_legendre, legendre
from numpy.linalg import eig

import gl

poly_Nplus1 = legendre(gl.N+1)
poly_Nplus1_der = poly_Nplus1.deriv()
a =  np.roots(poly_Nplus1_der)
roots = a[0] # shape (N+1, )


# Non-Linear Mapping 
def r(x):
    return gl.L * (1 + x) / (1 - x + gl.alpha)

def r_prime(x):
    return gl.L * (2 + gl.alpha) / (1 - x + gl.alpha)**2


# H_{l}^{0}(x) assembly functions
def V_l(x, l, Z):
    return (l*(l+1) / (2*r(x))) + Coulomb(r(x), Z)

def Coulomb(r, Z):
    return (-Z/r)


Z = 1
H = np.zeros(  (gl.N-1, gl.N-1, gl.L_max),   dtype=np.complex128)
pref = (gl.N+1) * (gl.N+2) / 6
diag_values_Kinetic = np.array([  (1/r_prime(x)) * pref * (1/(1-x**2)) * (1/r_prime(x)) for x in roots])

for l in range(gl.L_max):
    diag_values_Coulomb_and_Laser = np.array([  V_l(x, l, Z) for x in roots ])
    np.fill_diagonal( H[:, :, l],   diag_values_Kinetic + diag_values_Coulomb_and_Laser )
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if (i != j):
                H[i, j, l] = (1/r_prime(roots[i])) * (1/r_prime(roots[j])) * (1/(roots[i]-roots[j])**2) 


eig_values, eig_vectors = eig(H)
np.save("eigenVectors_N{}_alpha{}_rmax{}.npy".format(gl.N, gl.alpha, gl.r_max), eig_vectors)
np.save("eigenValues_N{}_alpha{}_rmax{}.npy".format(gl.N, gl.alpha, gl.r_max), eig_values)

