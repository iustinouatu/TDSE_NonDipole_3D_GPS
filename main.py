import numpy as np
from numba import jit, njit
import math

from matplotlib import pyplot as plt

from scipy.special import roots_legendre, legendre, eval_legendre, lpmv
from astropy.coordinates import cartesian_to_spherical
from astropy.units.equivalencies import dimensionless_angles

import gl

eigenenergies_memory = np.load("eigenValues_N_rs100_alpha25.0_rmax200.0.npy") # shall be 2D array shape (gl.N_rs - 1,   gl.N_thetas)
eigenvectors_memory = np.load("eigenVectors_N_rs100_alpha25.0_rmax200.0.npy") # shall be 3D array shape  (gl.N_rs - 1,   gl.N_rs - 1,   gl.N_thetas), first index is the r-dep, second index is the i-index, third index is the l-index
# so eigenvectors_memory[:, K, L] is the eigenvector corresponding to the eigenvalue eigenenergies_memory[K, L], i.e. the K-th eigenvalue for the partial wave-number L

# Gauss - Lobatto
# ----------------------
# poly_Nr = legendre(gl.N_rs) # poly_Nr will be full of NaN's if N_rs is big.
# poly_Nr_der = poly_Nr.deriv()
# a =  np.roots(poly_Nr_der) # does not work for large order if poly_Nr_der because np.roots() solves an eigenvalue problem on a matrix and that fails in my case
# roots = a # shape (gl.N_rs - 1, )
# lobatto_nodes = roots
# Do as below:
def gLLNodesAndWeights (n, epsilon = 1e-15):

    x = np.empty (n)
    w = np.empty (n)
    
    x[0] = -1 
    x[n - 1] = 1
    w[0] = 2.0 / ((n * (n - 1))) 
    w[n - 1] = w[0]
    
    n_2 = n // 2
    
    for i in range (1, n_2):
        xi = (1 - (3 * (n - 2)) / (8 * (n - 1) ** 3)) *\
           np.cos ((4 * i + 1) * np.pi / (4 * (n - 1) + 1))
      
        error = 1.0
      
        while error > epsilon:
            y  =  dLgP (n - 1, xi)
            y1 = d2LgP (n - 1, xi)
            y2 = d3LgP (n - 1, xi)
        
            dx = 2 * y * y1 / (2 * y1 ** 2 - y * y2)
        
            xi -= dx
            error = abs (dx)
      
        x[i] = -xi
        x[n - i - 1] =  xi
      
        w[i] = 2 / (n * (n - 1) * lgP (n - 1, x[i]) ** 2)
        w[n - i - 1] = w[i]

    if n % 2 != 0:
        x[n_2] = 0
        w[n_2] = 2.0 / ((n * (n - 1)) * lgP (n - 1, np.array (x[n_2])) ** 2)

    return x, w

def dLgP (n, xi):
  """
  Evaluates the first derivative of P_{n}(xi)
  """
  return n * (lgP (n - 1, xi) - xi * lgP (n, xi))\
           / (1 - xi ** 2)

def d2LgP (n, xi):
  """
  Evaluates the second derivative of P_{n}(xi)
  """
  return (2 * xi * dLgP (n, xi) - n * (n + 1)\
                                    * lgP (n, xi)) / (1 - xi ** 2)

def d3LgP (n, xi):
  """
  Evaluates the third derivative of P_{n}(xi)
  """
  return (4 * xi * d2LgP (n, xi)\
                 - (n * (n + 1) - 2) * dLgP (n, xi)) / (1 - xi ** 2)

def lgP (n, xi):
  """
  Evaluates P_{n}(xi) using an iterative algorithm
  """
  if n == 0:
    return np.ones (xi.size)
  
  elif n == 1:
    return xi

  else:
    fP = np.ones (xi.size); sP = xi.copy (); nP = np.empty (xi.size)
    for i in range (2, n + 1):

      nP = ((2 * i - 1) * xi * sP - (i - 1) * fP) / i
      fP = sP; sP = nP

    return nP


a = gLLNodesAndWeights(gl.N_rs + 1)[0]
roots = a[1:-1] # has shape (gl.N_rs - 1) if input to gLLNodesAndWeights is equal to (gl.N_rs + 1)
lobatto_nodes = roots



# Gauss - Legendre
# ----------------------
b =  roots_legendre(gl.N_thetas) # returned are the cosine-of-the-angle values, not the theta-angles themselves!
# also, l_{max} = gl.N_thetas - 1, so need the roots {cos(theta_k)} with k = 0, 1, 2, ..., gl.N_thetas - 1, of the Legendre polynomial of order n = gl.N_thetas
roots_here, weights_here = np.float64(b[0]), np.float64(b[1])


# Non-Linear Mapping 
# ------------------------
@njit
def r(x):
    return gl.L * (1 + x) / (1 - x + gl.alpha)

@njit
def r_prime(x):
    return gl.L * (2 + gl.alpha) / (1 - x + gl.alpha)**2


# START
# -------------------
@njit
def get_Cilmoft(Psi):
    Cilmoft = np.zeros( (gl.N_rs - 1,  gl.N_thetas,  2*gl.N_thetas-1,  2), dtype=np.complex128) 
    for i in range(Cilmoft.shape[0]):
        for l in range(Cilmoft.shape[1]):
            for m in range(-l, l+1, 1):
                Cilmoft[i, l, m+l, 0] = ThreeDim_quadrature(Psi, i, l, m)
                
    return Cilmoft

@njit
def propagate_in_fieldfree_H(Cilmoft):
    for i in range(Cilmoft.shape[0]):
        for l in range(Cilmoft.shape[1]):
            for m in range(-l, l+1, 1):
                Cilmoft[i, l, m+l, 1] = np.exp(-1j * eigenenergies_memory[i, l] * gl.delta_t/2)  *   Cilmoft[i, l, m+l, 0]   
    return Cilmoft


legendre_results_for_Psi_lm = np.zeros((roots.shape[0], ), dtype=np.complex128)
for j in range(roots.shape[0]):
    legendre_results_for_Psi_lm[j] = eval_legendre(gl.N_rs, roots[j])
@njit
def eval_legendre_for_Psi_lm(j):
    return legendre_results_for_Psi_lm[j]

containeeeer = np.zeros( (2*gl.N_thetas-2, gl.N_thetas,  gl.N_thetas ) , dtype=np.complex128) # will hold lpmv() function's results
for m in range(-gl.N_thetas+1,  gl.N_thetas,  1):
    for k in range(gl.N_thetas):
        for l in range(np.abs(m), gl.N_thetas):
            containeeeer[m, k, l] = lpmv(m, l, roots_here[k])
@njit
def lpmv_withNJIT(m, k, l):
    return containeeeer[m, k, l]

@njit
def get_PsiGrid_from_Cilmoft_part1(Cilmoft):
    # a) i -> r
    Psi_lm = np.zeros( (gl.N_thetas,   2*gl.N_thetas - 1,   gl.N_rs - 1), dtype=np.complex128 )  
    # first index: l, second index: m, third index: the r-dependence Psi_{lm}(r)
    for l in range(Cilmoft.shape[1]):
        for m in range(-l, l+1, 1):
            for j in range(Cilmoft.shape[0]):
                Psi_lm[l, m+l, j] = np.sum(  Cilmoft[:, l, m+l, 1] * (eigenvectors_memory[j, :, l] * eval_legendre_for_Psi_lm(j) / r_prime(roots[j]))   )  
                # multiplicative paranthesis comes from the conversion from A (in memory, obtained from C++, see Revisiting ... 2020) to the actual wavefunction value on grid     
    # b) l -> theta
    Psi_m = np.zeros( (2*gl.N_thetas - 1,    gl.N_rs - 1,    gl.N_thetas) , dtype=np.complex128 )
    for m in range(-gl.N_thetas+1,  gl.N_thetas,  1):
        for j in range(Cilmoft.shape[0]):
            for k in range(Cilmoft.shape[1]):
                normaliz = np.array(  [ np.sqrt( (2*l+1) * math.gamma(l-m) / (4*np.pi * math.gamma(l+m)) ) for l in range(np.abs(m), Cilmoft.shape[1]) ]  ) # a 1D array shape (Cilmoft.shape[1] - |m|,  ), i.e. shape (gl.N_thetas - |m|,  )
                lpmvs = np.array(  [ lpmv_withNJIT(m, k, l) for l in range(np.abs(m), Cilmoft.shape[1]) ]  ) # a 1D array shape (Cilmoft.shape[1] - |m|,  ) i.e. of shape (gl.N_thetas - |m|,  )
                Psi_m[m + (gl.N_thetas-1),  j,  k] = np.sum(Psi_lm[np.abs(m):,  m + (gl.N_thetas-1),  j] * normaliz * lpmvs)
    # c) m -> phi
    Psi_grid = np.zeros( (gl.N_rs - 1,  gl.N_thetas,  gl.N_phis,  2), dtype=np.complex128)  ###########################################################################################################################################################################
    phi = np.linspace(0.0, 2*np.pi, gl.N_phis)
    for j in range(gl.N_rs -  1):
        for k in range(gl.N_thetas):
            for n in range(gl.N_phis):
                Psi_grid[j, k, n, 0] = np.sum(  Psi_m[:, j, k] * np.exp(1j * np.arange(-gl.N_thetas+1, gl.N_thetas) * phi[n])  ) 
                    # np.exp(1j * np.arange(-gl.L_max, gl.L_max) * np.linspace(0.0, 2*np.pi, gl.N_phis)[n]) as a whole has shape (2*gl.L_max, ) has shape ()
    return Psi_grid

@njit
def get_PsiGrid_from_Cilmoft_part2(Cilmoft):
    # a) i -> r
    Psi_lm = np.zeros( (gl.N_thetas,   2*gl.N_thetas - 1,   gl.N_rs - 1), dtype=np.complex128 )  
    # first index: l, second index: m, third index: the r-dependence Psi_{lm}(r)
    for l in range(Cilmoft.shape[1]):
        for m in range(-l, l+1, 1):
            for j in range(Cilmoft.shape[0]):
                Psi_lm[l, m+l, j] = np.sum(  Cilmoft[:, l, m+l, 1] * (eigenvectors_memory[j, :, l] * eval_legendre_for_Psi_lm(j) / r_prime(roots[j]))   )  
                    # multiplicative paranthesis comes from the conversion from A (in memory, obtained from C++, see Revisiting ... 2020) to the actual wavefunction value on grid     
    # b) l -> theta
    Psi_m = np.zeros( (2*gl.N_thetas - 1,    gl.N_rs - 1,    gl.N_thetas) , dtype=np.complex128 )
    for m in range(-gl.N_thetas+1,  gl.N_thetas,  1):
        for j in range(Cilmoft.shape[0]):
            for k in range(Cilmoft.shape[1]):
                normaliz = np.array(  [ np.sqrt( (2*l+1) * math.gamma(l-m) / (4*np.pi * math.gamma(l+m)) ) for l in range(np.abs(m), Cilmoft.shape[1]) ]  ) # a 1D array shape (Cilmoft.shape[1] - |m|,  ), i.e. shape (gl.N_thetas - |m|,  )
                lpmvs = np.array(  [ lpmv_withNJIT(m, k, l) for l in range(np.abs(m), Cilmoft.shape[1]) ]  ) # a 1D array shape (Cilmoft.shape[1] - |m|,  ) i.e. of shape (gl.N_thetas - |m|,  )
                Psi_m[m + (gl.N_thetas-1),  j,  k] = np.sum(Psi_lm[np.abs(m):,  m + (gl.N_thetas-1),  j] * normaliz * lpmvs)
    # c) m -> phi
    Psi_grid = np.zeros( (gl.N_rs - 1,  gl.N_thetas,  gl.N_phis), dtype=np.complex128)  ###########################################################################################################################################################################
    phi = np.linspace(0.0, 2*np.pi, gl.N_phis)
    for j in range(gl.N_rs -  1):
        for k in range(gl.N_thetas):
            for n in range(gl.N_phis):
                Psi_grid[j, k, n] = np.sum(  Psi_m[:, j, k] * np.exp(1j * np.arange(-gl.N_thetas+1, gl.N_thetas) * phi[n])  ) 
                    # np.exp(1j * np.arange(-gl.L_max, gl.L_max) * np.linspace(0.0, 2*np.pi, gl.N_phis)[n]) as a whole has shape (2*gl.L_max, ) has shape ()
    return Psi_grid

@njit
def cartesian_to_spherical_withNJIT(Ex, Ey, Ez):
    r = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    theta = np.arccos(Ez / r)
    phi = np.arctan2(Ey, Ex)
    return r, theta, phi

@njit
def propagation_in_laser_field(Psi_grid_repr, timestep):
    # timestep from arguments is current_timestep
    # Psi_grid_repr from arguments is shape (gl.N_rs - 1,  gl.N_thetas,  gl.N_phis)

    Eprime_field_cartesian = Eprime_field(timestep + 0.5) # numpy array of shape (3, )
    # a, b, c = cartesian_to_spherical(Eprime_field_cartesian[0], Eprime_field_cartesian[1], Eprime_field_cartesian[2]) # astropy's cartesian_to_spherical doesn't work with numba
    # Eprime_field_spherical = np.array( [a, b.value, c.value], dtype=np.complex128 ) # astropy doesn't work with numba
    a, b, c = cartesian_to_spherical_withNJIT(Eprime_field_cartesian[0], Eprime_field_cartesian[1], Eprime_field_cartesian[2])
    Eprime_field_spherical = np.array( [a, b, c], dtype=np.complex128 )

    for j in range(gl.N_rs - 1):
        for k in range(gl.N_thetas):
            for n in range(gl.N_phis):

                x = roots[j]  # roots is an array shape (gl.N_rs - 1, ) containing the gl.N_rs - 1 roots of the derivative of the Legendre poly of order gl.N_rs of x (poly which after derivative becomes of order gl.N_rs - 1) 
                r_coord = r(x)
                theta = k * gl.delta_theta
                phi = n * gl.delta_phi
                vec_r = np.array( [r_coord, theta, phi], dtype=np.complex128 )

                V = np.dot(-vec_r, Eprime_field_spherical)

                Psi_grid_repr[j, k, n]  =  Psi_grid_repr[j, k, n] * np.exp(-1j * V * gl.delta_t)
    return Psi_grid_repr   


def main():
    Psi = np.zeros( (gl.N_rs - 1,  gl.N_thetas,  gl.N_phis,  gl.N_timesteps) , dtype=np.complex128)
    Psi[:, :, :, 0] = np.load("Initial_PSI_n1_l0_m0_r_0to200.0_theta_0to3.141592653589793_phi_0to6.283185307179586_3Dgrid.npy")

    for timestep in range(gl.N_timesteps):
        print("We are at timestep number {} out of a total of {} timesteps.".format(timestep+1, gl.N_timesteps))
        Cilmoft = get_Cilmoft(Psi[:, :, :, timestep])
        Cilmoft = propagate_in_fieldfree_H(Cilmoft)
        Psi_grid = get_PsiGrid_from_Cilmoft_part1(Cilmoft)
        Psi_grid[:, :, :, 1] = propagation_in_laser_field(Psi_grid[:, :, :, 0], timestep)
        Cilmoft = get_Cilmoft(Psi_grid[:, :, :, 1])
        Cilmoft = propagate_in_fieldfree_H(Cilmoft)
        Psi_grid = get_PsiGrid_from_Cilmoft_part2(Cilmoft)
        Psi[:, :, :, timestep+1] = Psi_grid

        # Cilmoft = np.zeros( (gl.N_rs - 1,  gl.N_thetas,  2*gl.N_thetas-1,  2), dtype=np.complex128) 
        # # first index: i,  second index: l,  third index: m,  fourth index: time dependence
        # for i in range(Cilmoft.shape[0]):
        #    for l in range(Cilmoft.shape[1]):
        #        for m in range(-l, l+1, 1):
        #            Cilmoft[i, l, m+l, 0] = ThreeDim_quadrature(Psi[:, :, :, timestep], i, l, m)

#        for i in range(Cilmoft.shape[0]):
#            for l in range(Cilmoft.shape[1]):
#                for m in range(-l, l+1, 1):
#                    Cilmoft[i, l, m+l, 1] = np.exp(-1j * eigenenergies_memory[i, l] * gl.delta_t/2)  *   Cilmoft[i, l, m+l, 0]

        # # a) i -> r
        # Psi_lm = np.zeros( (gl.N_thetas,   2*gl.N_thetas - 1,   gl.N_rs - 1), dtype=np.complex128 )  
        # # first index: l, second index: m, third index: the r-dependence Psi_{lm}(r)
        # for l in range(Cilmoft.shape[1]):
        #     for m in range(-l, l+1, 1):
        #         for j in range(Cilmoft.shape[0]):
        #             Psi_lm[l, m+l, j] = np.sum(  Cilmoft[:, l, m+l, 1] * (eigenvectors_memory[j, :, l] * eval_legendre(gl.N_rs, roots[j]) / r_prime(roots[j]))   )  
        #                 # multiplicative paranthesis comes from the conversion from A (in memory, obtained from C++, see Revisiting ... 2020) to the actual wavefunction value on grid     
        # # b) l -> theta
        # Psi_m = np.zeros( (2*gl.N_thetas - 1,    gl.N_rs - 1,    gl.N_thetas) , dtype=np.complex128 )
        # for m in range(-gl.N_thetas+1,  gl.N_thetas,  1):
        #     for j in range(Cilmoft.shape[0]):
        #         for k in range(Cilmoft.shape[1]):
        #             normaliz = np.array(  [ np.sqrt( (2*l+1) * np.math.factorial(l-m) / (4*np.pi * np.math.factorial(l+m)) ) for l in range(np.abs(m), Cilmoft.shape[1]) ]  ) # a 1D array shape (Cilmoft.shape[1] - |m|,  ), i.e. shape (gl.N_thetas - |m|,  )
        #             lpmvs = np.array(  [ lpmv(m, l, roots_here[k]) for l in range(np.abs(m), Cilmoft.shape[1]) ]  ) # a 1D array shape (Cilmoft.shape[1] - |m|,  ) i.e. of shape (gl.N_thetas - |m|,  )
        #             Psi_m[m + (gl.N_thetas-1),  j,  k] = np.sum(Psi_lm[np.abs(m):,  m + (gl.N_thetas-1),  j] * normaliz * lpmvs)
        # # c) m -> phi
        # Psi_grid = np.zeros( (gl.N_rs - 1,  gl.N_thetas,  gl.N_phis,  2), dtype=np.complex128)  ###########################################################################################################################################################################
        # phi = np.linspace(0.0, 2*np.pi, gl.N_phis)
        # for j in range(gl.N_rs -  1):
        #     for k in range(gl.N_thetas):
        #         for n in range(gl.N_phis):
        #             Psi_grid[j, k, n, 0] = np.sum(  Psi_m[:, j, k] * np.exp(1j * np.arange(-gl.N_thetas+1, gl.N_thetas) * phi[n])  ) 
        #                 # np.exp(1j * np.arange(-gl.L_max, gl.L_max) * np.linspace(0.0, 2*np.pi, gl.N_phis)[n]) as a whole has shape (2*gl.L_max, ) has shape ()

        # # 4) Propagate for 1 timestep in the laser field (its operator being diagonal in the coordinate representation)
        # Psi_grid[:, :, :, 1] = propagation_in_laser_field(Psi_grid[:, :, :, 0], timestep)
        
        # Cilmoft = np.zeros( (gl.N_rs - 1,  gl.N_thetas,  2*gl.N_thetas-1,  2), dtype=np.complex128) 
        # # first index: i,  second index: l,  third index: m,  fourth index: time dependence
        # for i in range(Cilmoft.shape[0]):
        #     for l in range(Cilmoft.shape[1]):
        #         for m in range(-l, l+1, 1):
        #             Cilmoft[i, l, m+l, 0] = ThreeDim_quadrature(Psi_grid[:, :, :, 1], i, l, m)

        # for i in range(Cilmoft.shape[0]):
        #     for l in range(Cilmoft.shape[1]):
        #         for m in range(-l, l+1, 1):
        #             Cilmoft[i, l, m+l, 1] = np.exp(-1j * eigenenergies_memory[i, l] * gl.delta_t/2)  *   Cilmoft[i, l, m+l, 0]

        # # a) i -> r
        # Psi_lm = np.zeros( (gl.N_thetas,   2*gl.N_thetas - 1,   gl.N_rs - 1), dtype=np.complex128 )  
        # # first index: l, second index: m, third index: the r-dependence Psi_{lm}(r)
        # for l in range(Cilmoft.shape[1]):
        #     for m in range(-l, l+1, 1):
        #         for j in range(Cilmoft.shape[0]):
        #             Psi_lm[l, m+l, j] = np.sum(  Cilmoft[:, l, m+l, 1] * (eigenvectors_memory[j, :, l] * eval_legendre(gl.N_rs, roots[j]) / r_prime(roots[j]))   )  
        #                 # multiplicative paranthesis comes from the conversion from A (in memory, obtained from C++, see Revisiting ... 2020) to the actual wavefunction value on grid     
        # # b) l -> theta
        # Psi_m = np.zeros( (2*gl.N_thetas - 1,    gl.N_rs - 1,    gl.N_thetas) , dtype=np.complex128 )
        # for m in range(-gl.N_thetas+1,  gl.N_thetas,  1):
        #     for j in range(Cilmoft.shape[0]):
        #         for k in range(Cilmoft.shape[1]):
        #             normaliz = np.array(  [ np.sqrt( (2*l+1) * np.math.factorial(l-m) / (4*np.pi * np.math.factorial(l+m)) ) for l in range(np.abs(m), Cilmoft.shape[1]) ]  ) # a 1D array shape (Cilmoft.shape[1] - |m|,  ), i.e. shape (gl.N_thetas - |m|,  )
        #             lpmvs = np.array(  [ lpmv(m, l, roots_here[k]) for l in range(np.abs(m), Cilmoft.shape[1]) ]  ) # a 1D array shape (Cilmoft.shape[1] - |m|,  ) i.e. of shape (gl.N_thetas - |m|,  )
        #             Psi_m[m + (gl.N_thetas-1),  j,  k] = np.sum(Psi_lm[np.abs(m):,  m + (gl.N_thetas-1),  j] * normaliz * lpmvs)
        # # c) m -> phi
        # Psi_grid = np.zeros( (gl.N_rs - 1,  gl.N_thetas,  gl.N_phis), dtype=np.complex128)  ###########################################################################################################################################################################
        # phi = np.linspace(0.0, 2*np.pi, gl.N_phis)
        # for j in range(gl.N_rs -  1):
        #     for k in range(gl.N_thetas):
        #         for n in range(gl.N_phis):
        #             Psi_grid[j, k, n] = np.sum(  Psi_m[:, j, k] * np.exp(1j * np.arange(-gl.N_thetas+1, gl.N_thetas) * phi[n])  ) 
        #                 # np.exp(1j * np.arange(-gl.L_max, gl.L_max) * np.linspace(0.0, 2*np.pi, gl.N_phis)[n]) as a whole has shape (2*gl.L_max, ) has shape ()

        # Psi[:, :, :, timestep+1] = Psi_grid
# END t = 0
#############################################################################



    # for timestep in range(gl.N_timesteps):
    #     print("We are at timestep {} out of a total of {} timesteps.".format(timestep+1, gl.N_timesteps))


# # Part 1
#     # 1) Obtain the spectral coefficients C_{ilm} at the current timestep t
#     # ---------------------------------------------------------------------
#         Cilmoft = np.zeros( (gl.N_rs - 1,  gl.N_thetas,  2*gl.N_thetas-1,  2), dtype=np.complex128) 
#         # first index: i,  second index: l,  third index: m,  fourth index: time dependence

#         for i in range(Cilmoft.shape[0]):
#             for l in range(Cilmoft.shape[1]):
#                 for m in range(-l, l+1, 1):
#                     Cilmoft[i, l, m+l, 0] = ThreeDim_quadrature(Psi[:, :, :, timestep], i, l, m)

#     # 2) Propagate for half a timestep in the field-free Hamiltonian H_0^{l}: multiply C_{ilm}(t) by a phase
#         for i in range(Cilmoft.shape[0]):
#             for l in range(Cilmoft.shape[1]):
#                 for m in range(-l, l+1, 1):
#                     Cilmoft[i, l, m+l, 1] = np.exp(-1j * eigenenergies_memory[i, l] * gl.delta_t/2)  *   Cilmoft[i, l, m+l, 0]

#     # 3) Start work for the transformation to obtain the grid representation of Psi
#     # a) i -> r
#         Psi_lm = np.zeros( (gl.N_thetas,   2*gl.N_thetas - 1,   gl.N_rs - 1), dtype=np.complex128 )  
#         # first index: l, second index: m, third index: the r-dependence Psi_{lm}(r)
#         for l in range(Cilmoft.shape[1]):
#             for m in range(-l, l+1, 1):
#                 for j in range(Cilmoft.shape[0]):
#                     Psi_lm[l, m+l, j] = np.sum(  Cilmoft[:, l, m+l, 1] * (eigenvectors_memory[j, :, l] * eval_legendre(gl.N_rs, roots[j]) / r_prime(roots[j]))   )  
#                     # multiplicative paranthesis comes from the conversion from A (in memory, obtained from C++, see Revisiting ... 2020) to the actual wavefunction value on grid 

#     # b) l -> theta
#         Psi_m = np.zeros( (2*gl.N_thetas - 1,    gl.N_rs - 1,    gl.N_thetas) , dtype=np.complex128 )
#         for m in range(-gl.N_thetas+1,  gl.N_thetas,  1):
#             for j in range(Cilmoft.shape[0]):
#                 for k in range(Cilmoft.shape[1]):
#                     normaliz = np.array(  [ np.sqrt( (2*l+1) * np.math.factorial(l-m) / (4*np.pi * np.math.factorial(l+m)) ) for l in range(np.abs(m), Cilmoft.shape[1]) ]  ) # a 1D array shape (Cilmoft.shape[1] - |m|,  ), i.e. shape (gl.N_thetas - |m|,  )
#                     lpmvs = np.array(  [ lpmv(m, l, roots_here[k]) for l in range(np.abs(m), Cilmoft.shape[1]) ]  ) # a 1D array shape (Cilmoft.shape[1] - |m|,  ) i.e. of shape (gl.N_thetas - |m|,  )
#                     Psi_m[m + (gl.N_thetas-1),  j,  k] = np.sum(Psi_lm[np.abs(m):,  m + (gl.N_thetas-1),  j] * normaliz * lpmvs)

#     # c) m -> phi
#         Psi_grid = np.zeros( (gl.N_rs - 1,  gl.N_thetas,  gl.N_phis,  2), dtype=np.complex128)  ###########################################################################################################################################################################
#         phi = np.linspace(0.0, 2*np.pi, gl.N_phis)
#         for j in range(gl.N_rs -  1):
#             for k in range(gl.N_thetas):
#                 for n in range(gl.N_phis):
#                     Psi_grid[j, k, n, 0] = np.sum(  Psi_m[:, j, k] * np.exp(1j * np.arange(-gl.N_thetas+1, gl.N_thetas) * phi[n])  ) 
#                     # np.exp(1j * np.arange(-gl.L_max, gl.L_max) * np.linspace(0.0, 2*np.pi, gl.N_phis)[n]) as a whole has shape (2*gl.L_max, ) has shape ()

#     # 4) Propagate for 1 timestep in the laser field (its operator being diagonal in the coordinate representation)
#         Psi_grid[:, :, :, 1] = propagation_in_laser_field(Psi_grid[:, :, :, 0], timestep)

# # Part 2
#     # 5) Transfer from real space to energy representation again (i.e. find C_ilm(t))
#         Cilmoft = np.zeros( (gl.N_rs - 1,  gl.N_thetas,  2*gl.N_thetas-1,  2), dtype=np.complex128) 
#         # first index: i,  second index: l,  third index: m,  fourth index: time dependence
#         for i in range(Cilmoft.shape[0]):
#             for l in range(Cilmoft.shape[1]):
#                 for m in range(-l, l+1, 1):
#                     Cilmoft[i, l, m+l, 0] = ThreeDim_quadrature(Psi_grid[:, :, :, 1], i, l, m)
#     # 6) Propagate for the remaining half of a timestep in the field-free hamiltonian H_0^{l} 
#         for i in range(Cilmoft.shape[0]):
#             for l in range(Cilmoft.shape[1]):
#                 for m in range(-l, l+1, 1):
#                     Cilmoft[i, l, m+l, 1] = np.exp(-1j * eigenenergies_memory[i, l] * gl.delta_t/2)  *   Cilmoft[i, l, m+l, 0] 

legendre_results = np.zeros( (gl.N_thetas, roots_here.shape[0]) , dtype=np.complex128)
for l in range(gl.N_thetas):
    legendre_results[l, :] = eval_legendre(l, roots_here)
@njit
def eval_legendre_withNJIT(l):
    return legendre_results[l, :]


@njit
def ThreeDim_quadrature(Psi, i, l, m):
    # Psi is a numpy 3 dimensional array of size (gl.N_rs - 1, gl.N_thetas + 1, gl.N_phis)
    # 1) Integration wrt phi variable:
    F1 = phi_Trapez_Quad(Psi, m) # F1 is 2 dimensional
    
    # 2) Integration wrt theta variable:
    # {cos(theta_k)} are the L+1 zeros of P_{L+1}(cos(theta_k)) where L = gl.N_thetas (i.e. the maximum partial wave number)
    F2 = theta_Gauss_Quad(F1, l, m) # F2 is 1 dimensional, F1 is 2 dimensional

    # 3) Integration wrt r variable (actually x):
    return  np.complex128(x_Lobatto_Quad(F2, i, l))

@njit
def phi_Trapez_Quad(Psi, m):
    # Psi from arguments is a 3 dimensional array
    exp_of_phis = np.exp(-1j * m * gl.phis)

    for contor in range(Psi.shape[2]):
        Psi[:, :, contor] = Psi[:, :, contor] * exp_of_phis[contor]

    # return np.trapz(Psi, axis=-1)
    res = np.zeros( (Psi.shape[0], Psi.shape[1]), dtype=np.complex128)
    for i in range(Psi.shape[0]):
        for j in range(Psi.shape[1]):
            res[i, j] = np.trapz(Psi[i, j, :])
    return res

@njit
def theta_Gauss_Quad(F1, l, m):
    # F1 from arguments is 2 dimensional
    poly_results = eval_legendre_withNJIT(l) # [P_l( cos(theta) ) for cos(theta) in roots_here]
    containerr = np.zeros( (gl.N_rs - 1, ), dtype=np.complex128)

    for idx in range(gl.N_rs - 1):
        containerr[idx] = np.sum(weights_here * poly_results * F1[idx, :])
    return containerr


lobatto_weights = 2 / ( (gl.N_rs) * (gl.N_rs+1) * eval_legendre(gl.N_rs, lobatto_nodes)**2 )

@njit
def x_Lobatto_Quad(F2, i, l):
    # F2 from arguments is 1 dimensional

    # actual numerical integration
    return np.sum(eigenvectors_memory[:, i, l] * F2 * lobatto_weights) # shall return a 0-dimensional complex128


# def propagation_in_laser_field(Psi_grid_repr, timestep):
#     # timestep from arguments is current_timestep
#     # Psi_grid_repr from arguments is shape (gl.N_rs - 1,  gl.N_thetas,  gl.N_phis)

#     Eprime_field_cartesian = Eprime_field(timestep + 0.5) # numpy array of shape (3, )
#     a, b, c = cartesian_to_spherical(Eprime_field_cartesian[0], Eprime_field_cartesian[1], Eprime_field_cartesian[2])
#     Eprime_field_spherical = np.array( [a, b.value, c.value] )

#     for j in range(gl.N_rs - 1):
#         for k in range(gl.N_thetas):
#             for n in range(gl.N_phis):

#                 x = roots[j]  # roots is an array shape (gl.N_rs - 1, ) containing the gl.N_rs - 1 roots of the derivative of the Legendre poly of order gl.N_rs of x (poly which after derivative becomes of order gl.N_rs - 1) 
#                 r_coord = r(x)
#                 theta = k * gl.delta_theta
#                 phi = n * gl.delta_phi
#                 vec_r = np.array( [r_coord, theta, phi] )

#                 V = np.dot(-vec_r, Eprime_field_spherical)

#                 Psi_grid_repr[j, k, n]  =  Psi_grid_repr[j, k, n] * np.exp(-1j * V * gl.delta_t)
#     return Psi_grid_repr


@njit
def Eprime_field(timestep_halfadded):
    # timestep_halfadded from arguments is current_timestep + 0.5
    container = A0oft(timestep_halfadded, gl.Temp_Env)
    Aoft = container[0]
    Aoft_der = container[1]
    E_prime_x = Aoft**2 * np.sin(2 * gl.omega * timestep_halfadded*gl.delta_t ) * gl.omega / (2*gl.c_au)
    E_prime_z = -gl.omega * Aoft*np.cos(gl.omega* timestep_halfadded*gl.delta_t) - Aoft_der * np.sin(gl.omega * timestep_halfadded*gl.delta_t)
    return np.array([E_prime_x, 0.0, E_prime_z])

@njit
def A0oft(timestep, Temporal_Envelope):
    if Temporal_Envelope == "sin_squared":
        T = gl.N * 2*np.pi / gl.omega
        Actual_A = np.sin(np.pi * timestep*gl.delta_t / T) ** 2
        Derivative_of_A = np.sin(2 * np.pi * timestep*gl.delta_t / T) * (np.pi/T)
        return np.array([Actual_A, Derivative_of_A])
    else:
        pass

if __name__ == "__main__":
    main()