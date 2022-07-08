import numpy as np
from matplotlib import pyplot as plt

from scipy.special import roots_legendre, legendre, eval_legendre, lpmv
from astropy.coordinates import cartesian_to_spherical

import gl

eigenenergies_memory = np.load(".npy") # shall be 2D array shape (gl.N_rs - 1, gl.N_thetas)
eigenvectors_memory = np.load(".npy") # shall be 3D array shape (gl.N_rs - 1, gl.N_rs - 1, gl.N_thetas)

# Gauss - Lobatto
# ----------------------
poly_Nr = legendre(gl.N_rs)
poly_Nr_der = poly_Nr.deriv()
a =  np.roots(poly_Nr_der)
roots = a[0] # shape (gl.N_rs - 1, )


# Gauss - Legendre
# ----------------------
b =  roots_legendre(gl.N_thetas) # returned are the cosine-of-the-angle values, not the theta-angles themselves!
roots_here, weights_here = b[0], b[1]



def r_prime(x):
    return gl.L * (2 + gl.alpha) / (1 - x + gl.alpha)**2


# -------------------
Psi = np.zeros( (gl.N_rs - 1, gl.N_thetas, gl.N_phis, gl.N_timesteps) , dtype=np.complex128)
Psi[:, :, :, 0] = np.load("Initial_PSI_n1_l0_m0_r_0to200.0_theta_0to3.141592653589793_phi_0to6.283185307179586_3Dgrid.npy")


def main():
    for timestep in range(gl.N_timesteps):

    # 1) Obtain the spectral coefficients C_{ilm} at the current timestep t
    # ---------------------------------------------------------------------
        Cilmoft = np.zeros( (gl.N_rs - 1,  gl.N_thetas,  2*gl.N_thetas,  2), dtype=np.complex128) 
        # first index: i,  second index: l,  third index: m,  fourth index: time dependence

        for i in range(Cilmoft.shape[0]):
            for l in range(Cilmoft.shape[1]):
                for m in range(-l, l+1, 1):
                    Cilmoft[i, l, m+l, 0] = ThreeDim_quadrature(Psi[:, :, :, timestep], i, l, m)

    # 2) Propagate for half a timestep in the field-free Hamiltonian H_0^{l} for the partial wave numbered l: multiply C_{ilm}(t) by a phase
        for i in range(gl.N_rs - 1):
            for l in range(gl.N_thetas):
                for m in range(-l, l+1, 1):
                    Cilmoft[i, l, m+l, 1] = np.exp(-1j * eigenenergies_memory[i, l] * gl.delta_t/2)  *   Cilmoft[i, l, m+l, 0]

    # 3) Start work for the transformation to obtain the grid representation of Psi
    # a) i -> r
        Psi_lm = np.zeros( (gl.N_thetas,  2*gl.N_thetas+1,  gl.N_rs - 1), dtype=np.complex128 )  
        # first index: l, second index: m, third index: the r-dependence Psi_{lm}(r)
        for l in range(gl.N_thetas):
            for m in range(-l, l+1, 1):
                for j in range(gl.N_rs - 1):
                    Psi_lm[l, m, j] = np.sum(  Cilmoft[:, l, m, 1] * (eigenvectors_memory[j, :, l] * eval_legendre(gl.N_rs, roots[j]) / r_prime(roots[j]))   )
    
    # b) l -> theta
        Psi_m = np.zeros( (2*gl.N_thetas+1,  gl.N_rs - 1,  gl.N_thetas) , dtype=np.complex128 )
        for m in range(-gl.N_thetas,  gl.N_thetas + 1,  1):
            for j in range(gl.N_rs - 1):
                for k in range(gl.N_thetas):
                    norma = np.array( [ np.sqrt( (2*l+1) * np.math.factorial(l-m) / (4*np.pi * np.math.factorial(l+m)) ) for l in range(gl.N_thetas) ] ) # a 1D array shape (gl.N_thetas, )
                    lpmvs = np.array( [ lpmv(m, l, roots_here[k]) for l in range(gl.N_thetas) ] ) # a 1D array shape (gl.N_thetas, )
                    Psi_m[m + gl.N_thetas,  j,  k] = np.sum(Psi_lm[:,  m + gl.N_thetas,  j] * norma * lpmvs)

    # c) m -> phi
        Psi = np.zeros( (gl.N_rs - 1,  gl.N_thetas,  gl.N_phis,  2), dtype=np.complex128)
        for j in range(gl.N_rs -  1):
            for k in range(gl.N_thetas):
                for n in range(gl.N_phis):
                    Psi[j, k, n, 0] = np.sum(  Psi_m[:, j, k] * np.exp(1j * np.arange(-gl.N_thetas, gl.N_thetas+1) * np.linspace(0.0, 2*np.pi, gl.N_phis)[n])  ) 
                    # np.exp(1j * np.arange(-gl.L_max, gl.L_max) * np.linspace(0.0, 2*np.pi, gl.N_phis)[n]) as a whole has shape (2*gl.L_max, )

    # 4) Propagate for 1 timestep in the laser field (its operator being diagonal in the coordinate representation)
        Psi[:, :, :, 1] = propagation_in_laser_field(Psi[:, :, :, 0], timestep)


def ThreeDim_quadrature(Psi, i, l, m):
    # Psi is a numpy 3 dimensional array of size (gl.N_rs - 1, gl.N_thetas + 1, gl.N_phis)
    # 1) Integration wrt phi variable:
    F1 = phi_Trapez_Quad(Psi, m) # F1 is 2 dimensional
    
    # 2) Integration wrt theta variable:
    # {cos(theta_k)} are the L+1 zeros of P_{L+1}(cos(theta_k)) where L = gl.N_thetas (i.e. the maximum partial wave number)
    F2 = theta_Gauss_Quad(F1, l, m) # F2 is 1 dimensional, F1 is 2 dimensional

    # 3) Integration wrt r variable (actually x):
    return  x_Lobatto_Quad(F2, i, l)


def phi_Trapez_Quad(Psi, m):
    # Psi from arguments is a 3 dimensional array
    exp_of_phis = np.exp(-1j * m * gl.phis)

    for contor in range(Psi.shape[2]):
        Psi[:, :, contor] = Psi[:, :, contor] * exp_of_phis[contor]

    return np.trapz(Psi, axis=-1)


def theta_Gauss_Quad(F1, l, m):
    # F1 from arguments is 2 dimensional

    poly_results = eval_legendre(l, roots_here) # [P_l( cos(theta) ) for cos(theta) in roots_here]
    containerr = np.zeros( (gl.N_rs - 1, ), dtype=np.complex128)

    for idx in range(gl.N_rs - 1):
        containerr[idx] = np.sum(weights_here * poly_results * F1[idx, :])
    return containerr

def x_Lobatto_Quad(F2, i, l):
    # F2 from arguments is 1 dimensional

    # lobatto quadrature preliminaries
    poly_Nr = legendre(gl.N_rs)
    poly_Nr_der = poly_Nr.deriv()
    a =  np.roots(poly_Nr_der)
    lobatto_nodes = a[0] # shape (gl.N_rs - 1, )

    lobatto_weights = 2 / ( (gl.N_rs + 1) * (gl.N_rs + 2) * eval_legendre(gl.N_rs + 1, lobatto_nodes)**2 )
    
    # actual numerical integration
    return np.sum(eigenvectors_memory[:, i, l] * F2 * lobatto_weights) # shall return a 0-dimensional complex128


def propagation_in_laser_field(Psi_grid_repr, timestep):
    # timestep from arguments is current_timestep
    Eprime_field_cartesian = Eprime_field(timestep + 0.5) # numpy array of shape (3, )
    a, b, c = cartesian_to_spherical(Eprime_field_cartesian[0], Eprime_field_cartesian[1], Eprime_field_cartesian[2])
    Eprime_field_spherical = np.array( [a, b, c] )

    for j in range(gl.N):
        for k in range(gl.N_thetas):
            for n in range(gl.N_phis):

                r = roots[j]
                theta = k * gl.delta_theta
                phi = n * gl.delta_phi
                vec_r = np.array( [r, theta, phi] )

                V = (-vec_r * Eprime_field_spherical)

                Psi_grid_repr[j, k, n]  =  Psi_grid_repr[j, k, n] * np.exp(-1j * V * gl.delta_t)
    return Psi_grid_repr



def Eprime_field(timestep_halfadded):
    # timestep_halfadded from arguments is current_timestep + 0.5
    container = A0oft(timestep_halfadded, gl.Temp_Env)
    Aoft = container[0]
    Aoft_der = container[1]
    E_prime_x = Aoft**2 * np.sin(2 * gl.omega * timestep_halfadded*gl.delta_time ) * gl.omega / (2*gl.c_au)
    E_prime_z = -gl.omega * Aoft*np.cos(gl.omega* timestep_halfadded*gl.delta_time) - Aoft_der * np.sin(gl.omega * timestep_halfadded*gl.delta_time)
    return np.array([E_prime_x, 0.0, E_prime_z])

def A0oft(timestep, Temporal_Envelope):
    if Temporal_Envelope == "sin_squared":
        T = gl.N * 2*np.pi / gl.omega
        Actual_A = np.sin(np.pi * timestep*gl.delta_time / T) ** 2
        Derivative_of_A = np.sin(2 * np.pi * timestep*gl.delta_time / T) * (np.pi/T)
        return np.array([Actual_A, Derivative_of_A])
    else:
        pass

