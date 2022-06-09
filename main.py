import numpy as np
from matplotlib import pyplot as plt

from scipy.special import roots_legendre, legendre, eval_legendre
from astropy.coordinates import cartesian_to_spherical

poly_Nplus1 = legendre(gl.N+1)
poly_Nplus1_der = poly_Nplus1.deriv()
a =  np.roots(poly_Nplus1_der)
roots = a[0] # shape (N+1, )

import gl

eigenenergies = np.load(".npy") # shall be 2D array shape (, )
Psi = np.zeros( (gl.N, gl.N_thetas, gl.N_phis, gl.N_timesteps) , dtype=np.complex128)
Psi[:, :, :, 0] = np.load("Initial_PSI_n1_l0_m0_r_0to200.0_theta_0to3.141592653589793_phi_0to6.283185307179586_3Dgrid.npy")

def main():
    for t in range(gl.N_timesteps):

    # 1) Obtain the spectral coefficients C_{ilm} at the current timestep t
    # -------------------------------------------------------------------
    # Actually calculate the products C_{ilm} * R_{il}(r) [5D tensor (i, l, r, m, time)] 
    # because this will be multiplied by a phase in the propagation: no need for the integral across r (only a 2D integral suffices)

        Cilmoft_times_Ril = np.zeros( (gl.N, gl.L_max, gl.N, 2*gl.L_max, 2), dtype=np.complex128) 
        # first index: i,  second index: l,  third index: the radial dependence (from R_{il}(r)),  fourth index: m,  fifth index: time dependence

        for i in range(gl.N):
            for l in range(gl.L_max):
                for m in range(-l, l+1, 1):
                    Cilmoft_times_Ril[i, l, :, m+l, 0] = TwoDim_quadrature(Psi[i, :, :, t], l, m)

    # 2) Propagate for half a timestep in the field-free Hamiltonian H_0^{l} for the partial wave numbered l: multiply C_{ilm} * R_{il}(r) by a phase
        for i in range(gl.N):
            for l in range(gl.L_max):
                for m in range(-l, l+1, 1):
                    Cilmoft_times_Ril[i, l, :, m+l, 1] = np.exp(-1j * eigenenergies[i, l] * gl.delta_t/2) * Cilmoft_times_Ril[i, l, :, m+l, 0]

    # 3) Start work for the transformation to obtain the grid representation of Psi
    # a) i -> r
        Psi_lm = np.zeros( (gl.L_max, 2*gl.L_max, gl.N), dtype=np.complex128 )  # first index: l, second index: m, third index: the r-dependence Psi_{lm}(r), fourth index: time (WHY DO YOU NEED TIME DEPENDENCE IN THIS TENSOR?)
        for l in range(gl.L_max):
            for m in range(-l, l+1, 1):
                for j in range(gl.N):
                    Psi_lm[l, m, j] = np.sum(Cilmoft_times_Ril[:, l, j, m, 1])
    
    # b) l -> theta
        Psi_m = np.zeros( (2*gl.L_max, gl.N, gl.N_thetas) , dtype=np.complex128 )
        for m in range(-gl.L_max, gl.L_max+1, 1):
            for j in range(gl.N):
                for k in range(gl.N_thetas):
                    Psi_m[m+gl.L_max, j, k] = np.sum(Psi_lm[:, m+gl.L_max, j])

    # c) m -> phi
        Psi = np.zeros( (gl.N, gl.N_thetas, gl.N_phis, 2), dtype=np.complex128)
        for j in range(gl.N):
            for k in range(gl.N_thetas):
                for n in range(gl.N_phis):
                    Psi[j, k, n, 0] = np.sum(  Psi_m[:, j, k] * np.exp(1j * np.arange(-gl.L_max, gl.L_max) * np.linspace(0.0, 2*np.pi, gl.N_phis)[n])  ) 
                    # np.exp(1j * np.arange(-gl.L_max, gl.L_max) * np.linspace(0.0, 2*np.pi, gl.N_phis)[n]) as a whole has shape (2*gl.L_max, )

    # 4) Propagate for 1 timestep in the laser field
        Psi[:, :, :, 1] = propagation_in_laser_field(Psi[:, :, :, 0], t)


def propagation_in_laser_field(Psi_grid_repr, timestep):
    E_field_cartesian = E_field(timestep + 0.5) # numpy array of shape (3, )
    a, b, c = cartesian_to_spherical(E_field_cartesian[0], E_field_cartesian[1], E_field_cartesian[2])
    E_field_spherical = np.array( [a, b, c] )

    for j in range(gl.N):
        for k in range(gl.N_thetas):
            for n in range(gl.N_phis):

                r = roots[j]
                theta = k * gl.delta_theta
                phi = n * gl.delta_phi
                vec_r = np.array( [r, theta, phi] )

                V = (-vec_r * E_field_spherical)

                Psi_grid_repr[j, k, n]  =  Psi_grid_repr[j, k, n] * np.exp(-1j * V * gl.delta_t)
    return Psi_grid_repr



def E_field(t):
    # t from arguments is current_timestep + 0.5
    
    pass

def TwoDim_quadrature(Psi, l, m):
    # Psi is a numpy 2 dimensional array of size (gl.N_thetas, gl.N_phis)

    # 1) Integration wrt theta variable:
    # {cos(theta_k)} are the L+1 zeros of P_{L+1}(cos(theta_k)) where L = gl.L_max (i.e. the maximum partial wave number)
    intermediate_res = theta_Gauss_Quad(Psi, l) # intermediate_res is 1 dimensional
    # 2) Integration wrt phi variable:
    final_res = phi_Trapez_Quad(intermediate_res, m)


def theta_Gauss_Quad(Psi, l):
    b =  roots_legendre(gl.L_max+1) 
    roots_here, weights_here = b[0], b[1]
    poly_results = eval_legendre(l, roots_here)

    containerr = np.zeros( (gl.N_phis, ), dtype=np.complex128)

    for idx in range(gl.N_phis):
        containerr[idx] = np.sum(weights_here[1:] * poly_results[1:] * Psi[1:, idx])
    return containerr

def phi_Trapez_Quad(bla, m):
    # bla is 1 dimensional
    exp_of_phis = np.exp(-1j * m * gl.phis)
    return np.trapz(bla * exp_of_phis)