import numpy as np
import scipy
from matplotlib import pyplot as plt

from scipy.special import roots_legendre, legendre
from numpy.linalg import eig

import gl

# poly_Nr = legendre(gl.N_rs) # a orthopoly1d object
# poly_Nr_der = poly_Nr.deriv() # a poly1d object
# a =  np.roots(poly_Nr_der)
# roots = a # shape (gl.N_rs - 1, )
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




# Non-Linear Mapping 
def r(x):
    return gl.L * (1 + x) / (1 - x + gl.alpha)

def r_prime(x):
    return gl.L * (2 + gl.alpha) / (1 - x + gl.alpha)**2


# H_{l}^{0}(x) assembly functions
def V_l(x, l, Z):
    return ( l*(l+1) / (2*r(x)**2) ) + Coulomb(r(x), Z)

def Coulomb(r, Z):
    return (-Z / r)


Z = 1
H = np.zeros(  (gl.N_rs - 1,  gl.N_rs - 1,  gl.N_thetas),   dtype=np.complex128)
pref = gl.N_rs * (gl.N_rs + 1) / 6
diag_values_Kinetic = np.array([  (1/r_prime(x)) * pref * (1/(1-x**2)) * (1/r_prime(x)) for x in roots])

for l in range(gl.N_thetas):
    diag_values_Coulomb_and_Laser = np.array([  V_l(x, l, Z) for x in roots ])
    np.fill_diagonal( H[:, :, l],   diag_values_Kinetic + diag_values_Coulomb_and_Laser )
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if (i != j):
                H[i, j, l] = (1/r_prime(roots[i])) * (1/r_prime(roots[j])) * (1/(roots[i]-roots[j])**2) 
            # else already filled by np.fill_diagonal() from above

eig_values = np.zeros( (gl.N_rs - 1,  gl.N_thetas), dtype=np.complex128)
eig_vectors = np.zeros( (gl.N_rs - 1,  gl.N_rs - 1,  gl.N_thetas), dtype = np.complex128)

for l in range(gl.N_thetas):
    eig_values[:, l], eig_vectors[:, :, l] = eig( H[:, :, l] )

np.save("eigenVectors_N_rs{}_alpha{}_rmax{}.npy".format(gl.N_rs, gl.alpha, gl.r_max), eig_vectors)
np.save("eigenValues_N_rs{}_alpha{}_rmax{}.npy".format(gl.N_rs, gl.alpha, gl.r_max), eig_values)

