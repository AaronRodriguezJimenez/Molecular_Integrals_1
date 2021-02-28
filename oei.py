import numpy as np
from scipy import misc
from scipy import special
from scipy import linalg


# ***************************************************************
# This contains the code for calculating matrix elements of the
# one-electron integral matrices S,T and V
# ***************************************************************

# *************Overlap integrals*********
def build_S(basis, S):
    """
    This function depends on  the complete basis as well as the basis size.
    It calls an intermediate function compute_Si and compute_third_center.
    :param basis: Contains the basis set information
    :return: the overlap integral
    """
    # Create an empty KxK Numpy matrix come from main.py
    # A is an integer counting variable, bA is equal to an element of the array basis
    for A, bA in enumerate(basis): # access row A os S matrix, call info for orbital A
        for B, bB in enumerate(basis): # access column B of S matrix, call info for orbital B
            # pull basis set info,
            for a, dA in zip(bA['a'],bA['d']): # a is alpha, dA is contraction coefficient
                for b, dB in zip(bB['a'],bB['d']):
                    RA = bA['R']   # collect atom-centered coordinates
                    RB = bB['R']
                    lA, mA, nA = bA['l'],bA['m'],bA['n'] # collect orbital angular moments
                    lB, mB, nB = bB['l'],bB['m'],bB['n']

                    RP = compute_third_center(a, RA, b, RB)

                    S[A,B] += dA * dB * N(a,lA, mA, nA) * N(b, lB, mB, nB) * \
                              np.exp(-( a*b / (a+b)) * compute_ABsq(RA,RB)) * \
                              compute_Si(lA, lB, RP[0]-RA[0], RP[0]-RB[0], a+b) * \
                              compute_Si(mA, mB, RP[1]-RA[1], RP[1]-RB[1], a+b) * \
                              compute_Si(nA, nB, RP[2]-RA[2], RP[2]-RB[2], a+b)

    return S


# *************Kinetic Energy integrals*********
def build_T(basis, T):
    """
    This function returns the Kinetic energy matrix computed from basis
    T-elements are only the sum of seven overlap integrals, here grouped in common terms.
    """
    # Create an empty KxK Numpy matrix come from main.py
    for A, bA in enumerate(basis): # access row A os S matrix, call info for orbital A
        for B, bB in enumerate(basis): # access column B of S matrix, call info for orbital B
            for a, dA in zip(bA['a'],bA['d']): # a is alpha, dA is contraction coefficient
                for b, dB in zip(bB['a'],bB['d']):
                    RA = bA['R']   # collect atom-centered coordinates
                    RB = bB['R']
                    lA, mA, nA = bA['l'],bA['m'],bA['n'] # collect orbital angular moments
                    lB, mB, nB = bB['l'],bB['m'],bB['n']


                    RP = compute_third_center(a, RA, b, RB)
                    g = a + b

                    T[A,B] += dA * dB * N(a,lA, mA, nA) * N(b, lB, mB, nB) * np.exp(-( a*b / g) * compute_ABsq(RA,RB)) *\
                              (b*(2*(lB+mB+nB)+3) *
                              compute_Si(lA, lB, RP[0]-RA[0], RP[0]-RB[0], g) *
                              compute_Si(mA, mB, RP[1]-RA[1], RP[1]-RB[1], g) *
                              compute_Si(nA, nB, RP[2]-RA[2], RP[2]-RB[2], g) -

                              compute_Si(mA, mB, RP[1]-RA[1], RP[1]-RB[1], g) * compute_Si(nA, nB, RP[2]-RA[2], RP[2]-RB[2], g) *
                               ((2*b**2) * compute_Si(lA, lB+2, RP[0]-RA[0], RP[0]-RB[0], g) +
                               (0.5 * lB*(lB-1) * compute_Si(lA, lB-2, RP[0]-RA[0], RP[0]-RB[0], g))) -

                              compute_Si(mA, mB, RP[0]-RA[0], RP[0]-RB[0], g)* compute_Si(nA, nB, RP[2]-RA[2], RP[2]-RB[2], g) *
                              ((2*b**2) * compute_Si(mA, mB+2, RP[1]-RA[1], RP[1]-RB[1], g) +
                               (0.5 * mB*(mB-1) * compute_Si(mA, mB-2, RP[1]-RA[1], RP[1]-RB[1], g))) -

                              compute_Si(lA, lB, RP[0]-RA[0], RP[0]-RB[0], g)* compute_Si(mA, mB, RP[1]-RA[1], RP[1]-RB[1], g) *
                              ((2*b**2) * compute_Si(nA, nB+2, RP[2]-RA[2], RP[2]-RB[2], g) +
                               (0.5 * nB*(nB-1) * compute_Si(nA, nB-2, RP[2]-RA[2], RP[2]-RB[2], g))) )
    return T


# *************Electron-nuclear attraction integrals*********
def build_V(basis, V, R, Z, atoms):
    """
    Calculate the product of the x, y, z components of the nuclear-attraction integral
    over the Gaussian primitives.
    """
    for A, bA in enumerate(basis):
        for B, bB in enumerate(basis):
            for C, rC in enumerate(R):

                for a, dA in zip(bA['a'], bA['d']):
                    for b, dB in zip(bB['a'], bB['d']):

                        RA, RB, RC = bA['R'], bB['R'], rC
                        lA, mA, nA = bA['l'], bA['m'], bA['n']
                        lB, mB, nB = bB['l'], bB['m'], bB['n']

                        V[C, A, B] += dA*dB*Vxyz(a,b,lA,mA,nA,lB,mB,nB,RA,RB,RC, Z[atoms[C]])
    return V


def Vxyz(a,b,lA,mA,nA,lB,mB,nB,RA,RB,RC,Z):
    """
     Working equation for the matrix element (A|r_c|B)
    """
    g = a + b
    RP = compute_third_center(a, RA, b, RB)

    ABsq = (compute_ABsq(RA,RB))
    PCsq = (compute_ABsq(RP,RC))
    Vxyz = 0.0
    for l in range(0,lA+lB+1):
        for r in range(0,int(l/2)+1):
            for i in range(0,int((l - 2*r)/2)+1):
                Vx = Vi(l,r,i,lA,lB,RA[0],RB[0],RC[0],RP[0],g)

                for m in range(0,mA+mB+1):
                    for s in range(0,int(m/2)+1):
                        for j in range(0,int((m - 2*s)/2)+1):
                            Vy = Vi(m,s,j,mA,mB,RA[1],RB[1],RC[1],RP[1],g)

                            for n in range(0,nA+nB+1):
                                for t in range(0,int(n/2)+1):
                                    for k in range(0,int((n - 2*t)/2)+1):
                                        Vz = Vi(n,t,k,nA,nB,RA[2],RB[2],RC[2],RP[2],g)

                                        nu = l+m+n - 2*(r+s+t) - (i+j+k)
                                        F = Boys_function(nu, g*abs(PCsq))
                                        Vxyz += Vx * Vy * Vz * F

    Na = N(a,lA,mA,nA) # Normalization factor for orbital A
    Nb = N(b,lB,mB,nB) # Normalization factor for orbital B

    Vxyz *= 2*np.pi/g
    Vxyz *= np.exp(-a*b*abs(ABsq)/g)
    Vxyz *= Na * Nb
    Vxyz *= -Z
    return Vxyz

# *************Complementary functions**********************
def compute_Si(lA,lB,PA,PB,gamma):
    """
    Calculate the i-th coordinate contribution to the matrix element S[A,B].
    """
    Si= 0.0
    for k in range( int((lA + lB)/2) + 1 ): # note range is up to add INCLUDING floor of (lA+lB)/2, hence +1
        Si += compute_ck(2*k, lA, lB, PA, PB) * np.sqrt(np.pi / gamma) * \
              (special.factorial2(2*k-1,exact=True) / (2*gamma)**k)
    return Si

def Vi(l,r,i, lA, lB, Ai, Bi, Ci, Pi, g):
    """
    Calculate the i-th component of the nuclear-attraction integral
    Gaussian primitives
    :return: Vx Vy or Vz used in Vxyz() function
    """
    epsilon = 1/(4*g)
    Vi = (-1)**l
    Vi *= compute_ck(l,lA, lB, Pi-Ai, Pi-Bi)
    Vi *= (-1)**i * special.factorial(l,exact=True)
    Vi *= (Pi - Ci)**(l - 2*r - 2*i)
    Vi *= epsilon**(r + i)
    Vi /= special.factorial(r,exact=True)
    Vi /= special.factorial(i,exact=True)
    Vi /= special.factorial(l-2*r-2*i,exact=True)

    return Vi


def compute_ck(k,l,m,a,b):
    ck = 0.0
    for i in range(l+1):
        for j in range(m+1):
            if i + j == k:
                ck += special.binom(l,i) * special.binom(m,j) * a**(l-i) * b**(m-j)

    return ck

# This function is used during computation of S,T,V and G
def N(alpha, l, m, n):
    """
    Compute the normalization constant factor for primitives
    """
    N = (2 * alpha / np.pi) ** (3 / 2) * (4 * alpha) ** (l + m + n) \
        / (special.factorial2(2 * l - 1, exact=True)) * \
        (special.factorial2(2 * m - 1, exact=True)) * \
        (special.factorial2(2 * n - 1, exact=True))
    N = N ** (1 / 2)

    return N


def compute_third_center(a, RA, b, RB):
    """
  Compute a Gaussian product third center (Example: P betwen A and B)
  ;param a, b: alphas of Gaussian A, B :param RA, RB: arrays of A B coordinates
  :return: P
  """
    P = []
    for i in range(len(RA)):
      P.append((a * RA[i] + b * RB[i]) / (a + b) )
    return P


def compute_ABsq(RA,RB):
  """
  This function compute the distance between two atoms (or two atom-centered orbitals)
  :param RA and param RB: arrays of the coordinates (or orbitals)
  :return: ABsquared
  """
  ABsq = 0.0
  for i in range(len(RA)):
    ABsq = ABsq + (RA[i] - RB[i])**2
  return ABsq


def Boys_function(nu, x):
    if x < 1e-7: # Small values of x (taylor expression to second order)
        F = (2*nu +1)**(-1) - (x)*(2 * nu +3)**(-1)
    else:
     F = 0.5 * x**(-(nu + 0.5)) * special.gammainc(nu + 0.5, x) * special.gamma(nu + 0.5)
    return F
