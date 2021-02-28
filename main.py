import os.path   #Necessary for reading in previously generated files
import numpy as np  # Mathematical, linear algebra library
import argparse     # to be used for allowing user to specify input file
import oei
import eri
import basis
import hf
# ############################################### #
#      --- Molecular integrals ---                #
# Code written by: José Aarón Rodríguez-Jiménez   #
# DIPC, FEB-2021                                  #
# This code follows the paper:                    #
# J. Chem. Educ. 2018, 95, 9, 1572–1578           #
# ############################################### #

#
# This is the central file to drive the code.
#

# *************General Scheme **************************
# - Read information from .xyz file
# - Build the basis by accessing functions in basis.py
# - Construct Empty Matrices
# - Call functions in oei.py
# - Perform a HF calculation with hf.py
# *****************************************************

# Improve the appearance of printed matrices:
np.set_printoptions(threshold=np.inf)  # suppresses "...", printing all elements
np.set_printoptions(precision=5)      # Reduces number of decimal places per elements to 5
np.set_printoptions(linewidth=200)   # Lengthless printable line width from 75 chartacters
np.set_printoptions(suppress=True)  # suppresses hart-to-read exponential notation

#part of the code used to specify an input file when running as
# python main.py [.xyz file]
parser = argparse.ArgumentParser()
parser.add_argument('coords_file', metavar= 'coordinates.xyz', type=str)
args = parser.parse_args()

#following variable will hold the contents of the .xyz input
inputfile = open(args.coords_file, 'r') #open .xyz file without being able to modify it (read only)

#-----------------------------------------------------
#Part to read in the .xyz and generate the atoms and coordinates array.
def read_inputfile(inputfile):
    """
    Builds two arrays from .xyz input file:
    :return: atoms (the elements of atomic symbols in order of the input),
             coords (a nested array containing elements of coordinates, also in order)
             (1 coordinates array per atom)
    """
    atoms = []
    coords = []

    #read through inputfile line by line
    n = 0
    for line in inputfile.readlines() [2:]:    # [2:] means read only from the third line of the input
        atom, x, y, z = line.split()           # from each line collect the information
        atoms.append( atom )                   # append to atoms and coords arrays
        coords.append( [float(x), float(y), float(z)])
        n += 1
    return atoms, coords, n

#______________________________________________________
# Build array of atoms and coordinates by calling the previous function.
# n gives the number of atoms in .xyz file, calculations assume neutrally charged molecules.
atoms, R, n = read_inputfile(inputfile)
# Build the STO-3G basis using these inputs
# basis size (number of AOs) ==> dimension of a matrix
orbitalbasis, K = basis.build_sto3Gbasis(atoms, R)
# Input matrices for S and T
S = np.zeros( (K,K) )
T = np.zeros( (K,K) )
# Input matrix and arguments for V, if one wants to include more atoms
# Z must be changed and the basis must be edited to contain the additional atom-basis
Z = {'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10}
V = np.zeros( (n,K,K) )
# Input matrix for G
G = np.zeros((K,K,K,K))

# **************************************************
# Call the integral functions:
S = oei.build_S(orbitalbasis, S)
print("******** S matrix ************")
print(S)

T = oei.build_T(orbitalbasis, T)
print("******** T matrix ************")
print(T)

V = oei.build_V(orbitalbasis, V, R, Z, atoms)
Vsum = np.zeros((K,K))
for i in range(len(V)):
    Vsum += V[i]
print('\nNuclear attraction integrals, complete!')
print("******** V matrix ************")
print(Vsum)

G = eri.build_G(orbitalbasis, G)
print("******** G matrix ************")
print(G)

# ***************************************************
# HF calculation, call in HF procedure from hf.py
# first, one need to compute:
#  - Number of electrons
#  - Nuclear-Nuclear repulsion energy
# ****************************************************
print("")
print("******** Hartree-Fock procedure ************")
def electronCount(atoms):
    """
    Compute the number of atoms (N), a neutrally charged molecule is assumed
     
    """
    N = 0
    for A in atoms:
        N += Z[A]
    return N
def IJsq(RI,RJ):
    """
    :return: The squared distance between two points
    """
    return sum( (RI[i] - RJ[i])**2 for i in (0,1,2) )

def nuclearRep(atoms):
    """
    :reurn: The nuclear-nuclear  repuslion energy
    """
    Vnn = 0.0
    for a, A in enumerate(atoms):
        for b, B in enumerate(atoms):
            if b > a:
                num = Z[A] * Z[B]
                den = np.sqrt(IJsq(R[a],R[b]) )
                Vnn += num/den
    return Vnn

# Call the HF procedure in hf.py
N = electronCount(atoms)
Vnn = nuclearRep(atoms)

hf.HFenergy(N,Vnn,S,T,Vsum,G,K)


