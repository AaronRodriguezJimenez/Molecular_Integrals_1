import numpy as np
from scipy import linalg

# This contains the code for calculating the electronic energy which depends on S, T, V and G
# matrices.

scf_max_iter = 20 # Maximum number of SCF iterations
print("HF PROCEDURE")

def HFenergy(N,Vnn,S,T,V,G,K):
    #some initial values
    # Build the core-Hamiltonian
    Hcore = T + V
    # Guess initial density matrix
    D = np.zeros((K,K))
    # Initial two-electron contribution to the Fock matrix
    P = np.zeros((K,K))
    # Generate the orthonormal transformation matrix X from S
    X = linalg.sqrtm(linalg.inv(S) )

    count = 1   # Iteration counter
    convergence = 1e-5  #HF convergence trheshold

    # Initialize the SCF procedure:
    Eel = 0.0 # Initial value of the total electronic energy
    for iteration in range(scf_max_iter+1):
        E_old = Eel   #set E_old to energy from previous cycle, for comparison
        # K is the basis size
        # the indices m and n correspond to "mu" and "nu" in standard notations
        # the indices l and s correspond to "lambda" and "sigma" in standard notations
        for m in range(K):
            for n in range(K):
                P[m,n] = 0.0
                for l in range(K):
                    for s in range(K):
                        P[m,n] += D[l,s] * ( G[m,n,s,l] - 0.5*G[m,l,s,n])
        # Add P to the core-Hamiltonian to obtain the Fock matrix:
        F = Hcore + P
        # Transform the Fock matrix to the orthonormal basis with dot products
        # Fp is F' in standard notations
        Fp = X @ F @ X
        # Diagonalize Fp to obtain eigenvalues (e) and eigenvectors (Cp)
        e, Cp = linalg.eigh(Fp)
        # Calculate the molecular orbitals
        C = X@Cp
        #Form a new and improved density matrix using C
        for m in range(K):
            for n in range(K):
                D[m,n] = 0.0
                for a in range(int(N/2)):
                    D[m,n] += 2 * (C[m,a] * C[n,a] )

        Eel = 0.0 # reset current iteration's electronic energy to 0 Hartrees
        # Calculate the electronic energy, an expectation value
        for m in range(K):
            for n in range(K):
                Eel += 0.5*(D[m,n]*(Hcore[m,n] + F[m,n]))

        print('Eel (iteration {:2d}): {:12.6f}'.format(count,Eel))
        if (np.fabs(Eel - E_old) < convergence) and (iteration > 0):
            break
        count +=1

    # Print total energy
    print("\nEt = Eel + Enn".format(Eel, Vnn, Eel+Vnn, count))
    print("   = {:.6f} + {:.6f}".format(Eel, Vnn))
    print("   = {:.6f} Hartrees ({} iterations)\n".format(Eel+Vnn, count))