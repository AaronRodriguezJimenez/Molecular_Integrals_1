# Molecular_Integrals_1
Code from the "Molecular Integrals" Project

This code follows the "Student-Friendly Guide to Molecular Integrals" paper of: Kevin V. Murphy*, Justin M. Turney, and Henry F. Schaefer III (J. Chem. Educ. 2018, 95, 9, 1572–1578). A complete derivation of the molecular integrals involved in the Hartree-Fock (HF) procedure is provided. 
This code allow restricted HF calculations for first- and second-row atoms within the STO-3G basis in python3. 
The proper calculation of S, T, V, and G matrices is performed and used to compute the electronic energy for a molecular system. 
The STO-3G basis set was taken from: https://www.basissetexchange.org/


- sto-3g.1.tm file contains the basis set obtained from basissetexchange.org
- main.py is the main file of the code, one can run the code by typing 
  $python3 main.py coordinates.xyz
- basis.py is a subroutine that allow the user to create the basis set dictionaries used in the molecular integral calculations.
- oei.py contains the functions to compute S, T and V matrices.
- eri.py contain the function to compute G matrix which contains the two-electron molecular integrals.
- hf.py uses the information from main.py and calculate the HF energy of the system.
- H2Ocoordinates.xyz contains the water-cartesian coordinates for test.
