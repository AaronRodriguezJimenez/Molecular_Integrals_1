import numpy as np

#*************************************************************
# Basis set information and construction of the STO-3G basis.
#*************************************************************

# Dictionary with atomic symbols and atomic numbers:
Z = {'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10}

# Nested array wich hold the STO-3G basis set
# STO-3G contraction applies to all atoms (Z = 1,10)
d = (( 0.1543289673, 0.5353281423, 0.4446345422), # contraction coefficients for 1s orbitals
     (-0.0999672292, 0.3995128261, 0.7001154689), # contraction coefficients for 2s orbitals
     ( 0.1559162750, 0.6076837186, 0.3919573931)) # contraction coefficients for 2p orbitals

# Orbital configurations dictionary:
orbitalconfiguration = {
  'H':['1s'],
  'He':['1s'],
  'Li':['1s','2s','2px','2py','2pz'],
  'Be':['1s','2s','2px','2py','2pz'],
  'B':['1s','2s','2px','2py','2pz'],
  'C':['1s','2s','2px','2py','2pz'],
  'N':['1s','2s','2px','2py','2pz'],
  'O':['1s','2s','2px','2py','2pz'],
  'F':['1s','2s','2px','2py','2pz'],
  'Ne':['1s','2s','2px','2py','2pz'],
}

# Nested array featuring the alpha factors, the set of contraction
# coefficients in a STO-NG basis apply to all atoms.
# Taken from https://www.basissetexchange.org/

a = (
     (
         ( 3.4252509140, 0.6239137298, 0.1688554040), #H 1s
         (0.0000000000, 0.0000000000, 0.0000000000)  #H 2s, 2p
                                                   ),
     (
         ( 6.3624213940, 1.1589229990, 0.3136497915), #He 1s
         ( 0.0000000000, 0.0000000000, 0.0000000000)  #He 2s, 2p
                                                    ),
     (
         ( 16.119574750, 2.9362006630, 0.7946504870), #Li 1s
         ( 0.6362897469, 0.1478600533, 0.0480886784), #Li 2s, 2p
                                                    ),
     (
          ( 30.167870690, 5.4951153060, 1.4871926530), #Be 1s
          ( 1.3148331100, 0.3055389383, 0.0993707456), #Be 2s, 2p
                                                    ),
     (
         ( 48.791113180, 8.8873621720, 2.4052670400), #B 1s
         ( 2.2369561420, 0.5198204999, 0.1690617600), #B 2s, 2p
                                                    ),
     (
         ( 71.616837350, 13.045096320, 3.5305121600), #C 1s
         ( 2.9412493550, 0.6834830964, 0.2222899159), #C 2s, 2p
                                                    ),
     (
         ( 99.106168960, 18.052312390, 4.8856602380), #N 1s
         ( 3.7804558790, 0.8784966449, 0.2857143744), #N 2s, 2p
                                                    ),
     (
         ( 130.70932140, 23.808866050, 6.4436083130), #O 1s
         ( 5.0331513190, 1.1695961250, 0.3803889600), #O 2s, 2p
                                                    ),
     (
         ( 166.67913400, 30.360812330, 8.2168206720), #F 1s
         ( 6.4648032490, 1.5022812450, 0.4885884864), #F 2s, 2p
                                                    ),
     (
         ( 207.01560700, 37.708151240, 10.205297310), #Ne 1s
         ( 8.2463151200, 1.9162662910, 0.6232292721), #Ne 2s, 2p
                                                    )
     )
sto3Gbasis = []   #Array representing our basis


def build_sto3Gbasis(atoms,R):
  """
  This function depends on the atoms array, which lists srtings of atomic symbols of the
  atoms of the input molecule, in the same order as the input .xyz file.
  This function additionally depends on the molecule's coordinates R (a nested array),
  where each element is an array holding the coordinates of each atom, also in the same
  order as atoms.
  """

# Loop trhough the atoms array, and append orbitals (dictionaries) to the basis array
# each atom (in atoms) has an orbital configuration associated with it, so loop through these too
  K = 0 # Initialize atomic orbital counter.
  for i, atom in enumerate(atoms):
    for orbital in orbitalconfiguration[atom]:
      if orbital == '1s':
       sto3Gbasis.append(
                          {
                           'Z': Z[atom],          # atom name --> atomic number
                           'o': orbital,          # append the orbital-type string
                           'R': R[ i ],           # get array [x,y,z] of ith atom coordinates
                           'l': 0,                # s orbital ==>0 angular momentum
                           'm': 0,
                           'n': 0,
                           'a': a[ (Z[atom]-1) ][0], # append list of 1s orbital exponential factors
                           'd': d[0]                # append list of 1s orbital contraction coefficients
                          }
                        )
       K = K + 1  # Increment orbital counter
      elif orbital == '2s':
        sto3Gbasis.append(
                           {
                            'Z': Z[atom],          # atom name --> atomic number
                            'o': orbital,          # append the orbital-type string
                            'R': R[ i ],           # get array [x,y,z] of ith atom coordinates
                            'l': 0,                # s orbital ==>0 angular momentum
                            'm': 0,
                            'n': 0,
                            'a': a[ (Z[atom]-1) ][1], # append list of 2s orbital exponential factors
                            'd': d[1]                # append list of 2s orbital contraction coefficients
                           }
                         )
        K = K + 1
      elif orbital == '2px':
        sto3Gbasis.append(
                           {
                            'Z': Z[atom],          
                            'o': '2px',          
                            'R': R[ i ],          
                            'l': 1,                #2 px orbital angular momentum
                            'm': 0,
                            'n': 0,
                            'a': a[ (Z[atom]-1) ][1], # same as 2s for STO-3G basis set
                            'd': d[2]                # append list of 2p orbital contraction coefficients
                           }
                         )
        K = K + 1
      elif orbital == '2py':
        sto3Gbasis.append(
                           {
                            'Z': Z[atom],
                            'o': '2py',
                            'R': R[ i ],
                            'l': 0,                #2 py orbital angular momentum
                            'm': 1,
                            'n': 0,
                            'a': a[ (Z[atom]-1) ][1], # same as 2s for STO-3G basis set
                            'd': d[2]                # append list of 2p orbital contraction coefficients
                           }
                         )
        K = K + 1
      elif orbital == '2pz':
        sto3Gbasis.append(
                           {
                            'Z': Z[atom],
                            'o': '2pz',
                            'R': R[ i ],
                            'l': 0,                #  2px orbital angular momentum
                            'm': 0,
                            'n': 1,
                            'a': a[ (Z[atom]-1) ][1], # same as 2s for STO-3G basis set
                            'd': d[2]                # append list of 2p orbital contraction coefficients
                           }
                         )
        K = K + 1
#  When this function is called, return both the basis and the orbital count
  return sto3Gbasis, K
