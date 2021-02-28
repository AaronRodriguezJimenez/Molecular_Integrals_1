from scipy import special
import numpy as math
import oei


# Code for calculating matrix elements of the two-electron integrals
# Two electron repulsion integrals are effectively just and extension of
# the Nucleus-electron integrals (one needs more loops)

def build_G(basis, G):
    """
    This function calls the intermediate function Gxyz to calculate
    integrals over primitives, compute all elements of the G=(AB|CD)
    4-dimensional matrix
    """
    for A, bA in enumerate(basis):
        for B, bB in enumerate(basis):
            for C, bC in enumerate(basis):
                for D, bD in enumerate(basis):

                    for a, dA in zip(bA['a'], bA['d']):  # a is alpha, dA is contraction coefficient
                        for b, dB in zip(bB['a'], bB['d']):
                            for c, dC in zip(bC['a'], bC['d']):
                                for d, dD in zip(bD['a'], bD['d']):
                                    RA, RB = bA['R'], bB['R']
                                    RC, RD = bC['R'], bD['R']

                                    lA, mA, nA = bA['l'], bA['m'], bA['n']
                                    lB, mB, nB = bB['l'], bB['m'], bB['n']
                                    lC, mC, nC = bC['l'], bC['m'], bC['n']
                                    lD, mD, nD = bD['l'], bD['m'], bD['n']

                                    tei = dA * dB * dC * dD  # Product of contraction coefficients
                                    tei *= Gxyz(a, b, c, d, lA, mA, nA, lB, mB, nB,
                                                lC, mC, nC, lD, mD, nD, RA, RB, RC,RD)
                                    G[A,B,C,D] += tei
    return G


def Gxyz(a, b, c, d, lA, mA, nA, lB, mB, nB, lC, mC, nC, lD, mD, nD, RA, RB, RC, RD):
    """
     Working equation for the matrix element (AB|CD)
    """
    gp = a + b
    gq = c + d
    delta = 1 / (4 * gp) + 1 / (4 * gq)

    RP = oei.compute_third_center(a, RA, b, RB)
    RQ = oei.compute_third_center(c, RC, d, RD)

    # New centers PA, PB, QC, QD and PQ
    ABsq = oei.compute_ABsq(RA, RB)
    CDsq = oei.compute_ABsq(RC, RD)
    PQsq = oei.compute_ABsq(RP, RQ)

    Gxyz = 0.0
    for lp in range(0, lA + lB + 1):
        for rp in range(0, int(lp / 2) + 1):
            for lq in range(0, lC + lD + 1):
                for rq in range(0, int(lq / 2) + 1):
                    for i in range(0, int((lp + lq - 2 * rp - 2 * rq) / 2) + 1):
                        Gx = Gi(lp,lq,rp,rq, i , lA,lB,RA[0],RB[0],RP[0],gp ,lC,lD,RC[0],RD[0],RQ[0],gq)

                        for mp in range(0, mA + mB + 1):
                            for sp in range(0, int(mp / 2) + 1):
                                for mq in range(0, mC + mD + 1):
                                    for sq in range(0, int(mq / 2) + 1):
                                        for j in range(0, int((mp + mq - 2 * sp - 2 * rq) / 2) + 1):
                                            Gy = Gi(mp,mq,sp,sq, j , mA,mB,RA[1],RB[1],RP[1],gp ,mC,mD,RC[1],RD[1],RQ[1],gq)

                                            for np in range(0, nA + nB + 1):
                                                for tp in range(0, int(np / 2) + 1):
                                                    for nq in range(0, nC + nD + 1):
                                                        for tq in range(0, int(nq / 2) + 1):
                                                            for k in range(0, int((np + nq - 2 * tp - 2 * tq) / 2) + 1):
                                                                Gz = Gi(np,nq,tp,tq, k , nA,nB,RA[2],RB[2],RP[2],gp ,nC,nD,RC[2],RD[2],RQ[2],gq)

                                                                nu = lp + lq + mp + mq + np + nq - \
                                                                     2 * (rp + rq + sp + sq + tp + tq) - \
                                                                     (i + j + k)

                                                                F = oei.Boys_function(nu, PQsq / (4 * delta))
                                                                Gxyz += Gx * Gy * Gz * F

    Gxyz *= 2 * (math.pi ** 2 )/ (gp * gq)
    Gxyz *= math.sqrt(math.pi / (gp + gq))
    Gxyz *= math.exp(-(a * b * ABsq) / gp)
    Gxyz *= math.exp(-(c * d * CDsq) / gq)

    Na = oei.N(a, lA, mA, nA)  # Normalization factor for orbital A
    Nb = oei.N(b, lB, mB, nB)  # Normalization factor for orbital B
    Nc = oei.N(c, lC, mC, nC)  # Normalization factor for orbital C
    Nd = oei.N(d, lD, mD, nD)  # Normalization factor for orbital D

    Gxyz *= Na * Nb * Nc * Nd

    return Gxyz


def Gi(lp,lq,rp,rq, i , lA,lB,RAi,RBi,RPi,gp ,lC,lD,RCi,RDi,RQi,gq):
    """
    Auxiliary function to compute the G cartesian components.
    :param delta = (1/4gammap) + (1/4gammaq)
    PAi, PBi, PQi are components of the vectos PA, PB and PQ
    similar for QCi and QDi
    """
    delta = 1 / (4 * gp) + 1 / (4 * gq)

    Gi = (-1) ** lp
    Gi *= theta(lp, lA, lB, RPi - RAi, RPi - RBi, rp, gp)
    Gi *= theta(lq, lC, lD, RQi - RCi, RQi - RDi, rq, gq)
    Gi *= (-1)**i * (2 * delta) ** (2 * (rp + rq))
    Gi *= special.factorial(lp + lq - 2*rp - 2*rq, exact=True) * delta ** i
    Gi *= (RPi-RQi)**(lp + lq - 2 *(rp + rq + i))
    Gi /= (4 * delta)**(lp + lq)
    Gi /= special.factorial(i, exact=True)
    Gi /= special.factorial(lp + lq - 2*(rp + rq + i), exact=True)

    return Gi


def theta(l, lA, lB, a, b, r, g):
    """
    Second auxiliary function for Eris, it depends on ck coefficients
    that are computed in oei.py
    g is gamma
    """
    th = oei.compute_ck(l, lA, lB, a, b)
    th *= special.factorial(l, exact=True)
    th *= g ** (r - l)
    th /= special.factorial(r, exact=True)
    th /= special.factorial(l - 2 * r, exact=True)

    return th

