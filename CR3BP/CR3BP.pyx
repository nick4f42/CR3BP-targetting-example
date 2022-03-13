import numpy as np
cimport numpy as cnp

cdef extern from "c_CR3BP.h":
    void c_dS_CR3BP(double* S, double mu, double* dS)
    void c_dSTM_CR3BP(double* S, double* STM, double mu, double *dS, double *dSTM)

def dS_CR3BP(double t, cnp.ndarray[cnp.double_t, ndim=1] S, double mu):
    """CR3BP derivative of the state function [x, y, z, vx, vy, vz]."""
    cdef cnp.ndarray[cnp.double_t, ndim=1] dS = np.empty(6, dtype=np.double)
    c_dS_CR3BP(<double*> S.data, mu, <double*> dS.data)
    return dS

def dSTM_CR3BP(double t, cnp.ndarray[cnp.double_t, ndim=1] S, double mu):
    """CR3BP derivative of the state function and flattened state transition matrix.

    S[:6] = [x, y, z, vx, vy, vz]
    S[6:] = STM.flat
    """
    cdef cnp.ndarray[cnp.double_t, ndim=1] dS = np.empty(S.size, dtype=np.double)
    c_dSTM_CR3BP(
        <double*> S.data, <double*> S.data + 6, mu,
        <double*> dS.data, <double*> dS.data + 6)
    return dS
