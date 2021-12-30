import numpy as np
import numpy.linalg as LA

def householder_transf(A):
    """
    Receives a matrix and return the m*m householder
    matrix
    """
    m,n = A.shape
    e1 = np.zeros(m)
    e1[0] = 1
    x = A[:,0]
    α = -np.sign(x[0])*LA.norm(x)
    u = x - α*e1
    v = np.reshape((1/LA.norm(u))*u,(u.shape[0],1))
    Q = np.identity(m) - 2*np.matmul(v,np.transpose(v))
    verif = np.zeros(m)
    verif[0] = α
    assert np.allclose(np.matmul(Q,x),verif), f"Wrong, got {np.matmul(Q,x)} and {verif}"
    return Q


def QRdecomposition(A):
    m,n = A.shape
    t = min(m-1,n)
    i = 1
    k = 1
    Q = householder_transf(A)
    R = np.matmul(Q,A)
    Aprime = np.matmul(Q,A)[1:,1:]
    while i < t+1:
        Qprime_k = householder_transf(Aprime)
        Qk = np.block([[np.identity(k),np.zeros((k,m-k))],[np.zeros((m-k,k)),Qprime_k]])
        R = np.matmul(Qk,R)
        Q = np.matmul(Q,Qk)
        Aprime = Aprime[k:,k:]
        i += 1
        k += 1
    assert np.allclose(np.matmul(Q,R),A), "Wrong"
    R = np.triu(R)
    return Q,R