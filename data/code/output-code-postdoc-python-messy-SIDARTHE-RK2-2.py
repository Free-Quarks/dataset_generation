import numpy as np
import matplotlib.pyplot as plt

def SIDARTHE_RK2(initial_conditions, parameters, t_max, dt):
    S_0, I_0, D_0, A_0, R_0, T_0, H_0, E_0 = initial_conditions
    alpha, beta, gamma, delta, theta, rho, epsilon, pi = parameters
    
    n_steps = int(t_max // dt)
    t = np.linspace(0, t_max, n_steps)
    dt = t[1] - t[0]
    
    S = np.zeros(n_steps)
    I = np.zeros(n_steps)
    D = np.zeros(n_steps)
    A = np.zeros(n_steps)
    R = np.zeros(n_steps)
    T = np.zeros(n_steps)
    H = np.zeros(n_steps)
    E = np.zeros(n_steps)
    
    S[0], I[0], D[0], A[0], R[0], T[0], H[0], E[0] = S_0, I_0, D_0, A_0, R_0, T_0, H_0, E_0
    
    for i in range(1, n_steps):
        k1_1 = dt * (- (beta + alpha) * S[i-1] * I[i-1] - (rho + epsilon) * S[i-1] * A[i-1])
        k1_2 = dt * (alpha * S[i-1] * I[i-1] + epsilon * S[i-1] * A[i-1] - delta * I[i-1])
        k1_3 = dt * (delta * I[i-1] - gamma * D[i-1])
        k1_4 = dt * (rho * S[i-1] * A[i-1] - theta * A[i-1])
        k1_5 = dt * (gamma * D[i-1] + theta * A[i-1] - pi * R[i-1])
        k1_6 = dt * (pi * R[i-1])
        k1_7 = dt * (beta * S[i-1] * I[i-1])
        k1_8 = dt * (alpha * S[i-1] * I[i-1])
        
        k2_1 = dt * (- (beta + alpha) * (S[i-1] + k1_1/2) * (I[i-1] + k1_2/2) - (rho + epsilon) * (S[i-1] + k1_1/2) * (A[i-1] + k1_4/2))
        k2_2 = dt * (alpha * (S[i-1] + k1_1/2) * (I[i-1] + k1_2/2) + epsilon * (S[i-1] + k1_1/2) * (A[i-1] + k1_4/2) - delta * (I[i-1] + k1_2/2))
        k2_3 = dt * (delta * (I[i-1] + k1_2/2) - gamma * (D[i-1] + k1_3/2))
        k2_4 = dt * (rho * (S[i-1] + k1_1/2) * (A[i-1] + k1_4/2) - theta * (A[i-1] + k1_4/2))
        k2_5 = dt * (gamma * (D[i-1] + k1_3/2) + theta * (A[i-1] + k1_4/2) - pi * (R[i-1] + k1_5/2))
        k2_6 = dt * (pi * (R[i-1] + k1_5/2))
        k2_7 = dt * (beta * (S[i-1] + k1_1/2) * (I[i-1] + k1_2/2))
        k2_8 = dt * (alpha * (S[i-1] + k1_1/2) * (I[i-1] + k1_2/2))
        
        S[i] = S[i-1] + k2_1
        I[i] = I[i-1] + k2_2
        D[i] = D[i-1] + k2_3
        A[i] = A[i-1] + k2_4
        R[i] = R[i-1] + k2_5
        T[i] = T[i-1] + k2_6
        H[i] = H[i-1] + k2_7
        E[i] = E[i-1] + k2_8
    
    return S, I, D, A, R, T, H, E
}
