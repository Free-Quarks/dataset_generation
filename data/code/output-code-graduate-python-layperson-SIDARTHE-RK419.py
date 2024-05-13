import numpy as np


def run_model(S0, I0, D0, R0, A0, T0, H0, E0, t_max, N, beta, delta, gamma, alpha, rho, theta):
    # Initialize arrays
    S = np.zeros(t_max)
    I = np.zeros(t_max)
    D = np.zeros(t_max)
    R = np.zeros(t_max)
    A = np.zeros(t_max)
    T = np.zeros(t_max)
    H = np.zeros(t_max)
    E = np.zeros(t_max)
    
    # Set initial conditions
    S[0] = S0
    I[0] = I0
    D[0] = D0
    R[0] = R0
    A[0] = A0
    T[0] = T0
    H[0] = H0
    E[0] = E0
    
    # Run simulation
    for t in range(1, t_max):
        # Compute derivatives
        dS_dt = -beta * S[t-1] * (I[t-1] + alpha * A[t-1]) / N
        dI_dt = beta * S[t-1] * (I[t-1] + alpha * A[t-1]) / N - delta * I[t-1] - gamma * I[t-1]
        dD_dt = delta * I[t-1]
        dR_dt = gamma * I[t-1]
        dA_dt = rho * delta * I[t-1] - theta * A[t-1]
        dT_dt = theta * A[t-1]
        dH_dt = (1 - rho) * delta * I[t-1]
        dE_dt = beta * S[t-1] * (I[t-1] + alpha * A[t-1]) / N
        
        # Update state variables using RK4 method
        h = 1
        k1_S = h * dS_dt
        k1_I = h * dI_dt
        k1_D = h * dD_dt
        k1_R = h * dR_dt
        k1_A = h * dA_dt
        k1_T = h * dT_dt
        k1_H = h * dH_dt
        k1_E = h * dE_dt
        
        k2_S = h * (dS_dt + 0.5 * k1_S)
        k2_I = h * (dI_dt + 0.5 * k1_I)
        k2_D = h * (dD_dt + 0.5 * k1_D)
        k2_R = h * (dR_dt + 0.5 * k1_R)
        k2_A = h * (dA_dt + 0.5 * k1_A)
        k2_T = h * (dT_dt + 0.5 * k1_T)
        k2_H = h * (dH_dt + 0.5 * k1_H)
        k2_E = h * (dE_dt + 0.5 * k1_E)
        
        k3_S = h * (dS_dt + 0.5 * k2_S)
        k3_I = h * (dI_dt + 0.5 * k2_I)
        k3_D = h * (dD_dt + 0.5 * k2_D)
        k3_R = h * (dR_dt + 0.5 * k2_R)
        k3_A = h * (dA_dt + 0.5 * k2_A)
        k3_T = h * (dT_dt + 0.5 * k2_T)
        k3_H = h * (dH_dt + 0.5 * k2_H)
        k3_E = h * (dE_dt + 0.5 * k2_E)
        
        k4_S = h * (dS_dt + k3_S)
        k4_I = h * (dI_dt + k3_I)
        k4_D = h * (dD_dt + k3_D)
        k4_R = h * (dR_dt + k3_R)
        k4_A = h * (dA_dt + k3_A)
        k4_T = h * (dT_dt + k3_T)
        k4_H = h * (dH_dt + k3_H)
        k4_E = h * (dE_dt + k3_E)
        
        S[t] = S[t-1] + (1/6) * (k1_S + 2*k2_S + 2*k3_S + k4_S)
        I[t] = I[t-1] + (1/6) * (k1_I + 2*k2_I + 2*k3_I + k4_I)
        D[t] = D[t-1] + (1/6) * (k1_D + 2*k2_D + 2*k3_D + k4_D)
        R[t] = R[t-1] + (1/6) * (k1_R + 2*k2_R + 2*k3_R + k4_R)
        A[t] = A[t-1] + (1/6) * (k1_A + 2*k2_A + 2*k3_A + k4_A)
        T[t] = T[t-1] + (1/6) * (k1_T + 2*k2_T + 2*k3_T + k4_T)
        H[t] = H[t-1] + (1/6) * (k1_H + 2*k2_H + 2*k3_H + k4_H)
        E[t] = E[t-1] + (1/6) * (k1_E + 2*k2_E + 2*k3_E + k4_E)
    
    return S, I, D, R, A, T, H, E
}
