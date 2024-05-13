import numpy as np


def sidarthe_model(N, I0, D0, A0, R0, T0, H0, E0, beta, gamma, delta, alpha, mu, sigma, rho, t_max):
    
    def dSdt(S, I, D, A, R, T, H, E):
        N = S + I + D + A + R + T + H + E
        return -beta * S * (I + alpha * A + T) / N
    
    def dIdt(S, I, D, A, R, T, H, E):
        N = S + I + D + A + R + T + H + E
        return beta * S * (I + alpha * A + T) / N - gamma * I - delta * I
    
    def dDdt(S, I, D, A, R, T, H, E):
        return delta * I - mu * D
    
    def dAdt(S, I, D, A, R, T, H, E):
        N = S + I + D + A + R + T + H + E
        return rho * gamma * I - sigma * A - mu * A
    
    def dRdt(S, I, D, A, R, T, H, E):
        return gamma * I + sigma * A
    
    def dTdt(S, I, D, A, R, T, H, E):
        return alpha * rho * gamma * I - mu * T
    
    def dHdt(S, I, D, A, R, T, H, E):
        return (1 - alpha) * rho * gamma * I - mu * H
    
    def dEdt(S, I, D, A, R, T, H, E):
        N = S + I + D + A + R + T + H + E
        return beta * S * (I + alpha * A + T) / N + sigma * A - rho * gamma * I - mu * E
    
    # Initial conditions
    S0 = N - I0 - D0 - A0 - R0 - T0 - H0 - E0
    
    # Create empty arrays to store the results
    S = np.zeros(t_max)
    I = np.zeros(t_max)
    D = np.zeros(t_max)
    A = np.zeros(t_max)
    R = np.zeros(t_max)
    T = np.zeros(t_max)
    H = np.zeros(t_max)
    E = np.zeros(t_max)
    
    # Set initial conditions
    S[0] = S0
    I[0] = I0
    D[0] = D0
    A[0] = A0
    R[0] = R0
    T[0] = T0
    H[0] = H0
    E[0] = E0
    
    # Numerical integration
    for t in range(1, t_max):
        S[t] = S[t-1] + dt * dSdt(S[t-1], I[t-1], D[t-1], A[t-1], R[t-1], T[t-1], H[t-1], E[t-1])
        I[t] = I[t-1] + dt * dIdt(S[t-1], I[t-1], D[t-1], A[t-1], R[t-1], T[t-1], H[t-1], E[t-1])
        D[t] = D[t-1] + dt * dDdt(S[t-1], I[t-1], D[t-1], A[t-1], R[t-1], T[t-1], H[t-1], E[t-1])
        A[t] = A[t-1] + dt * dAdt(S[t-1], I[t-1], D[t-1], A[t-1], R[t-1], T[t-1], H[t-1], E[t-1])
        R[t] = R[t-1] + dt * dRdt(S[t-1], I[t-1], D[t-1], A[t-1], R[t-1], T[t-1], H[t-1], E[t-1])
        T[t] = T[t-1] + dt * dTdt(S[t-1], I[t-1], D[t-1], A[t-1], R[t-1], T[t-1], H[t-1], E[t-1])
        H[t] = H[t-1] + dt * dHdt(S[t-1], I[t-1], D[t-1], A[t-1], R[t-1], T[t-1], H[t-1], E[t-1])
        E[t] = E[t-1] + dt * dEdt(S[t-1], I[t-1], D[t-1], A[t-1], R[t-1], T[t-1], H[t-1], E[t-1])
    
    return S, I, D, A, R, T, H, E

