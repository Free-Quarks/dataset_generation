import numpy as np

def serid_rk3(t0, tf, dt, S0, E0, I0, R0, beta, gamma, delta):
    
    def dSdt(t, S, E, I, R):
        return -beta(t) * S * I
    
    def dEdt(t, S, E, I, R):
        return beta(t) * S * I - delta * E
    
    def dIdt(t, S, E, I, R):
        return delta * E - gamma * I
    
    def dRdt(t, S, E, I, R):
        return gamma * I
    
    t = np.arange(t0, tf + dt, dt)
    N = len(t)
    
    S = np.zeros(N)
    E = np.zeros(N)
    I = np.zeros(N)
    R = np.zeros(N)
    
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    
    for i in range(1, N):
        k1_S = dSdt(t[i-1], S[i-1], E[i-1], I[i-1], R[i-1])
        k1_E = dEdt(t[i-1], S[i-1], E[i-1], I[i-1], R[i-1])
        k1_I = dIdt(t[i-1], S[i-1], E[i-1], I[i-1], R[i-1])
        k1_R = dRdt(t[i-1], S[i-1], E[i-1], I[i-1], R[i-1])
        
        k2_S = dSdt(t[i-1] + 0.5 * dt, S[i-1] + 0.5 * dt * k1_S, E[i-1] + 0.5 * dt * k1_E, I[i-1] + 0.5 * dt * k1_I, R[i-1] + 0.5 * dt * k1_R)
        k2_E = dEdt(t[i-1] + 0.5 * dt, S[i-1] + 0.5 * dt * k1_S, E[i-1] + 0.5 * dt * k1_E, I[i-1] + 0.5 * dt * k1_I, R[i-1] + 0.5 * dt * k1_R)
        k2_I = dIdt(t[i-1] + 0.5 * dt, S[i-1] + 0.5 * dt * k1_S, E[i-1] + 0.5 * dt * k1_E, I[i-1] + 0.5 * dt * k1_I, R[i-1] + 0.5 * dt * k1_R)
        k2_R = dRdt(t[i-1] + 0.5 * dt, S[i-1] + 0.5 * dt * k1_S, E[i-1] + 0.5 * dt * k1_E, I[i-1] + 0.5 * dt * k1_I, R[i-1] + 0.5 * dt * k1_R)
        
        k3_S = dSdt(t[i-1] + dt, S[i-1] - dt * k1_S + 2 * dt * k2_S, E[i-1] - dt * k1_E + 2 * dt * k2_E, I[i-1] - dt * k1_I + 2 * dt * k2_I, R[i-1] - dt * k1_R + 2 * dt * k2_R)
        k3_E = dEdt(t[i-1] + dt, S[i-1] - dt * k1_S + 2 * dt * k2_S, E[i-1] - dt * k1_E + 2 * dt * k2_E, I[i-1] - dt * k1_I + 2 * dt * k2_I, R[i-1] - dt * k1_R + 2 * dt * k2_R)
        k3_I = dIdt(t[i-1] + dt, S[i-1] - dt * k1_S + 2 * dt * k2_S, E[i-1] - dt * k1_E + 2 * dt * k2_E, I[i-1] - dt * k1_I + 2 * dt * k2_I, R[i-1] - dt * k1_R + 2 * dt * k2_R)
        k3_R = dRdt(t[i-1] + dt, S[i-1] - dt * k1_S + 2 * dt * k2_S, E[i-1] - dt * k1_E + 2 * dt * k2_E, I[i-1] - dt * k1_I + 2 * dt * k2_I, R[i-1] - dt * k1_R + 2 * dt * k2_R)
        
        S[i] = S[i-1] - (dt / 6) * (k1_S + 4 * k2_S + k3_S)
        E[i] = E[i-1] - (dt / 6) * (k1_E + 4 * k2_E + k3_E)
        I[i] = I[i-1] - (dt / 6) * (k1_I + 4 * k2_I + k3_I)
        R[i] = R[i-1] - (dt / 6) * (k1_R + 4 * k2_R + k3_R)
    
    return t, S, E, I, R
}
