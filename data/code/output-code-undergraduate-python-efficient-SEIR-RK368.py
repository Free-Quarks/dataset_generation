import numpy as np
import matplotlib.pyplot as plt

def SEIR_RK3(beta, gamma, sigma, S0, E0, I0, R0, t_end, n_steps):
    dt = t_end / n_steps
    t = np.linspace(0, t_end, n_steps+1)
    S = np.zeros(n_steps+1)
    E = np.zeros(n_steps+1)
    I = np.zeros(n_steps+1)
    R = np.zeros(n_steps+1)
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    
    for i in range(n_steps):
        k1_s = -beta * S[i] * I[i] / (S[i] + E[i] + I[i] + R[i])
        k1_e = beta * S[i] * I[i] / (S[i] + E[i] + I[i] + R[i]) - sigma * E[i]
        k1_i = sigma * E[i] - gamma * I[i]
        k1_r = gamma * I[i]
        
        k2_s = -beta * (S[i] + dt/2 * k1_s) * (I[i] + dt/2 * k1_i) / (S[i] + E[i] + I[i] + R[i])
        k2_e = beta * (S[i] + dt/2 * k1_s) * (I[i] + dt/2 * k1_i) / (S[i] + E[i] + I[i] + R[i]) - sigma * (E[i] + dt/2 * k1_e)
        k2_i = sigma * (E[i] + dt/2 * k1_e) - gamma * (I[i] + dt/2 * k1_i)
        k2_r = gamma * (I[i] + dt/2 * k1_i)
        
        k3_s = -beta * (S[i] + dt/2 * k2_s) * (I[i] + dt/2 * k2_i) / (S[i] + E[i] + I[i] + R[i])
        k3_e = beta * (S[i] + dt/2 * k2_s) * (I[i] + dt/2 * k2_i) / (S[i] + E[i] + I[i] + R[i]) - sigma * (E[i] + dt/2 * k2_e)
        k3_i = sigma * (E[i] + dt/2 * k2_e) - gamma * (I[i] + dt/2 * k2_i)
        k3_r = gamma * (I[i] + dt/2 * k2_i)
        
        S[i+1] = S[i] + dt/6 * (k1_s + 4*k2_s + k3_s)
        E[i+1] = E[i] + dt/6 * (k1_e + 4*k2_e + k3_e)
        I[i+1] = I[i] + dt/6 * (k1_i + 4*k2_i + k3_i)
        R[i+1] = R[i] + dt/6 * (k1_r + 4*k2_r + k3_r)
    
    return t, S, E, I, R
}

