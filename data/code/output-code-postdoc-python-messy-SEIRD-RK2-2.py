import numpy as np
import matplotlib.pyplot as plt

def seird_model(beta, sigma, gamma, mu, N, I0, E0, R0, D0, T):
    # Total population
    S0 = N - I0 - R0 - D0
    
    # Initial conditions
    S, E, I, R, D = [S0], [E0], [I0], [R0], [D0]
    
    # Time vector
    t = np.linspace(0, T, T)
    
    # Step size
    h = t[1] - t[0]
    
    # RK2 method
    for i in range(T-1):
        S_star = S[i] - beta * I[i] * S[i] * h
        E_star = E[i] + beta * I[i] * S[i] * h - sigma * E[i] * h
        I_star = I[i] + sigma * E[i] * h - (gamma + mu) * I[i] * h
        R_star = R[i] + gamma * I[i] * h
        D_star = D[i] + mu * I[i] * h
        
        S.append(S[i] - 0.5 * beta * (I[i] + I_star) * (S[i] + S_star) * h)
        E.append(E[i] + 0.5 * (beta * (I[i] + I_star) * (S[i] + S_star) - sigma * (E[i] + E_star)) * h)
        I.append(I[i] + 0.5 * (sigma * (E[i] + E_star) - (gamma + mu) * (I[i] + I_star)) * h)
        R.append(R[i] + 0.5 * gamma * (I[i] + I_star) * h)
        D.append(D[i] + 0.5 * mu * (I[i] + I_star) * h)
    
    # Return results
    return t, S, E, I, R, D

