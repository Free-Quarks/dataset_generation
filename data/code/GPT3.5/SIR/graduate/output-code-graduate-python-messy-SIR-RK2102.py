import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(N, beta, gamma, I0, T):
    # Initialize arrays
    t = np.linspace(0, T, T+1)
    S = np.zeros(T+1)
    I = np.zeros(T+1)
    R = np.zeros(T+1)
    
    # Initial conditions
    S[0] = N - I0
    I[0] = I0
    R[0] = 0
    
    # RK2 method
    dt = t[1] - t[0]
    for i in range(T):
        S_star = S[i] - beta*S[i]*I[i]*dt/2
        I_star = I[i] + (beta*S[i]*I[i] - gamma*I[i])*dt/2
        S[i+1] = S[i] - beta*S_star*I_star*dt
        I[i+1] = I[i] + (beta*S_star*I_star - gamma*I_star)*dt
        R[i+1] = R[i] + gamma*I_star*dt
    
    # Return results
    return S, I, R

