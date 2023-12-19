import numpy as np


def simulate_SIR_RK2(beta, gamma, N, I0, T):
    # Step size
    h = 0.1
    num_steps = int(T / h)
    
    # Initialize arrays
    S = np.zeros(num_steps)
    I = np.zeros(num_steps)
    R = np.zeros(num_steps)
    
    # Initial conditions
    S[0] = N - I0
    I[0] = I0
    R[0] = 0
    
    # Runge-Kutta 2nd order method
    for i in range(num_steps - 1):
        k1_S = -beta * S[i] * I[i] / N
        k1_I = beta * S[i] * I[i] / N - gamma * I[i]
        k2_S = -beta * (S[i] + h * k1_S) * (I[i] + h * k1_I) / N
        k2_I = beta * (S[i] + h * k1_S) * (I[i] + h * k1_I) / N - gamma * (I[i] + h * k1_I)
        
        S[i+1] = S[i] + h * (k1_S + k2_S) / 2
        I[i+1] = I[i] + h * (k1_I + k2_I) / 2
        R[i+1] = R[i] + h * gamma * (I[i] + h * k1_I) / 2
    
    return S, I, R
}

