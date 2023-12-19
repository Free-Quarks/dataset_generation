import numpy as np
import matplotlib.pyplot as plt


def SIR_RK3(N, I0, R0, beta, gamma, t_end, dt):
    # Initialize arrays
    t = np.arange(0, t_end, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    
    # Set initial conditions
    S[0] = N - I0 - R0
    I[0] = I0
    R[0] = R0
    
    for i in range(len(t) - 1):
        # Compute derivatives
        dS = -beta * S[i] * I[i] / N
        dI = beta * S[i] * I[i] / N - gamma * I[i]
        dR = gamma * I[i]
        
        # Compute RK3 update
        S1 = S[i] + dt * dS
        I1 = I[i] + dt * dI
        R1 = R[i] + dt * dR
        
        dS1 = -beta * S1 * I1 / N
        dI1 = beta * S1 * I1 / N - gamma * I1
        dR1 = gamma * I1
        
        S2 = (3/4) * S[i] + (1/4) * S1 + (1/4) * dt * dS1
        I2 = (3/4) * I[i] + (1/4) * I1 + (1/4) * dt * dI1
        R2 = (3/4) * R[i] + (1/4) * R1 + (1/4) * dt * dR1
        
        dS2 = -beta * S2 * I2 / N
        dI2 = beta * S2 * I2 / N - gamma * I2
        dR2 = gamma * I2
        
        S[i+1] = (1/3) * S[i] + (2/3) * S2 + (2/3) * dt * dS2
        I[i+1] = (1/3) * I[i] + (2/3) * I2 + (2/3) * dt * dI2
        R[i+1] = (1/3) * R[i] + (2/3) * R2 + (2/3) * dt * dR2
    
    # Return arrays
    return S, I, R
}
