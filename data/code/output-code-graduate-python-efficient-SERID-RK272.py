# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt

# Function to implement the SERID model


def serid_model(beta, gamma, delta, S0, E0, R0, I0, D0, N, dt, T):
    # Initializing arrays
    S = np.zeros(T+1)
    E = np.zeros(T+1)
    R = np.zeros(T+1)
    I = np.zeros(T+1)
    D = np.zeros(T+1)
    t = np.linspace(0, T, T+1)
    
    # Assigning initial values
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    D[0] = D0
    
    # Implementing SERID model using RK2 method
    for i in range(T):
        F1 = -beta*S[i]*I[i]/N
        F2 = -beta*(S[i] + dt*F1/2)*(I[i] + dt*F1/2)/N
        S[i+1] = S[i] + dt*F2
        
        F1 = beta*S[i]*I[i]/N - gamma*E[i]
        F2 = beta*(S[i] + dt*F1/2)*(I[i] + dt*F1/2)/N - gamma*(E[i] + dt*F1/2)
        E[i+1] = E[i] + dt*F2
        
        F1 = gamma*E[i] - delta*I[i]
        F2 = gamma*(E[i] + dt*F1/2) - delta*(I[i] + dt*F1/2)
        I[i+1] = I[i] + dt*F2
        
        F1 = delta*I[i]
        F2 = delta*(I[i] + dt*F1/2)
        R[i+1] = R[i] + dt*F2
        
        D[i+1] = N - S[i+1] - E[i+1] - I[i+1] - R[i+1]
    
    # Returning the arrays
    return S, E, R, I, D

