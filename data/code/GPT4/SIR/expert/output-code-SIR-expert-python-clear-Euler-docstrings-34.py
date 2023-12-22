import numpy as np
import matplotlib.pyplot as plt

def SIR_model_euler(S0, I0, R0, beta, gamma, t):
    """
    Function implementing SIR model using Euler's method

    Parameters:
    S0 : Initial susceptible population
    I0 : Initial infected population
    R0 : Initial recovered population
    beta : Contact rate
    gamma : Recovery rate
    t : Time
    
    Returns:
    S, I, R : arrays containing the susceptible, infected and recovered population at each timestep
    """

    # Total population
    N = S0 + I0 + R0

    # Number of timesteps
    timesteps = len(t)

    # Initializing arrays
    S, I, R = np.zeros(timesteps), np.zeros(timesteps), np.zeros(timesteps)
    S[0], I[0], R[0] = S0, I0, R0

    # Time step size
    dt = t[1] - t[0]

    # Euler's method
    for i in range(timesteps - 1):
        S[i + 1] = S[i] - dt * beta * S[i] * I[i] / N
        I[i + 1] = I[i] + dt * (beta * S[i] * I[i] / N - gamma * I[i])
        R[i + 1] = R[i] + dt * gamma * I[i]

    return S, I, R
