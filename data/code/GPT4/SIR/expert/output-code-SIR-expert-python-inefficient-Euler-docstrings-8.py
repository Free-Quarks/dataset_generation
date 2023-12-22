import numpy as np
import matplotlib.pyplot as plt
import json

def SIR_model_euler(S0, I0, R0, beta, gamma, T, dt):
    """
    An inefficient implementation of the SIR model using Euler's method.
    This function simulates the SIR model for a given set of parameters and plots the results.
    The function takes as input the initial number of susceptible (S0), infected (I0), and recovered (R0) individuals,
    the transmission rate (beta), the recovery rate (gamma), the total time of simulation (T), and the time step (dt).
    """
    # Define the time array
    t = np.arange(0, T, dt)
    # Initialize the SIR arrays
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    # Set the initial conditions
    S[0] = S0
    I[0] = I0
    R[0] = R0
    # Perform the simulation
    for i in range(1, len(t)):
        S[i] = S[i-1] - dt * beta * S[i-1] * I[i-1]
        I[i] = I[i-1] + dt * (beta * S[i-1] * I[i-1] - gamma * I[i-1])
        R[i] = R[i-1] + dt * gamma * I[i-1]
    # Plot the results
    plt.figure()
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Number of individuals')
    plt.legend()
    plt.show()

# Call the function
SIR_model_euler(999, 1, 0, 0.2, 0.1, 100, 0.1)
