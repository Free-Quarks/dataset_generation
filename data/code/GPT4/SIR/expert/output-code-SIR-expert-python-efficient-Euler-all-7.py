import numpy as np
import matplotlib.pyplot as plt
import json

def sir_model_euler(S0, I0, R0, beta, gamma, dt, num_steps):
    """
    Simulate the SIR epidemic model using the Euler method.

    Parameters:
    S0, I0, R0: initial fraction of the population that are susceptible, infected, and recovered
    beta: infection rate parameter
    gamma: recovery rate parameter
    dt: time step
    num_steps: number of simulation steps

    Returns:
    S, I, R: fraction of the population that are susceptible, infected, and recovered at each time step
    """

    # Initialize arrays to store the fractions of the population in the S, I, R categories
    S = np.zeros(num_steps + 1)
    I = np.zeros(num_steps + 1)
    R = np.zeros(num_steps + 1)

    # Set initial conditions
    S[0] = S0
    I[0] = I0
    R[0] = R0

    # Simulate the model
    for step in range(num_steps):
        dSdt = -beta * S[step] * I[step]
        dIdt = beta * S[step] * I[step] - gamma * I[step]
        dRdt = gamma * I[step]

        S[step + 1] = S[step] + dt * dSdt
        I[step + 1] = I[step] + dt * dIdt
        R[step + 1] = R[step] + dt * dRdt

    return S, I, R

# Parameters
S0 = 0.99
I0 = 0.01
R0 = 0.00
beta = 0.35
gamma = 0.1
dt = 0.01
num_steps = 1000

# Run simulation
S, I, R = sir_model_euler(S0, I0, R0, beta, gamma, dt, num_steps)

# Plot results
plt.figure()
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.legend()
plt.show()
