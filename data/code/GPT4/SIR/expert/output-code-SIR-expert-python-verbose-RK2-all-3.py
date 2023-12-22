import numpy as np
import matplotlib.pyplot as plt
import json

def SIR_model(y, t, beta, gamma):
    """
    SIR model dynamics.

    Parameters:
    y (array_like): List of variables in the SIR model (Susceptible, Infected, and Recovered)
    t (float): Time
    beta (float): Contact rate
    gamma (float): Recovery rate

    Returns:
    ds: Change in the susceptible population
    di: Change in the infected population
    dr: Change in the recovered population
    """

    S, I, R = y

    ds = -beta * S * I
    di = beta * S * I - gamma * I
    dr = gamma * I

    return np.array([ds, di, dr])

def RK2(y, t, dt, model, beta, gamma):
    """
    RK2 method for numerical integration.

    Parameters:
    y (array_like): List of variables in the SIR model (Susceptible, Infected, and Recovered)
    t (float): Time
    dt (float): Time step
    model (function): Model dynamics function
    beta (float): Contact rate
    gamma (float): Recovery rate

    Returns:
    y (array_like): Updated variables
    """

    k1 = model(y, t, beta, gamma)
    k2 = model(y + dt * k1, t + dt, beta, gamma)

    y = y + dt * (k1 + k2) / 2

    return y

# Initial conditions
S0 = 0.99
I0 = 0.01
R0 = 0.0

y0 = np.array([S0, I0, R0])

# SIR model parameters
beta = 0.4
gamma = 0.1

# Time grid for the simulation
t = np.linspace(0, 200, 2001)
dt = t[1] - t[0]

# Array to store the solution
solution = np.empty((len(t), len(y0)))
solution[0] = y0

# Main loop for RK2 integration
for i in range(1, len(t)):
    solution[i] = RK2(solution[i - 1], t[i - 1], dt, SIR_model, beta, gamma)

# Plotting the solution
plt.figure(figsize=(9, 6))
plt.plot(t, solution[:, 0], label='Susceptible')
plt.plot(t, solution[:, 1], label='Infected')
plt.plot(t, solution[:, 2], label='Recovered')
plt.xlabel('Time')
plt.ylabel('Fraction of population')
plt.legend()
plt.show()
