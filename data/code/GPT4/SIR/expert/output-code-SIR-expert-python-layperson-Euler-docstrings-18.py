import numpy as np
import matplotlib.pyplot as plt
import json

def SIR_model(S, I, R, beta, gamma, dt):
    """
    This function computes the SIR model using Euler's method.
    :param S: Susceptible population
    :param I: Infected population
    :param R: Recovered population
    :param beta: Infection rate
    :param gamma: Recovery rate
    :param dt: Time step
    :return: S, I, R for the next time step
    """
    S_new = S - (beta * S * I) * dt
    I_new = I + (beta * S * I - gamma * I) * dt
    R_new = R + (gamma * I) * dt

    return S_new, I_new, R_new

def plot_SIR_model(S, I, R):
    """
    This function plots the SIR model.
    :param S: Susceptible population
    :param I: Infected population
    :param R: Recovered population
    :return: None
    """
    plt.plot(S, label='Susceptible')
    plt.plot(I, label='Infected')
    plt.plot(R, label='Recovered')
    plt.legend()
    plt.show()


# Define initial parameters
S = 900
I = 100
R = 0
beta = 0.3
gamma = 0.1
dt = 0.1
t = np.arange(0, 100, dt)

S_list, I_list, R_list = [S], [I], [R]

# Run the simulation
for _ in t:
    S, I, R = SIR_model(S, I, R, beta, gamma, dt)
    S_list.append(S)
    I_list.append(I)
    R_list.append(R)

# Plot the results
plot_SIR_model(S_list, I_list, R_list)
