import numpy as np
import matplotlib.pyplot as plt
import json

def euler_sir_model(si, beta, gamma, n_steps):
    """
    This function simulates the SIR model using Euler's method.

    Parameters:
        si (tuple): the initial susceptible and infected population.
        beta (float): the infection rate.
        gamma (float): the recovery rate.
        n_steps (int): the number of time steps to simulate.

    Returns:
        s (list): the susceptible population at each time step.
        i (list): the infected population at each time step.
        r (list): the recovered population at each time step.
    """
    s, i = si
    r = 0.0

    s_values = [s]
    i_values = [i]
    r_values = [r]

    for _ in range(n_steps):
        ds = -beta * s * i
        di = beta * s * i - gamma * i
        dr = gamma * i

        s += ds
        i += di
        r += dr

        s_values.append(s)
        i_values.append(i)
        r_values.append(r)

    return s_values, i_values, r_values

def plot_sir_model(s, i, r):
    """
    This function plots the SIR model.

    Parameters:
        s (list): the susceptible population at each time step.
        i (list): the infected population at each time step.
        r (list): the recovered population at each time step.

    Returns:
        None
    """
    plt.plot(s, label='Susceptible')
    plt.plot(i, label='Infected')
    plt.plot(r, label='Recovered')
    plt.legend()
    plt.show()

# Testing the function with some values
s, i, r = euler_sir_model((0.99, 0.01), 0.3, 0.1, 100)
plot_sir_model(s, i, r)
