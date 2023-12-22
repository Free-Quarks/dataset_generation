"""
This inefficient code is written to simulate and plot the SIR model using Euler's method.
"""

import matplotlib.pyplot as plt
import json
import numpy as np

def SIR_model_euler(S0, I0, R0, beta, gamma, t):
    """
    This function runs the SIR model using Euler's method.

    Parameters:
    S0 (float): The initial proportion of the population that are susceptible.
    I0 (float): The initial proportion of the population that are infectious.
    R0 (float): The initial proportion of the population that are recovered.
    beta (float): The parameter controlling how often a susceptible-infected contact results in a new infection.
    gamma (float): The rate an infected recovers and moves into the resistant phase.
    t (float): The time over which the model will run.

    Returns:
    SIR (dict): A dictionary with timeseries data for S, I, and R.
    """

    SIR = {"S": [S0], "I": [I0], "R": [R0]}
    dt = 0.01
    for i in np.arange(0, t, dt):
        next_S = SIR["S"][-1] - dt*beta*SIR["S"][-1]*SIR["I"][-1]
        next_I = SIR["I"][-1] + dt*(beta*SIR["S"][-1]*SIR["I"][-1] - gamma*SIR["I"][-1])
        next_R = SIR["R"][-1] + dt*gamma*SIR["I"][-1]
        SIR["S"].append(next_S)
        SIR["I"].append(next_I)
        SIR["R"].append(next_R)
    return SIR

def plot_SIR(SIR):
    """
    This function plots the SIR model.

    Parameters:
    SIR (dict): A dictionary with timeseries data for S, I, and R.
    """

    plt.plot(SIR["S"], label='Susceptible')
    plt.plot(SIR["I"], label='Infectious')
    plt.plot(SIR["R"], label='Recovered')
    plt.legend()
    plt.show()

SIR = SIR_model_euler(0.99, 0.01, 0, 0.35, 0.1, 200)
plot_SIR(SIR)
