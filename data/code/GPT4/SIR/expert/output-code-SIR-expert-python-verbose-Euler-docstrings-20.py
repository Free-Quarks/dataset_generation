import numpy as np
import matplotlib.pyplot as plt
import json

def sir_model_euler(s0, i0, r0, beta, gamma, t):
    """
    Simulate the SIR epidemic model using the Euler's method.

    Parameters:
    s0 (float): Initial proportion of the population that are susceptible
    i0 (float): Initial proportion of the population that are infectious
    r0 (float): Initial proportion of the population that are recovered
    beta (float): Infection rate parameter
    gamma (float): Recovery rate parameter
    t (array): Time points to evaluate the model at

    Returns:
    s (array): Proportion of the population that are susceptible at points in t
    i (array): Proportion of the population that are infectious at points in t
    r (array): Proportion of the population that are recovered at points in t
    """

    s = np.zeros(t.shape)
    i = np.zeros(t.shape)
    r = np.zeros(t.shape)

    s[0] = s0
    i[0] = i0
    r[0] = r0

    dt = t[1] - t[0]

    for n in range(0, len(t)-1):
        s[n+1] = s[n] - beta * s[n] * i[n] * dt
        i[n+1] = i[n] + (beta * s[n] * i[n] - gamma * i[n]) * dt
        r[n+1] = r[n] + gamma * i[n] * dt

    return s, i, r

def plot_sir_model(s, i, r, t):
    """
    Plot the results of the SIR model.

    Parameters:
    s (array): Proportion of the population that are susceptible at points in t
    i (array): Proportion of the population that are infectious at points in t
    r (array): Proportion of the population that are recovered at points in t
    t (array): Time points the model was evaluated at

    Returns:
    None
    """

    plt.figure(figsize=(12,8))
    plt.plot(t, s, label='Susceptible')
    plt.plot(t, i, label='Infectious')
    plt.plot(t, r, label='Recovered')
    plt.legend(loc='best')
    plt.xlabel('Time')
    plt.ylabel('Proportion')
    plt.title('SIR Model')
    plt.grid()
    plt.show()

s0 = 0.99
i0 = 0.01
r0 = 0.0
beta = 0.4
gamma = 0.04
t = np.linspace(0, 100, 10000)

s, i, r = sir_model_euler(s0, i0, r0, beta, gamma, t)

plot_sir_model(s, i, r, t)
