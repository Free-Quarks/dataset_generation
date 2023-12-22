import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def sir_model(t, y, beta, gamma):
    S, I, R = y
    return [-beta*S*I, beta*S*I - gamma*I, gamma*I]

def sir_simulation(beta, gamma, S0, I0, R0, t):
    sol = solve_ivp(sir_model, [0, t], [S0, I0, R0], args=(beta, gamma), dense_output=True)
    return sol

def plot_sir(t, sol):
    S, I, R = sol.y
    plt.figure(figsize=(10, 5))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.legend()
    plt.grid()
    plt.show()

beta = 0.2
gamma = 0.1
S0 = 0.99
I0 = 0.01
R0 = 0.0
t = np.linspace(0, 160, 160)

sol = sir_simulation(beta, gamma, S0, I0, R0, t)
plot_sir(t, sol)
