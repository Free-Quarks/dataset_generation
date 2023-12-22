import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def SIR_dynamics(t, y, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def plot_SIR_dynamics(t, sol):
    plt.figure(figsize=(6,4))
    plt.plot(t, sol[:, 0], color='blue', label='Susceptible')
    plt.plot(t, sol[:, 1], color='red', label='Infected')
    plt.plot(t, sol[:, 2], color='green', label='Recovered')
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Numbers", fontsize=14)
    plt.legend(loc='best')
    plt.grid()
    plt.show()

S0, I0, R0 = 0.99, 0.01, 0.0
beta, gamma = 1.0, 0.5
t = np.linspace(0, 100, 10000)

sol = solve_ivp(SIR_dynamics, [0, 100], [S0, I0, R0], args=(beta, gamma), dense_output=True)
sol = sol.sol(t).T

plot_SIR_dynamics(t, sol)
