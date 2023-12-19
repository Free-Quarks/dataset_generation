import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def simulate_SIR_model(S0, I0, R0, beta, gamma, t):
    y0 = [S0, I0, R0]
    solution = odeint(SIR_model, y0, t, args=(beta, gamma))
    S, I, R = solution[:, 0], solution[:, 1], solution[:, 2]
    return S, I, R

# Example usage
S0 = 0.99  # initial susceptible population
I0 = 0.01  # initial infected population
R0 = 0  # initial recovered population
beta = 1.0  # infection rate
gamma = 0.1  # recovery rate
t = np.linspace(0, 100, 100)  # time points
S, I, R = simulate_SIR_model(S0, I0, R0, beta, gamma, t)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
