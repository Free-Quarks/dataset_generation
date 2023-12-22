import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import json

# Define the SIR model
def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Initial conditions
S0, I0, R0 = 999, 1, 0  # initial population
beta, gamma = 0.2, 0.1  # infection rate, recovery rate
t = np.linspace(0, 100, 100)  # time grid

# Solve the SIR model
solution = odeint(SIR_model, [S0, I0, R0], t, args=(beta, gamma))
solution = np.array(solution)

# Plot the data on three separate curves for S(t), I(t) and R(t)
plt.figure(figsize=[6, 4])
plt.plot(t, solution[:, 0], label="S(t)")
plt.plot(t, solution[:, 1], label="I(t)")
plt.plot(t, solution[:, 2], label="R(t)")
plt.grid()
plt.legend()
plt.xlabel("Time")
plt.ylabel("Proportions")
plt.title("SIR model")
plt.show()
