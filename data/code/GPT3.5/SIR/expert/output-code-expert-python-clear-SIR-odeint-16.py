import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function that defines the SIR model

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Function to solve the SIR model

def solve_sir_model(S0, I0, R0, beta, gamma, t_max):
    t = np.linspace(0, t_max, t_max + 1)
    y0 = S0, I0, R0
    solution = odeint(sir_model, y0, t, args=(beta, gamma))
    return solution

# Parameters

S0 = 999
I0 = 1
R0 = 0
beta = 0.2
gamma = 0.1
t_max = 100

# Solve the SIR model

solution = solve_sir_model(S0, I0, R0, beta, gamma, t_max)

# Plotting the results

plt.plot(solution[:, 0], label='S')
plt.plot(solution[:, 1], label='I')
plt.plot(solution[:, 2], label='R')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()

