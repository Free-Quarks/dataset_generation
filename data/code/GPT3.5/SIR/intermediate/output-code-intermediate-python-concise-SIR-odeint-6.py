import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def plot_SIR_model(S, I, R, t):
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.legend()
    plt.show()

# Initial conditions
S0 = 1000
I0 = 1
R0 = 0

# Parameters
beta = 0.2
gamma = 0.1

# Time vector
t = np.linspace(0, 100, 100)

# Solve the differential equations
solution = odeint(SIR_model, [S0, I0, R0], t, args=(beta, gamma))

# Extracting the population compartments
S = solution[:, 0]
I = solution[:, 1]
R = solution[:, 2]

# Plotting the results
plot_SIR_model(S, I, R, t)
