import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function that returns the derivatives
def serid(y, t, beta, gamma):
    S, E, R, I, D = y
    N = S + E + R + I + D
    dSdt = -beta * (I + E) * S / N
    dEdt = beta * (I + E) * S / N - gamma * E
    dRdt = gamma * E
    dIdt = gamma * I - alpha * I
    dDdt = alpha * I
    return dSdt, dEdt, dRdt, dIdt, dDdt

# Function to solve the ODE system
def solve_ode(beta, gamma, alpha, N, E0, I0, D0, t):
    S0 = N - (E0 + I0 + D0)
    y0 = S0, E0, R0, I0, D0
    solution = odeint(serid, y0, t, args=(beta, gamma, alpha))
    return solution[:, 0], solution[:, 1], solution[:, 2], solution[:, 3], solution[:, 4]

# Define parameters
beta = 0.2
gamma = 0.1
alpha = 0.05
N = 1000
E0 = 10
I0 = 5
D0 = 0

# Define time points
t = np.linspace(0, 100, 100)

# Solve the ODE system
S, E, R, I, D = solve_ode(beta, gamma, alpha, N, E0, I0, D0, t)

# Plot the results
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, R, label='Recovered')
plt.plot(t, I, label='Infected')
plt.plot(t, D, label='Dead')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
