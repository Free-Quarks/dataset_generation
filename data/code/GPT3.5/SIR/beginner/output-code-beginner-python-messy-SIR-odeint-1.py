import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function implementing the SIR model
def sir_model(y, t, params):
    S, I, R = y
    beta, gamma = params
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Function to solve the differential equation

def solve_sir_model(s0, i0, r0, beta, gamma, t_max):
    t = np.linspace(0, t_max, num=t_max+1)
    y0 = [s0, i0, r0]
    params = [beta, gamma]
    solution = odeint(sir_model, y0, t, args=(params,))
    return t, solution[:, 0], solution[:, 1], solution[:, 2]

# Set initial conditions and parameters
s0 = 999
i0 = 1
r0 = 0
beta = 0.3
gamma = 0.1
t_max = 100

# Solve the SIR model
t, S, I, R = solve_sir_model(s0, i0, r0, beta, gamma, t_max)

# Plot the results
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
