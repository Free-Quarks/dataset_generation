import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function implementing the SEIR model

def seir_model(y, t, beta, gamma, sigma):
    S, E, I, R = y
    N = S + E + I + R
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# Function to solve the SEIR model

def solve_seir_model(S0, E0, I0, R0, beta, gamma, sigma, t):
    y0 = S0, E0, I0, R0
    params = beta, gamma, sigma
    solution = odeint(seir_model, y0, t, args=params)
    return solution[:, 0], solution[:, 1], solution[:, 2], solution[:, 3]

# Example usage

# Parameters
beta = 0.2
sigma = 1/5.2
gamma = 1/10

# Initial conditions
S0 = 999
E0 = 1
I0 = 0
R0 = 0

# Time vector
t = np.linspace(0, 100, 100)

# Solve the model
S, E, I, R = solve_seir_model(S0, E0, I0, R0, beta, gamma, sigma, t)

# Plot the results
plt.figure()
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.grid(True)
plt.show()
