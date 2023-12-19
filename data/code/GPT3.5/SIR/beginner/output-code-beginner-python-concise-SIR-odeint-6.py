# Import the required libraries
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the SIR model

# Function to define the SIR model

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Define the initial conditions
initial_conditions = [0.99, 0.01, 0.0]

# Define the time points
t = np.linspace(0, 100, 100)

# Define the parameters
beta = 0.3
gamma = 0.1

# Solve the SIR model
solution = odeint(sir_model, initial_conditions, t, args=(beta, gamma))

# Plot the results
plt.plot(t, solution[:, 0], label='Susceptible')
plt.plot(t, solution[:, 1], label='Infected')
plt.plot(t, solution[:, 2], label='Recovered')
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.title('SIR Model')
plt.legend()
plt.show()
