import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function to define the system of ODEs

def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Function to simulate and plot the SIR model

def simulate_SIR_model(S0, I0, R0, beta, gamma, T):
    # Create the time points
    t = np.linspace(0, T, T+1)
    # Set the initial conditions
    y0 = [S0, I0, R0]
    # Solve the ODEs
    solution = odeint(SIR_model, y0, t, args=(beta, gamma))
    # Extract the individual compartments
    S = solution[:, 0]
    I = solution[:, 1]
    R = solution[:, 2]
    # Plot the results
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()

# Example usage

S0 = 1000
I0 = 1
R0 = 0
beta = 0.2
gamma = 0.1
T = 100

simulate_SIR_model(S0, I0, R0, beta, gamma, T)
