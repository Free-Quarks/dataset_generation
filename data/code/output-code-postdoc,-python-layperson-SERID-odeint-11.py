import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function that defines the differential equations of the SEIR model

def seir_model(y, t, N, beta, gamma, sigma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# Function to run the simulation and plot the results

def run_seir_model(N, beta, gamma, sigma, E0, I0, R0, days):
    # Initial conditions
    S0 = N - E0 - I0 - R0
    # Time points
    t = np.linspace(0, days, days)
    # Initial conditions vector
    y0 = S0, E0, I0, R0
    # Integrate the SEIR equations over the time grid
    result = odeint(seir_model, y0, t, args=(N, beta, gamma, sigma))
    S, E, I, R = result.T
    # Plot the results
    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, E, 'y', label='Exposed')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of individuals')
    plt.legend()
    plt.show()

# Example usage
run_seir_model(1000, 0.2, 0.1, 0.01, 1, 1, 0, 100)
