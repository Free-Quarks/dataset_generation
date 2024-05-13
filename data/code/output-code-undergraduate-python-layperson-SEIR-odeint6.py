import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function to define the SEIR model


def seir_model(y, t, N, beta, gamma, sigma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


# Function to simulate and plot the SEIR model


def simulate_seir_model(N, E0, I0, R0, beta, gamma, sigma, days):
    S0 = N - E0 - I0 - R0
    t = np.linspace(0, days, days)
    y0 = S0, E0, I0, R0
    result = odeint(seir_model, y0, t, args=(N, beta, gamma, sigma))
    S, E, I, R = result.T
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(t, S, 'b', label='Susceptible')
    ax.plot(t, E, 'y', label='Exposed')
    ax.plot(t, I, 'r', label='Infected')
    ax.plot(t, R, 'g', label='Recovered')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Number of individuals')
    ax.set_title('SEIR Model')
    ax.legend()
    plt.show()


# Example usage


simulate_seir_model(N=1000, E0=10, I0=1, R0=0, beta=0.3, gamma=0.1, sigma=0.2, days=100)
