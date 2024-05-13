import numpy as np
from scipy.integrate import odeint


# The SEIR model

# The differential equations for the SEIR model


def seir_model(y, t, N, beta, gamma, sigma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


# The main function that simulates and plots the SEIR model


def simulate_seir_model(N, E0, I0, R0, beta, gamma, sigma, t_max):
    # Initial conditions
    S0 = N - E0 - I0 - R0
    y0 = S0, E0, I0, R0

    # Create a time vector
    t = np.linspace(0, t_max, t_max + 1)

    # Integrate the SEIR equations over the time grid
    result = odeint(seir_model, y0, t, args=(N, beta, gamma, sigma))
    S, E, I, R = result.T

    # Plot the results
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SEIR Model Simulation')
    plt.legend()
    plt.show()


# Example usage


N = 1000
E0 = 1
I0 = 1
R0 = 0
beta = 0.2
gamma = 0.1
sigma = 0.05
t_max = 200

simulate_seir_model(N, E0, I0, R0, beta, gamma, sigma, t_max)
