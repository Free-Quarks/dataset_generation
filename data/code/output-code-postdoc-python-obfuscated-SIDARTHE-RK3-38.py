#!/usr/bin/env python

import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt


# Function to define the SIDARTHE model

def sidarthe_model(y, t, beta, sigma, alpha, gamma, delta, epsilon, N):
    S, I, D, A, R, T, H, E = y
    dSdt = -beta * S * (I + alpha * A) / N
    dIdt = beta * S * (I + alpha * A) / N - (sigma + delta) * I
    dDdt = delta * I
    dAdt = sigma * I - (gamma + epsilon) * A
    dRdt = gamma * A
    dTdt = epsilon * A
    dHdt = alpha * delta * I
    dEdt = sigma * I
    return [dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt]


# Function to run the simulation and plot the results


def run_simulation():
    N = 1000  # Total population
    n_days = 100  # Number of days
    t = np.linspace(0, n_days, n_days)
    S0, I0, D0, A0, R0, T0, H0, E0 = N-1, 1, 0, 0, 0, 0, 0, 0  # Initial conditions
    beta, sigma, alpha, gamma, delta, epsilon = 0.2, 0.1, 0.05, 0.1, 0.01, 0.02  # Parameters
    y0 = [S0, I0, D0, A0, R0, T0, H0, E0]  # Initial condition vector
    result = odeint(sidarthe_model, y0, t, args=(beta, sigma, alpha, gamma, delta, epsilon, N))
    S, I, D, A, R, T, H, E = result.T
    plt.plot(t, S, 'b-', label='Susceptible')
    plt.plot(t, I, 'r-', label='Infected')
    plt.plot(t, D, 'g-', label='Deceased')
    plt.plot(t, A, 'y-', label='Asymptomatic')
    plt.plot(t, R, 'm-', label='Recovered')
    plt.plot(t, T, 'c-', label='Tested')
    plt.plot(t, H, 'k-', label='Hospitalized')
    plt.plot(t, E, 'orange', label='Exposed')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SIDARTHE Model')
    plt.legend(loc='best')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    run_simulation()
