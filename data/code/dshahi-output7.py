import numpy as np
import matplotlib.pyplot as plt


def serid_model(beta, gamma, population, initial_infected, days):
    # Initialize arrays
    S = np.zeros(days)
    E = np.zeros(days)
    R = np.zeros(days)
    I = np.zeros(days)
    D = np.zeros(days)
    N = population
    S[0] = N - initial_infected
    E[0] = initial_infected
    I[0] = initial_infected

    # Run the simulation
    for t in range(1, days):
        S[t] = S[t-1] - beta * I[t-1] * S[t-1] / N
        E[t] = E[t-1] + beta * I[t-1] * S[t-1] / N - gamma * E[t-1]
        I[t] = I[t-1] + gamma * E[t-1] - gamma * I[t-1]
        R[t] = R[t-1] + gamma * I[t-1]
        D[t] = D[t-1] + gamma * I[t-1]

    # Plot the results
    plt.plot(range(days), S, label='Susceptible')
    plt.plot(range(days), E, label='Exposed')
    plt.plot(range(days), I, label='Infected')
    plt.plot(range(days), R, label='Recovered')
    plt.plot(range(days), D, label='Deaths')
    plt.xlabel('Days')
    plt.ylabel('Number of individuals')
    plt.title('SERID Model Simulation')
    plt.legend()
    plt.show()


serid_model(0.2, 0.1, 100000, 10, 100)
