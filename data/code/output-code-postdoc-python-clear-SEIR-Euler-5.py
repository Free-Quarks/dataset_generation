import numpy as np
import matplotlib.pyplot as plt


def seir_model(N, beta, gamma, sigma, initial_infected, days):
    # Initialize arrays
    S = np.zeros(days)
    E = np.zeros(days)
    I = np.zeros(days)
    R = np.zeros(days)
    S[0] = N - initial_infected
    E[0] = initial_infected

    # Euler's method
    for t in range(days-1):
        S[t+1] = S[t] - beta * S[t] * I[t] / N
        E[t+1] = E[t] + beta * S[t] * I[t] / N - sigma * E[t]
        I[t+1] = I[t] + sigma * E[t] - gamma * I[t]
        R[t+1] = R[t] + gamma * I[t]

    return S, E, I, R


# Example usage
if __name__ == '__main__':
    N = 10000
    beta = 0.2
    gamma = 0.1
    sigma = 0.05
    initial_infected = 10
    days = 100

    S, E, I, R = seir_model(N, beta, gamma, sigma, initial_infected, days)

    # Plotting
    plt.plot(S, label='Susceptible')
    plt.plot(E, label='Exposed')
    plt.plot(I, label='Infected')
    plt.plot(R, label='Recovered')
    plt.xlabel('Days')
    plt.ylabel('Number of individuals')
    plt.title('SEIR Model')
    plt.legend()
    plt.show()
