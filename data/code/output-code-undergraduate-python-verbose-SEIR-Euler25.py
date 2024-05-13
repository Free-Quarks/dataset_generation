import matplotlib.pyplot as plt
import numpy as np


def seir_model(beta, sigma, gamma, population, initial_infected, num_days):
    # Initialize arrays
    S = np.zeros(num_days)
    E = np.zeros(num_days)
    I = np.zeros(num_days)
    R = np.zeros(num_days)

    # Set initial conditions
    S[0] = population - initial_infected
    E[0] = initial_infected

    # Euler method
    for t in range(num_days-1):
        # Calculate derivatives
        dS_dt = -beta * S[t] * I[t] / population
        dE_dt = beta * S[t] * I[t] / population - sigma * E[t]
        dI_dt = sigma * E[t] - gamma * I[t]
        dR_dt = gamma * I[t]

        # Update variables
        S[t+1] = S[t] + dS_dt
        E[t+1] = E[t] + dE_dt
        I[t+1] = I[t] + dI_dt
        R[t+1] = R[t] + dR_dt

    return S, E, I, R


# Example usage
beta = 0.8
sigma = 0.2
gamma = 0.5
population = 10000
initial_infected = 10
num_days = 100

S, E, I, R = seir_model(beta, sigma, gamma, population, initial_infected, num_days)

# Plotting
plt.plot(range(num_days), S, label='Susceptible')
plt.plot(range(num_days), E, label='Exposed')
plt.plot(range(num_days), I, label='Infected')
plt.plot(range(num_days), R, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Number of Individuals')
plt.title('SEIR Model')
plt.legend()
plt.show()
