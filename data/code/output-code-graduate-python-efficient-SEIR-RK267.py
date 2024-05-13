import numpy as np


def seir_model(beta, gamma, sigma, population, exposed, infected, recovered, days):
    
    # Initialize arrays
    S = np.zeros(days)
    E = np.zeros(days)
    I = np.zeros(days)
    R = np.zeros(days)

    # Set initial conditions
    S[0] = population - exposed - infected - recovered
    E[0] = exposed
    I[0] = infected
    R[0] = recovered

    # Run the model
    for i in range(1, days):
        dS = -beta * S[i-1] * I[i-1] / population
        dE = beta * S[i-1] * I[i-1] / population - sigma * E[i-1]
        dI = sigma * E[i-1] - gamma * I[i-1]
        dR = gamma * I[i-1]

        S[i] = S[i-1] + dS
        E[i] = E[i-1] + dE
        I[i] = I[i-1] + dI
        R[i] = R[i-1] + dR

    return S, E, I, R


# Example usage
beta = 0.5
gamma = 0.2
sigma = 0.1
population = 1000
exposed = 10
infected = 5
recovered = 0
days = 100

S, E, I, R = seir_model(beta, gamma, sigma, population, exposed, infected, recovered, days)
