import numpy as np
import matplotlib.pyplot as plt

def seir_model(beta, gamma, sigma, N, I0, E0, R0, T):
    # Compute the number of time steps
    num_steps = len(T)

    # Create arrays to store the number of individuals in each compartment
    S = np.zeros(num_steps)
    E = np.zeros(num_steps)
    I = np.zeros(num_steps)
    R = np.zeros(num_steps)

    # Set the initial conditions
    S[0] = N - I0 - E0 - R0
    E[0] = E0
    I[0] = I0
    R[0] = R0

    # Perform Euler's method to simulate the model
    for t in range(1, num_steps):
        dS = -beta * S[t-1] * I[t-1] / N
        dE = beta * S[t-1] * I[t-1] / N - sigma * E[t-1]
        dI = sigma * E[t-1] - gamma * I[t-1]
        dR = gamma * I[t-1]

        S[t] = S[t-1] + dS
        E[t] = E[t-1] + dE
        I[t] = I[t-1] + dI
        R[t] = R[t-1] + dR

    # Return the arrays of compartment sizes
    return S, E, I, R

# Example usage
beta = 0.2
gamma = 0.1
sigma = 0.5
N = 1000
I0 = 10
E0 = 5
R0 = 0
T = np.arange(0, 100, 1)
S, E, I, R = seir_model(beta, gamma, sigma, N, I0, E0, R0, T)

plt.plot(T, S, label='Susceptible')
plt.plot(T, E, label='Exposed')
plt.plot(T, I, label='Infected')
plt.plot(T, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.legend()
plt.show()
