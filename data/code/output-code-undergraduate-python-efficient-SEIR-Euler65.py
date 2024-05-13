import numpy as np
import matplotlib.pyplot as plt

# Function to implement the SEIR model using Euler's method

def seir_model(beta, gamma, sigma, N, I0, E0, R0, T):
    # Initialize arrays to store the values of each compartment
    S = np.zeros(T)
    E = np.zeros(T)
    I = np.zeros(T)
    R = np.zeros(T)

    # Set initial values
    S[0] = N - I0 - E0 - R0
    E[0] = E0
    I[0] = I0
    R[0] = R0

    # Iterate over time steps
    for t in range(1, T):
        # Calculate the derivatives
        dSdt = -beta * S[t-1] * I[t-1] / N
        dEdt = beta * S[t-1] * I[t-1] / N - sigma * E[t-1]
        dIdt = sigma * E[t-1] - gamma * I[t-1]
        dRdt = gamma * I[t-1]

        # Update the values using Euler's method
        S[t] = S[t-1] + dSdt
        E[t] = E[t-1] + dEdt
        I[t] = I[t-1] + dIdt
        R[t] = R[t-1] + dRdt

    return S, E, I, R

# Parameters
beta = 0.2
gamma = 0.1
sigma = 0.1
N = 1000
I0 = 1
E0 = 0
R0 = 0
T = 100

# Call the function
S, E, I, R = seir_model(beta, gamma, sigma, N, I0, E0, R0, T)

# Plot the results
plt.plot(range(T), S, label='Susceptible')
plt.plot(range(T), E, label='Exposed')
plt.plot(range(T), I, label='Infected')
plt.plot(range(T), R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.title('SEIR Model Simulation')
plt.legend()
plt.show()
