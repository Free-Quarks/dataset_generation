import numpy as np
import matplotlib.pyplot as plt

# Function for the SEIR model

def seir_model(beta, sigma, gamma, N, I0, E0, R0, T):
    # Initialize arrays
    t = np.linspace(0, T, T+1)
    S = np.zeros(T+1)
    E = np.zeros(T+1)
    I = np.zeros(T+1)
    R = np.zeros(T+1)

    # Set initial conditions
    S[0] = N - E0 - I0 - R0
    E[0] = E0
    I[0] = I0
    R[0] = R0

    # Euler's method
    for i in range(T):
        S[i+1] = S[i] - beta*S[i]*I[i]/N
        E[i+1] = E[i] + beta*S[i]*I[i]/N - sigma*E[i]
        I[i+1] = I[i] + sigma*E[i] - gamma*I[i]
        R[i+1] = R[i] + gamma*I[i]

    # Return arrays
    return t, S, E, I, R

# Example usage

# Parameters
beta = 0.5
sigma = 0.1
gamma = 0.05
N = 1000
I0 = 10
E0 = 5
R0 = 0
T = 100

# Run the model
t, S, E, I, R = seir_model(beta, sigma, gamma, N, I0, E0, R0, T)

# Plot the results
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.show()
