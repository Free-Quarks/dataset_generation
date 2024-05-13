import numpy as np
import matplotlib.pyplot as plt


def seir_model(beta, gamma, sigma, S0, E0, I0, R0, N, t):
    # Function to implement the SEIR model
    
    # Initialize arrays to store the values
    S = np.zeros(t)
    E = np.zeros(t)
    I = np.zeros(t)
    R = np.zeros(t)
    
    # Assign initial values
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    
    # Iterate over time steps
    for i in range(1, t):
        # Calculate the new values
        dS = -beta * S[i-1] * I[i-1] / N
        dE = beta * S[i-1] * I[i-1] / N - sigma * E[i-1]
        dI = sigma * E[i-1] - gamma * I[i-1]
        dR = gamma * I[i-1]
        
        # Update the values
        S[i] = S[i-1] + dS
        E[i] = E[i-1] + dE
        I[i] = I[i-1] + dI
        R[i] = R[i-1] + dR
        
    return S, E, I, R


# Define the parameters
beta = 0.5
sigma = 0.1
gamma = 0.05
N = 10000

# Set the initial conditions
S0 = N - 1
E0 = 1
I0 = 0
R0 = 0

# Define the time vector
t = np.linspace(0, 100, num=100)

# Call the SEIR model
S, E, I, R = seir_model(beta, gamma, sigma, S0, E0, I0, R0, N, t)

# Plot the results
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.title('SEIR Model')
plt.legend()
plt.show()
