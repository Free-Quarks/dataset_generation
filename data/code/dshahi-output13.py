import numpy as np
import matplotlib.pyplot as plt


def seir_model(N, beta, gamma, sigma, days):
    # Initial conditions
    S0 = N - 1
    E0 = 1
    I0 = 0
    R0 = 0
    
    # Create arrays to store the results
    S = np.zeros(days)
    E = np.zeros(days)
    I = np.zeros(days)
    R = np.zeros(days)
    
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    
    # Run the simulation
    for t in range(1, days):
        dS = -beta * S[t-1] * I[t-1] / N
        dE = beta * S[t-1] * I[t-1] / N - sigma * E[t-1]
        dI = sigma * E[t-1] - gamma * I[t-1]
        dR = gamma * I[t-1]
        
        S[t] = S[t-1] + dS
        E[t] = E[t-1] + dE
        I[t] = I[t-1] + dI
        R[t] = R[t-1] + dR
    
    # Return the results
    return S, E, I, R


# Set the parameters
N = 1000
beta = 0.3
sigma = 1/5
gamma = 1/14
days = 100

# Run the model
S, E, I, R = seir_model(N, beta, gamma, sigma, days)

# Plot the results
plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Number of individuals')
plt.legend()
plt.title('SEIR Model')
plt.show()
