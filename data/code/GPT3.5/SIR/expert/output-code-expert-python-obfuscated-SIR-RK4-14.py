import numpy as np
import matplotlib.pyplot as plt


# Function implementing the SIR model
def sir_model(beta, gamma, N, I0, R0, days):
    # Set the initial conditions
    S0 = N - I0 - R0
    # Create arrays to store the values of S, I, and R
    S = np.zeros(days)
    I = np.zeros(days)
    R = np.zeros(days)
    # Set the initial values
    S[0] = S0
    I[0] = I0
    R[0] = R0
    # Loop through each day
    for i in range(1, days):
        # Calculate the derivatives
        dSdt = -beta * S[i-1] * I[i-1] / N
        dIdt = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dRdt = gamma * I[i-1]
        # Calculate the new values
        S[i] = S[i-1] + dSdt
        I[i] = I[i-1] + dIdt
        R[i] = R[i-1] + dRdt
    return S, I, R


# Parameters
beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
days = 100

# Run the model
S, I, R = sir_model(beta, gamma, N, I0, R0, days)

# Plot the results
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()

