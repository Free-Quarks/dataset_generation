import numpy as np
import matplotlib.pyplot as plt

def sir_model(N, I0, R0, beta, gamma, t_max):
    # Initialize arrays to store the results
    S = np.zeros(t_max)
    I = np.zeros(t_max)
    R = np.zeros(t_max)

    # Set initial conditions
    S[0] = N - I0 - R0
    I[0] = I0
    R[0] = R0

    # Euler's method
    for t in range(1, t_max):
        dS = -beta * S[t-1] * I[t-1] / N
        dI = beta * S[t-1] * I[t-1] / N - gamma * I[t-1]
        dR = gamma * I[t-1]
        S[t] = S[t-1] + dS
        I[t] = I[t-1] + dI
        R[t] = R[t-1] + dR

    return S, I, R

# Set model parameters
N = 1000
I0 = 1
R0 = 0
beta = 0.2
gamma = 0.1
t_max = 100

# Simulate SIR model
S, I, R = sir_model(N, I0, R0, beta, gamma, t_max)

# Plot results
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SIR Model Simulation')
plt.show()
