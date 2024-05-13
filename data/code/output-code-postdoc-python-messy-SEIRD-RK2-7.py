import numpy as np
import matplotlib.pyplot as plt


def seird_model(beta, gamma, sigma, delta, N, I0, E0, R0, D0, T):
    # Initialize arrays
    S = np.zeros(T)
    E = np.zeros(T)
    I = np.zeros(T)
    R = np.zeros(T)
    D = np.zeros(T)

    # Set initial conditions
    S[0] = N - I0 - E0 - R0 - D0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    D[0] = D0

    # Step size
    dt = 1

    # Run the simulation
    for t in range(1, T):
        # Compute derivatives
        dSdt = -beta * S[t-1] * I[t-1] / N
        dEdt = beta * S[t-1] * I[t-1] / N - sigma * E[t-1]
        dIdt = sigma * E[t-1] - gamma * I[t-1] - delta * I[t-1]
        dRdt = gamma * I[t-1]
        dDdt = delta * I[t-1]

        # Update variables using RK2 method
        S[t] = S[t-1] + dt * dSdt
        E[t] = E[t-1] + dt * dEdt
        I[t] = I[t-1] + dt * dIdt
        R[t] = R[t-1] + dt * dRdt
        D[t] = D[t-1] + dt * dDdt

    return S, E, I, R, D


# Parameters
beta = 0.2
sigma = 0.1
gamma = 0.05
delta = 0.01
N = 1000
I0 = 10
E0 = 5
R0 = 0
D0 = 0
T = 100

# Run the simulation
S, E, I, R, D = seird_model(beta, gamma, sigma, delta, N, I0, E0, R0, D0, T)

# Plot the results
plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.plot(D, label='Dead')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.legend()
plt.show()
