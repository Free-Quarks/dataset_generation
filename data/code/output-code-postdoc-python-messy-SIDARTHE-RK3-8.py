import numpy as np
import matplotlib.pyplot as plt


def SIDARTHE_model(beta, sigma, alpha, rho, theta, eta, population, I0, D0, E0, A0, R0, T0, H0, S0, N, days):
    # Initialize arrays
    S = np.zeros(days)
    I = np.zeros(days)
    D = np.zeros(days)
    A = np.zeros(days)
    R = np.zeros(days)
    T = np.zeros(days)
    H = np.zeros(days)
    E = np.zeros(days)

    # Set initial conditions
    S[0] = S0
    I[0] = I0
    D[0] = D0
    A[0] = A0
    R[0] = R0
    T[0] = T0
    H[0] = H0
    E[0] = E0

    for t in range(1, days):
        # Compute derivatives
        dSdt = -beta * S[t-1] * (I[t-1] + alpha * A[t-1]) / N
        dEdt = beta * S[t-1] * (I[t-1] + alpha * A[t-1]) / N - sigma * E[t-1]
        dIdt = sigma * (1 - rho) * E[t-1] - (1 - theta) * (1 - eta) * I[t-1] - theta * (1 - eta) * I[t-1]
        dAdt = sigma * rho * E[t-1] - (1 - theta) * A[t-1] - theta * A[t-1]
        dRdt = (1 - theta) * (1 - eta) * I[t-1] + (1 - theta) * A[t-1]
        dTdt = theta * (1 - eta) * I[t-1] + theta * A[t-1]
        dHdt = theta * eta * I[t-1] + theta * eta * A[t-1]
        dDdt = theta * (1 - eta) * I[t-1] + theta * A[t-1]

        # Update variables
        S[t] = S[t-1] + dSdt
        E[t] = E[t-1] + dEdt
        I[t] = I[t-1] + dIdt
        A[t] = A[t-1] + dAdt
        R[t] = R[t-1] + dRdt
        T[t] = T[t-1] + dTdt
        H[t] = H[t-1] + dHdt
        D[t] = D[t-1] + dDdt

    return S, E, I, A, R, T, H, D


# Set parameters
beta = 0.4
sigma = 1/5
alpha = 0.6
rho = 0.2
theta = 1/9
eta = 0.4
population = 100000
I0 = 1
D0 = 0
E0 = 0
A0 = 0
R0 = 0
T0 = 0
H0 = 0
S0 = population - I0 - D0 - E0 - A0 - R0 - T0 - H0
N = population

# Set number of days
days = 100

# Run SIDARTHE model
S, E, I, A, R, T, H, D = SIDARTHE_model(beta, sigma, alpha, rho, theta, eta, population, I0, D0, E0, A0, R0, T0, H0, S0, N, days)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(range(days), S, label='Susceptible')
plt.plot(range(days), E, label='Exposed')
plt.plot(range(days), I, label='Infected')
plt.plot(range(days), A, label='Asymptomatic')
plt.plot(range(days), R, label='Recovered')
plt.plot(range(days), T, label='Treated')
plt.plot(range(days), H, label='Hospitalized')
plt.plot(range(days), D, label='Deceased')
plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SIDARTHE Model')
plt.legend()
plt.show()
