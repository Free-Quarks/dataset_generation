import numpy as np
import matplotlib.pyplot as plt


def SIR_model(S0, I0, R0, beta, gamma, N, days):
    S = [S0]
    I = [I0]
    R = [R0]
    for day in range(days):
        dS = -beta * S[-1] * I[-1] / N
        dI = beta * S[-1] * I[-1] / N - gamma * I[-1]
        dR = gamma * I[-1]
        S.append(S[-1] + dS)
        I.append(I[-1] + dI)
        R.append(R[-1] + dR)
    return S, I, R


# Define the parameters
N = 1000  # Total population
beta = 0.2  # Infection rate
gamma = 0.1  # Recovery rate
S0 = N - 1  # Initial susceptible
I0 = 1  # Initial infected
R0 = 0  # Initial recovered

# Simulate the model
days = 100
S, I, R = SIR_model(S0, I0, R0, beta, gamma, N, days)

# Plot the results
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Number of Individuals')
plt.title('SIR Model')
plt.legend(loc='upper right')
plt.show()

