import numpy as np
import matplotlib.pyplot as plt

def seir_model(beta, gamma, sigma, N, I0, R0, days):
    S0 = N - I0 - R0
    E0 = 0
    t = np.linspace(0, days, days)
    S = np.zeros(days)
    E = np.zeros(days)
    I = np.zeros(days)
    R = np.zeros(days)
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    dt = days / (days-1)
    for i in range(1, days):
        dSdt = -beta * S[i-1] * I[i-1] / N
        dEdt = beta * S[i-1] * I[i-1] / N - sigma * E[i-1]
        dIdt = sigma * E[i-1] - gamma * I[i-1]
        dRdt = gamma * I[i-1]
        S[i] = S[i-1] + dSdt * dt
        E[i] = E[i-1] + dEdt * dt
        I[i] = I[i-1] + dIdt * dt
        R[i] = R[i-1] + dRdt * dt
    return S, E, I, R


# Example usage
beta = 0.2
gamma = 0.1
sigma = 0.4
N = 1000
I0 = 1
R0 = 0
days = 100

S, E, I, R = seir_model(beta, gamma, sigma, N, I0, R0, days)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.legend()
plt.title('SEIR Model')
plt.show()
