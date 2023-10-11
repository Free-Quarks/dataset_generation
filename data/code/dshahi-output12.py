import numpy as np
import matplotlib.pyplot as plt

def seir_model(beta, gamma, sigma, N, I0, R0, num_days):
    S0 = N - I0 - R0
    S = np.zeros(num_days)
    E = np.zeros(num_days)
    I = np.zeros(num_days)
    R = np.zeros(num_days)

    S[0] = S0
    E[0] = I0
    I[0] = R0

    for t in range(num_days-1):
        S[t+1] = S[t] - beta*S[t]*I[t]
        E[t+1] = E[t] + beta*S[t]*I[t] - sigma*E[t]
        I[t+1] = I[t] + sigma*E[t] - gamma*I[t]
        R[t+1] = R[t] + gamma*I[t]

    return S, E, I, R


# Example usage
beta = 0.2
gamma = 0.1
sigma = 0.05
N = 100000
I0 = 10
R0 = 0
num_days = 100

S, E, I, R = seir_model(beta, gamma, sigma, N, I0, R0, num_days)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.show()
