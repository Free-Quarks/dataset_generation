import numpy as np
import matplotlib.pyplot as plt


def SIR_model(S0, I0, R0, beta, gamma, timesteps):
    N = S0 + I0 + R0
    S = np.zeros(timesteps)
    I = np.zeros(timesteps)
    R = np.zeros(timesteps)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    dt = 1
    for t in range(1, timesteps):
        dS = -beta * S[t-1] * I[t-1] / N
        dI = beta * S[t-1] * I[t-1] / N - gamma * I[t-1]
        dR = gamma * I[t-1]
        S[t] = S[t-1] + dt * dS
        I[t] = I[t-1] + dt * dI
        R[t] = R[t-1] + dt * dR
    return S, I, R


# Example usage
S0 = 999
I0 = 1
R0 = 0
beta = 0.3
gamma = 0.1
timesteps = 100

S, I, R = SIR_model(S0, I0, R0, beta, gamma, timesteps)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.legend()
plt.title('SIR Model')
plt.show()
