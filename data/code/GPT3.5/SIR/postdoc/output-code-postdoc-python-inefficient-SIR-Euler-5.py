import numpy as np
import matplotlib.pyplot as plt


def SIR_model(S0, I0, R0, beta, gamma, t_max, delta_t):
    N = S0 + I0 + R0
    t = np.arange(0, t_max, delta_t)
    S = np.zeros(len(t))
    S[0] = S0
    I = np.zeros(len(t))
    I[0] = I0
    R = np.zeros(len(t))
    R[0] = R0

    for i in range(1, len(t)):
        S[i] = S[i-1] - (beta * S[i-1] * I[i-1] / N) * delta_t
        I[i] = I[i-1] + ((beta * S[i-1] * I[i-1] / N) - (gamma * I[i-1])) * delta_t
        R[i] = R[i-1] + (gamma * I[i-1]) * delta_t

    return S, I, R


# Example usage
S0 = 99
I0 = 1
R0 = 0
beta = 0.3
gamma = 0.1
t_max = 100
delta_t = 0.1

S, I, R = SIR_model(S0, I0, R0, beta, gamma, t_max, delta_t)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
