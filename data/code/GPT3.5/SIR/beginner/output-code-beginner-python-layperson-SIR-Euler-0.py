import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, S0, I0, R0, N, t_max):
    dt = 0.1
    t = np.linspace(0, t_max, int(t_max/dt) + 1)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    S[0] = S0
    I[0] = I0
    R[0] = R0

    for i in range(1, len(t)):
        dS = -beta * S[i-1] * I[i-1] / N
        dI = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dR = gamma * I[i-1]

        S[i] = S[i-1] + dt * dS
        I[i] = I[i-1] + dt * dI
        R[i] = R[i-1] + dt * dR

    return S, I, R


# Example usage
beta = 0.3
gamma = 0.1
S0 = 999
I0 = 1
R0 = 0
N = S0 + I0 + R0
t_max = 100

S, I, R = SIR_model(beta, gamma, S0, I0, R0, N, t_max)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
