import numpy as np
import matplotlib.pyplot as plt


def SIR_euler(S0, I0, R0, beta, gamma, N, t_max, dt):
    t = np.linspace(0, t_max, int(t_max/dt)+1)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = S0
    I[0] = I0
    R[0] = R0
    for i in range(len(t)-1):
        dS = - beta * S[i] * I[i] / N
        dI = beta * S[i] * I[i] / N - gamma * I[i]
        dR = gamma * I[i]
        S[i+1] = S[i] + dt * dS
        I[i+1] = I[i] + dt * dI
        R[i+1] = R[i] + dt * dR
    return S, I, R


# Example usage
S0 = 1000
I0 = 1
R0 = 0
beta = 0.2
gamma = 0.1
N = S0 + I0 + R0
t_max = 100
dt = 0.1

S, I, R = SIR_euler(S0, I0, R0, beta, gamma, N, t_max, dt)

plt.figure()
plt.plot(S, label='S')
plt.plot(I, label='I')
plt.plot(R, label='R')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SIR Model - Euler Method')
plt.show()
