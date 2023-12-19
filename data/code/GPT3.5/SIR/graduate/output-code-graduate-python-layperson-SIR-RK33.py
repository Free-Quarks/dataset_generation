import numpy as np
import matplotlib.pyplot as plt


def SIR_model(N, beta, gamma, I0, R0, t_max, dt):
    S0 = N - I0 - R0
    t = np.linspace(0, t_max, int(t_max/dt)+1)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
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


N = 1000
beta = 0.3
gamma = 0.1
I0 = 1
R0 = 0
t_max = 100
dt = 0.1

S, I, R = SIR_model(N, beta, gamma, I0, R0, t_max, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SIR Model')
plt.show()
