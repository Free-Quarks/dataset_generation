import numpy as np
import matplotlib.pyplot as plt


def SIR_RK2(N, I0, R0, beta, gamma, t_max, dt):
    S0 = N - I0 - R0
    t = np.linspace(0, t_max, int(t_max/dt) + 1)
    S = np.zeros(t.shape)
    I = np.zeros(t.shape)
    R = np.zeros(t.shape)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    for i in range(len(t)-1):
        k1 = -beta * S[i] * I[i] / N
        k2 = -beta * (S[i] + dt/2*k1) * (I[i] + dt/2*k1) / N
        l1 = beta * S[i] * I[i] / N - gamma * I[i]
        l2 = beta * (S[i] + dt/2*k1) * (I[i] + dt/2*k1) / N - gamma * (I[i] + dt/2*l1)
        S[i+1] = S[i] + dt*k2
        I[i+1] = I[i] + dt*l2
        R[i+1] = R[i] + dt*gamma*(I[i]+dt/2*l1)
    return S, I, R


N = 1000
I0 = 1
R0 = 0
beta = 0.3
gamma = 0.1
t_max = 100
dt = 0.1

S, I, R = SIR_RK2(N, I0, R0, beta, gamma, t_max, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()

