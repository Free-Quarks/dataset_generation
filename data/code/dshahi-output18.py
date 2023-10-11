import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(beta, gamma, S0, I0, R0, t_end, dt):
    N = S0 + I0 + R0
    t = np.arange(0, t_end, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = S0
    I[0] = I0
    R[0] = R0
    for i in range(1, len(t)):
        k1 = -beta * S[i-1] * I[i-1] / N
        k2 = -beta * (S[i-1] + 0.5*dt*k1) * (I[i-1] + 0.5*dt*k1) / N
        k3 = -beta * (S[i-1] + 0.75*dt*k2) * (I[i-1] + 0.75*dt*k2) / N
        S[i] = S[i-1] + dt * (k1 + 2*k2 + 2*k3) / 6
        I[i] = I[i-1] + dt * (k1 + 2*k2 + 2*k3) / 6
        R[i] = N - S[i] - I[i]
    return S, I, R

beta = 0.2
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
t_end = 100
dt = 0.1
S, I, R = SIR_RK3(beta, gamma, S0, I0, R0, t_end, dt)

plt.figure(figsize=(10, 6))
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.title('SIR Model using RK3')
plt.legend()
plt.show()
