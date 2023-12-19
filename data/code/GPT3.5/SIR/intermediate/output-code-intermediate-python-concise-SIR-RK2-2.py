import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(beta, gamma, S0, I0, R0, N, t)
    def dSdt(t, S, I):
        return -beta * S * I / N

    def dIdt(t, S, I):
        return beta * S * I / N - gamma * I

    def dRdt(t, I):
        return gamma * I

    dt = t[1] - t[0]

    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))

    S[0] = S0
    I[0] = I0
    R[0] = R0

    for i in range(1, len(t)):
        k1_s = dt * dSdt(t[i-1], S[i-1], I[i-1])
        k1_i = dt * dIdt(t[i-1], S[i-1], I[i-1])
        k1_r = dt * dRdt(t[i-1], I[i-1])

        k2_s = dt * dSdt(t[i-1] + dt/2, S[i-1] + k1_s/2, I[i-1] + k1_i/2)
        k2_i = dt * dIdt(t[i-1] + dt/2, S[i-1] + k1_s/2, I[i-1] + k1_i/2)
        k2_r = dt * dRdt(t[i-1] + dt/2, I[i-1] + k1_i/2)

        S[i] = S[i-1] + k2_s
        I[i] = I[i-1] + k2_i
        R[i] = R[i-1] + k2_r

    return S, I, R

beta = 0.2
    gamma = 0.1
    S0 = 999
    I0 = 1
    R0 = 0
    N = S0 + I0 + R0
    t = np.linspace(0, 100, 100)

S, I, R = SIR_RK2(beta, gamma, S0, I0, R0, N, t)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.legend()
plt.grid(True)
plt.show()
