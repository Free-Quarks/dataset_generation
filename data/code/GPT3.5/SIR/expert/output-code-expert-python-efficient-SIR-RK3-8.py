import numpy as np
import matplotlib.pyplot as plt


def SIR_RK3(beta, gamma, S0, I0, R0, N, dt, tmax):
    t = np.arange(0, tmax, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = S0
    I[0] = I0
    R[0] = R0
    for i in range(1, len(t)):
        k1 = dt * (-beta * S[i-1] * I[i-1] / N)
        k2 = dt * (-beta * (S[i-1] + 0.5 * k1) * (I[i-1] + 0.5 * k1) / N)
        k3 = dt * (-beta * (S[i-1] - k1 + 2 * k2) * (I[i-1] - k1 + 2 * k2) / N)
        S[i] = S[i-1] + (1/6) * (k1 + 4*k2 + k3)
        k1 = dt * (beta * S[i-1] * I[i-1] / N - gamma * I[i-1])
        k2 = dt * (beta * (S[i-1] + 0.5 * k1) * (I[i-1] + 0.5 * k1) / N - gamma * (I[i-1] + 0.5 * k1))
        k3 = dt * (beta * (S[i-1] - k1 + 2 * k2) * (I[i-1] - k1 + 2 * k2) / N - gamma * (I[i-1] - k1 + 2 * k2))
        I[i] = I[i-1] + (1/6) * (k1 + 4*k2 + k3)
        R[i] = N - S[i] - I[i]
    return S, I, R


beta = 0.2
gamma = 0.1
S0 = 0.9
I0 = 0.1
R0 = 0
N = 1

# Simulation parameters
dt = 0.1
tmax = 100

S, I, R = SIR_RK3(beta, gamma, S0, I0, R0, N, dt, tmax)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK3')
plt.legend()
plt.grid()
plt.show()
