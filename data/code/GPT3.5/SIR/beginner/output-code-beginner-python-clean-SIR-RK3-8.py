import numpy as np
import matplotlib.pyplot as plt


def SIR_RK3(N, I0, R0, beta, gamma, t_end, dt):
    # Initialize arrays
    t = np.arange(0, t_end, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = N - I0
    I[0] = I0
    R[0] = R0

    # Runge-Kutta 3rd order
    for i in range(1, len(t)):
        k1 = -beta * S[i-1] * I[i-1] / N
        l1 = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        k2 = -beta * (S[i-1] + 0.5 * dt * k1) * (I[i-1] + 0.5 * dt * l1) / N
        l2 = beta * (S[i-1] + 0.5 * dt * k1) * (I[i-1] + 0.5 * dt * l1) / N - gamma * (I[i-1] + 0.5 * dt * l1)
        k3 = -beta * (S[i-1] - dt * k1 + 2 * dt * k2) * (I[i-1] - dt * l1 + 2 * dt * l2) / N
        l3 = beta * (S[i-1] - dt * k1 + 2 * dt * k2) * (I[i-1] - dt * l1 + 2 * dt * l2) / N - gamma * (I[i-1] - dt * l1 + 2 * dt * l2)

        S[i] = S[i-1] + dt * (1/6 * k1 + 2/3 * k2 + 1/6 * k3)
        I[i] = I[i-1] + dt * (1/6 * l1 + 2/3 * l2 + 1/6 * l3)
        R[i] = R[i-1] + dt * gamma * (l1 + 4 * l2 + l3) / 6

    return t, S, I, R


# Parameters
N = 1000000
I0 = 100
R0 = 0
beta = 0.3
gamma = 0.1

# Simulation
t, S, I, R = SIR_RK3(N, I0, R0, beta, gamma, 100, 0.1)

# Plot
plt.plot(t, S, label='S')
plt.plot(t, I, label='I')
plt.plot(t, R, label='R')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model - RK3')
plt.legend()
plt.show()

