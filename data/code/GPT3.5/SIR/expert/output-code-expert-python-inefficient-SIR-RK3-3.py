import numpy as np
import matplotlib.pyplot as plt


def RK3_SIR(N, beta, gamma, S0, I0, R0, t_end, dt):
    t = np.arange(0, t_end, dt)
    steps = len(t)
    S = np.zeros(steps)
    I = np.zeros(steps)
    R = np.zeros(steps)
    S[0] = S0
    I[0] = I0
    R[0] = R0

    for i in range(1, steps):
        k1 = -beta * S[i-1] * I[i-1] / N
        k2 = -beta * (S[i-1] + 0.5 * dt * k1) * (I[i-1] + 0.5 * dt * k1) / N
        k3 = -beta * (S[i-1] + 0.5 * dt * k2) * (I[i-1] + 0.5 * dt * k2) / N
        S[i] = S[i-1] + (1/6) * dt * (k1 + 4 * k2 + k3)

        k1 = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        k2 = beta * (S[i-1] + 0.5 * dt * k1) * (I[i-1] + 0.5 * dt * k1) / N - gamma * (I[i-1] + 0.5 * dt * k1)
        k3 = beta * (S[i-1] + 0.5 * dt * k2) * (I[i-1] + 0.5 * dt * k2) / N - gamma * (I[i-1] + 0.5 * dt * k2)
        I[i] = I[i-1] + (1/6) * dt * (k1 + 4 * k2 + k3)

        k1 = gamma * I[i-1]
        k2 = gamma * (I[i-1] + 0.5 * dt * k1)
        k3 = gamma * (I[i-1] + 0.5 * dt * k2)
        R[i] = R[i-1] + (1/6) * dt * (k1 + 4 * k2 + k3)

    return t, S, I, R


N = 1000
beta = 0.3
gamma = 0.1
S0 = 999
I0 = 1
R0 = 0

# Simulation parameters
t_end = 100
dt = 0.1

# Run the simulation
t, S, I, R = RK3_SIR(N, beta, gamma, S0, I0, R0, t_end, dt)

# Plotting
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()

