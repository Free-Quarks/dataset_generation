import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(N, beta, gamma, tmax, dt, initial_conditions):
    t = np.arange(0, tmax, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0], I[0], R[0] = initial_conditions

    for i in range(1, len(t)):
        k1s = -beta * S[i-1] * I[i-1] / N
        k1i = (beta * S[i-1] * I[i-1] / N) - (gamma * I[i-1])
        k1r = gamma * I[i-1]

        k2s = -beta * (S[i-1] + 0.5 * dt * k1s) * (I[i-1] + 0.5 * dt * k1i) / N
        k2i = (beta * (S[i-1] + 0.5 * dt * k1s) * (I[i-1] + 0.5 * dt * k1i) / N) - (gamma * (I[i-1] + 0.5 * dt * k1i))
        k2r = gamma * (I[i-1] + 0.5 * dt * k1i)

        k3s = -beta * (S[i-1] - dt * k1s + 2.0 * dt * k2s) * (I[i-1] - dt * k1i + 2.0 * dt * k2i) / N
        k3i = (beta * (S[i-1] - dt * k1s + 2.0 * dt * k2s) * (I[i-1] - dt * k1i + 2.0 * dt * k2i) / N) - (gamma * (I[i-1] - dt * k1i + 2.0 * dt * k2i))
        k3r = gamma * (I[i-1] - dt * k1i + 2.0 * dt * k2i)

        S[i] = S[i-1] + (dt / 6.0) * (k1s + 4.0 * k2s + k3s)
        I[i] = I[i-1] + (dt / 6.0) * (k1i + 4.0 * k2i + k3i)
        R[i] = R[i-1] + (dt / 6.0) * (k1r + 4.0 * k2r + k3r)

    return S, I, R


# Example usage
N = 1000
beta = 0.3
gamma = 0.1
initial_conditions = (N-1, 1, 0)
tmax = 100
dt = 0.1

S, I, R = SIR_RK3(N, beta, gamma, tmax, dt, initial_conditions)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
