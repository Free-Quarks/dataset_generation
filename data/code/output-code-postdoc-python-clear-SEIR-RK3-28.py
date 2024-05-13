import numpy as np
import matplotlib.pyplot as plt


def SEIR_RK3(N, I0, beta, gamma, sigma, t_end, dt):
    # Initialize arrays
    t = np.arange(0, t_end, dt)
    S = np.zeros_like(t)
    E = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)

    # Set initial conditions
    S[0] = N - I0
    E[0] = 0
    I[0] = I0
    R[0] = 0

    # Runge-Kutta 3 method
    for i in range(1, len(t)):
        k1_S = -beta * S[i - 1] * I[i - 1] / N
        k1_E = beta * S[i - 1] * I[i - 1] / N - sigma * E[i - 1]
        k1_I = sigma * E[i - 1] - gamma * I[i - 1]
        k1_R = gamma * I[i - 1]
        k2_S = -beta * (S[i - 1] + dt * k1_S / 2) * (I[i - 1] + dt * k1_I / 2) / N
        k2_E = beta * (S[i - 1] + dt * k1_S / 2) * (I[i - 1] + dt * k1_I / 2) / N - sigma * (E[i - 1] + dt * k1_E / 2)
        k2_I = sigma * (E[i - 1] + dt * k1_E / 2) - gamma * (I[i - 1] + dt * k1_I / 2)
        k2_R = gamma * (I[i - 1] + dt * k1_I / 2)
        k3_S = -beta * (S[i - 1] + dt * k2_S / 2) * (I[i - 1] + dt * k2_I / 2) / N
        k3_E = beta * (S[i - 1] + dt * k2_S / 2) * (I[i - 1] + dt * k2_I / 2) / N - sigma * (E[i - 1] + dt * k2_E / 2)
        k3_I = sigma * (E[i - 1] + dt * k2_E / 2) - gamma * (I[i - 1] + dt * k2_I / 2)
        k3_R = gamma * (I[i - 1] + dt * k2_I / 2)
        S[i] = S[i - 1] + dt * (k1_S + 2 * k2_S + 2 * k3_S) / 6
        E[i] = E[i - 1] + dt * (k1_E + 2 * k2_E + 2 * k3_E) / 6
        I[i] = I[i - 1] + dt * (k1_I + 2 * k2_I + 2 * k3_I) / 6
        R[i] = R[i - 1] + dt * (k1_R + 2 * k2_R + 2 * k3_R) / 6

    return S, E, I, R


# Example usage
N = 1000
I0 = 10
beta = 0.5
sigma = 0.1
gamma = 0.2
t_end = 100
dt = 0.1

S, E, I, R = SEIR_RK3(N, I0, beta, gamma, sigma, t_end, dt)

# Plot results
plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.title('SEIR Model using RK3')
plt.legend()
plt.show()

