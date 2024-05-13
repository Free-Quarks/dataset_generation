import numpy as np


def SEIR_RK2(N, beta, gamma, sigma, duration, dt):
    steps = int(duration / dt)
    S = np.zeros(steps)
    E = np.zeros(steps)
    I = np.zeros(steps)
    R = np.zeros(steps)
    t = np.linspace(0, duration, steps)
    S[0] = N - 1
    I[0] = 1
    dt2 = dt / 2.0

    for i in range(1, steps):
        S[i] = S[i-1] + dt * (-beta * S[i-1] * I[i-1] / N)
        E[i] = E[i-1] + dt * (beta * S[i-1] * I[i-1] / N - sigma * E[i-1])
        I[i] = I[i-1] + dt * (sigma * E[i-1] - gamma * I[i-1])
        R[i] = R[i-1] + dt * (gamma * I[i-1])

    return S, E, I, R
