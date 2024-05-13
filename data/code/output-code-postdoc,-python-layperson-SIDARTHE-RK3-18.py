import numpy as np
import matplotlib.pyplot as plt


def sidarthe_rk3(N, I0, E0, R0, D0, beta, gamma, alpha, theta, sigma, delta, t_max, dt):
    t = np.arange(0, t_max+dt, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    D = np.zeros_like(t)
    A = np.zeros_like(t)
    R = np.zeros_like(t)
    T = np.zeros_like(t)
    H = np.zeros_like(t)
    E = np.zeros_like(t)
    S[0] = N - I0 - E0 - R0 - D0
    I[0] = I0
    D[0] = D0
    A[0] = alpha * I0
    R[0] = R0
    T[0] = theta * I0
    H[0] = sigma * I0
    E[0] = E0
    for i in range(1, len(t)):
        S[i] = S[i-1] - dt * beta * S[i-1] * I[i-1] / N
        E[i] = E[i-1] + dt * beta * S[i-1] * I[i-1] / N - dt * sigma * E[i-1]
        A[i] = A[i-1] + dt * alpha * I[i-1] - dt * delta * A[i-1]
        I[i] = I[i-1] + dt * sigma * E[i-1] - dt * gamma * I[i-1] - dt * alpha * I[i-1]
        R[i] = R[i-1] + dt * gamma * I[i-1] + dt * delta * A[i-1]
        T[i] = T[i-1] + dt * theta * I[i-1]
        H[i] = H[i-1] + dt * sigma * (1 - alpha) * I[i-1]
        D[i] = D[i-1] + dt * delta * A[i-1]
    return t, S, E, A, I, R, T, H, D


def plot_sidarthe(t, S, E, A, I, R, T, H, D):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    ax[0, 0].plot(t, S, label='Susceptible')
    ax[0, 0].set_xlabel('Time (days)')
    ax[0, 0].set_ylabel('Population')
    ax[0, 0].legend()

    ax[0, 1].plot(t, E, label='Exposed')
    ax[0, 1].set_xlabel('Time (days)')
    ax[0, 1].set_ylabel('Population')
    ax[0, 1].legend()

    ax[1, 0].plot(t, A, label='Asymptomatic')
    ax[1, 0].set_xlabel('Time (days)')
    ax[1, 0].set_ylabel('Population')
    ax[1, 0].legend()

    ax[1, 1].plot(t, I, label='Infected')
    ax[1, 1].plot(t, R, label='Recovered')
    ax[1, 1].plot(t, T, label='Tested')
    ax[1, 1].plot(t, H, label='Hospitalized')
    ax[1, 1].plot(t, D, label='Dead')
    ax[1, 1].set_xlabel('Time (days)')
    ax[1, 1].set_ylabel('Population')
    ax[1, 1].legend()

    plt.tight_layout()
    plt.show()
