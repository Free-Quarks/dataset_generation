import numpy as np
import matplotlib.pyplot as plt


def sidarthe_euler(beta, sigma, gamma, mu, delta, rho, population, initial_conditions, t_start, t_end, dt):
    # Parameters
    N = population
    beta = beta
    sigma = sigma
    gamma = gamma
    mu = mu
    delta = delta
    rho = rho

    # Initial conditions
    S0, I0, D0, A0, R0, T0, H0, E0 = initial_conditions
    R0 = 0

    # Time vector
    t = np.arange(t_start, t_end, dt)

    # Empty arrays to store the results
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    D = np.zeros(len(t))
    A = np.zeros(len(t))
    R = np.zeros(len(t))
    T = np.zeros(len(t))
    H = np.zeros(len(t))
    E = np.zeros(len(t))

    # Initial conditions
    S[0] = S0
    I[0] = I0
    D[0] = D0
    A[0] = A0
    R[0] = R0
    T[0] = T0
    H[0] = H0
    E[0] = E0

    # Euler's method
    for i in range(1, len(t)):
        dS = -beta * S[i-1] * I[i-1] / N
        dI = beta * S[i-1] * I[i-1] / N - gamma * I[i-1] - sigma * I[i-1] - rho * I[i-1]
        dD = sigma * I[i-1] - mu * D[i-1]
        dA = rho * I[i-1] - delta * A[i-1]
        dR = gamma * I[i-1] + delta * A[i-1]
        dT = mu * D[i-1]
        dH = sigma * I[i-1]
        dE = rho * I[i-1]

        S[i] = S[i-1] + dt * dS
        I[i] = I[i-1] + dt * dI
        D[i] = D[i-1] + dt * dD
        A[i] = A[i-1] + dt * dA
        R[i] = R[i-1] + dt * dR
        T[i] = T[i-1] + dt * dT
        H[i] = H[i-1] + dt * dH
        E[i] = E[i-1] + dt * dE

    return S, I, D, A, R, T, H, E
}

