import numpy as np


def rk3_serid(beta, gamma, N, I0, R0, T):
    # Initialize arrays
    t = np.linspace(0, T, T+1)
    S = np.zeros(T+1)
    E = np.zeros(T+1)
    I = np.zeros(T+1)
    R = np.zeros(T+1)
    D = np.zeros(T+1)
    S[0] = N - I0 - R0
    E[0] = 0
    I[0] = I0
    R[0] = R0

    # Step size
    dt = t[1] - t[0]

    # Runge-Kutta method
    for n in range(T):
        k1_S = -beta * S[n] * I[n] / N
        k1_E = beta * S[n] * I[n] / N - gamma * E[n]
        k1_I = gamma * E[n] - gamma * I[n]
        k1_R = gamma * I[n]

        S_half = S[n] + k1_S * dt / 2
        E_half = E[n] + k1_E * dt / 2
        I_half = I[n] + k1_I * dt / 2
        R_half = R[n] + k1_R * dt / 2

        k2_S = -beta * S_half * I_half / N
        k2_E = beta * S_half * I_half / N - gamma * E_half
        k2_I = gamma * E_half - gamma * I_half
        k2_R = gamma * I_half

        S_new = S[n] + k2_S * dt
        E_new = E[n] + k2_E * dt
        I_new = I[n] + k2_I * dt
        R_new = R[n] + k2_R * dt

        k3_S = -beta * S_new * I_new / N
        k3_E = beta * S_new * I_new / N - gamma * E_new
        k3_I = gamma * E_new - gamma * I_new
        k3_R = gamma * I_new

        S[n+1] = S[n] + (k1_S + 4*k2_S + k3_S) * dt / 6
        E[n+1] = E[n] + (k1_E + 4*k2_E + k3_E) * dt / 6
        I[n+1] = I[n] + (k1_I + 4*k2_I + k3_I) * dt / 6
        R[n+1] = R[n] + (k1_R + 4*k2_R + k3_R) * dt / 6

    result = {
        'S': S,
        'E': E,
        'I': I,
        'R': R,
        'D': D
    }
    return result


beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
T = 100

result = rk3_serid(beta, gamma, N, I0, R0, T)
