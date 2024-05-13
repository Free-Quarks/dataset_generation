import numpy as np


def serid_rk4(y, t, N, beta, gamma, delta):
    S, E, I, R, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - delta * E
    dIdt = delta * E - gamma * I
    dRdt = gamma * I
    dDdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt, dDdt


def run_serid_rk4(N, E0, I0, R0, D0, beta, gamma, delta, t_max, dt):
    t = np.linspace(0, t_max, int(t_max / dt) + 1)
    y0 = N - E0 - I0 - R0 - D0
    S0, E0, I0, R0, D0 = y0, E0, I0, R0, D0
    y = S0, E0, I0, R0, D0
    result = np.zeros((len(t), 5))
    result[0] = y
    for i in range(1, len(t)):
        k1 = dt * serid_rk4(y, t[i-1], N, beta, gamma, delta)
        k2 = dt * serid_rk4(y + 0.5 * k1, t[i-1] + 0.5 * dt, N, beta, gamma, delta)
        k3 = dt * serid_rk4(y + 0.5 * k2, t[i-1] + 0.5 * dt, N, beta, gamma, delta)
        k4 = dt * serid_rk4(y + k3, t[i-1] + dt, N, beta, gamma, delta)
        y = y + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
        result[i] = y
    return t, result
