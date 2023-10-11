import numpy as np
import matplotlib.pyplot as plt

def sidarthe_model(y, t, params):
    S, I, D, A, R, T, H, E = y
    beta, gamma, mu, alpha, theta, delta, rho, sigma, kappa = params
    N = S + I + D + A + R + T + H + E
    dSdt = -beta * S * (I + D + A) / N
    dIdt = beta * S * (I + D + A) / N - (gamma + mu) * I
    dDdt = mu * I - (alpha + theta + delta) * D
    dAdt = alpha * D - (gamma + rho) * A
    dRdt = gamma * (I + A) + rho * A - kappa * R
    dTdt = theta * D - sigma * T
    dHdt = delta * D - sigma * H
    dEdt = sigma * (T + H) - (alpha + rho) * E
    return [dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt]


def sidarthe_rk4(initial_conditions, params, t_range, h):
    t = np.arange(t_range[0], t_range[1] + h, h)
    y = np.zeros((len(t), len(initial_conditions)))
    y[0] = initial_conditions
    for i in range(1, len(t)):
        k1 = h * sidarthe_model(y[i-1], t[i-1], params)
        k2 = h * sidarthe_model(y[i-1] + 0.5 * k1, t[i-1] + 0.5 * h, params)
        k3 = h * sidarthe_model(y[i-1] + 0.5 * k2, t[i-1] + 0.5 * h, params)
        k4 = h * sidarthe_model(y[i-1] + k3, t[i-1] + h, params)
        y[i] = y[i-1] + (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, y
