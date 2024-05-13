import numpy as np


def sidarthe_model(t, y, params):
    S, I, D, A, R, T, H, E = y
    beta, gamma, delta, alpha, rho, theta, eta = params
    N = S + I + D + A + R + T + H + E
    dSdt = -beta * S * (I + alpha * A) / N
    dIdt = beta * S * (I + alpha * A) / N - gamma * I - delta * I
    dDdt = delta * rho * I - theta * D
    dAdt = delta * (1 - rho) * I - eta * A
    dRdt = gamma * I + eta * A
    dTdt = theta * D
    dHdt = theta * (1 - rho) * D
    dEdt = theta * rho * D
    return [dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt]


def run_sim(sidarthe_model, params, initial_conditions, t_max, dt):
    t = np.arange(0, t_max+dt, dt)
    y = np.zeros((len(t), len(initial_conditions)))
    y[0] = initial_conditions
    for i in range(1, len(t)):
        y[i] = rk4_step(sidarthe_model, y[i-1], params, dt)
    return t, y


def rk4_step(f, y, params, dt):
    k1 = f(0, y, params)
    k2 = f(0, y + dt/2 * k1, params)
    k3 = f(0, y + dt/2 * k2, params)
    k4 = f(0, y + dt * k3, params)
    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

