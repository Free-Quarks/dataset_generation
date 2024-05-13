import numpy as np


def sidarthe_model(y, t, N, beta, gamma, delta, theta, alpha, rho, sigma):
    S, I, D, A, R, T, H, E = y
    dSdt = -beta * S * I / N
    dIdt = (1 - delta) * beta * S * I / N - (1 - alpha) * gamma * I - alpha * theta * I
    dDdt = delta * beta * S * I / N
    dAdt = alpha * theta * I - rho * A
    dRdt = (1 - alpha) * gamma * I + rho * A
    dTdt = sigma * (1 - alpha) * gamma * I
    dHdt = sigma * alpha * theta * I
    dEdt = beta * S * I / N
    return dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt


def run_sidarthe_model(N, I0, D0, A0, R0, T0, H0, E0, beta, gamma, delta, theta, alpha, rho, sigma, days):
    S0 = N - I0 - D0 - A0 - R0 - T0 - H0 - E0
    y0 = S0, I0, D0, A0, R0, T0, H0, E0
    t = np.linspace(0, days, days)
    ret = odeint(sidarthe_model, y0, t, args=(N, beta, gamma, delta, theta, alpha, rho, sigma))
    S, I, D, A, R, T, H, E = ret.T
    return S, I, D, A, R, T, H, E
