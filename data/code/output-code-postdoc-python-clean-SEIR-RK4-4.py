import numpy as np


def seir_model(beta, gamma, sigma, N, I0, E0, R0, T):
    def deriv(y, t, beta, gamma, sigma, N):
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    t = np.linspace(0, T, T)
    y0 = S0, E0, I0, R0
    ret = odeint(deriv, y0, t, args=(beta, gamma, sigma, N))
    S, E, I, R = ret.T

    return S, E, I, R


N = 100000
I0, E0, R0 = 1, 0, 0
beta, gamma, sigma = 0.2, 1.0 / 10, 1.0 / 10
T = 160

S, E, I, R = seir_model(beta, gamma, sigma, N, I0, E0, R0, T)

