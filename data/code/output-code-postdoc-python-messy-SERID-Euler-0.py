import numpy as np
import matplotlib.pyplot as plt


def SERID_model(N, beta, gamma, delta, alpha, rho, I0, E0, R0, D0, t_max):
    S = [N - (I0 + E0 + R0 + D0)]
    E = [E0]
    I = [I0]
    R = [R0]
    D = [D0]

    t = np.linspace(0, t_max, t_max + 1)
    dt = t[1] - t[0]

    for i in range(t_max):
        dSdt = -beta * S[i] * (I[i] + alpha * E[i]) / N
        dEdt = beta * S[i] * (I[i] + alpha * E[i]) / N - delta * E[i]
        dIdt = delta * E[i] - gamma * I[i] - rho * I[i]
        dRdt = gamma * I[i]
        dDdt = rho * I[i]

        S.append(S[i] + dSdt * dt)
        E.append(E[i] + dEdt * dt)
        I.append(I[i] + dIdt * dt)
        R.append(R[i] + dRdt * dt)
        D.append(D[i] + dDdt * dt)

    return S, E, I, R, D

