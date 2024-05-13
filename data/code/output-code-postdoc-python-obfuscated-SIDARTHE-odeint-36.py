import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def sidarthe(y, t, p):
    S, I, D, A, R, T, H = y
    beta, gamma, mu_I, mu_A, mu_R, mu_T, mu_H = p
    N = S + I + D + A + R + T + H
    dydt = [-beta * S * I / N,
            beta * S * I / N - (1 - gamma) * mu_I * I - gamma * mu_R * I - mu_T * I - mu_H * I,
            (1 - gamma) * mu_I * I,
            gamma * mu_R * I - mu_A * A,
            (1 - gamma) * mu_T * I - mu_R * R,
            (1 - gamma) * mu_A * A - mu_H * T,
            mu_T * I + mu_R * I + mu_H * I]
    return dydt


def simulate_epidemic(S, I, D, A, R, T, H, beta, gamma, mu_I, mu_A, mu_R, mu_T, mu_H, days):
    y0 = [S, I, D, A, R, T, H]
    t = np.linspace(0, days, days)
    p = [beta, gamma, mu_I, mu_A, mu_R, mu_T, mu_H]
    sol = odeint(sidarthe, y0, t, args=(p,))
    S, I, D, A, R, T, H = sol.T
    
    plt.plot(t, S, 'b', label='S')
    plt.plot(t, I, 'r', label='I')
    plt.plot(t, D, 'g', label='D')
    plt.plot(t, A, 'c', label='A')
    plt.plot(t, R, 'm', label='R')
    plt.plot(t, T, 'y', label='T')
    plt.plot(t, H, 'k', label='H')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.legend()
    plt.show()
}

