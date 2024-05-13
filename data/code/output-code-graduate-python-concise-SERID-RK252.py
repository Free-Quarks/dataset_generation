import numpy as np
import matplotlib.pyplot as plt


def serid_rk2(s0, e0, r0, i0, d0, beta, gamma, delta, alpha, mu, t_max, dt):
    t = np.arange(0, t_max, dt)
    n = len(t)
    s = np.zeros(n)
    e = np.zeros(n)
    r = np.zeros(n)
    i = np.zeros(n)
    d = np.zeros(n)
    s[0] = s0
    e[0] = e0
    r[0] = r0
    i[0] = i0
    d[0] = d0

    for k in range(n-1):
        ds1 = -beta * s[k] * i[k]
        de1 = beta * s[k] * i[k] - delta * e[k] - alpha * e[k]
        dr1 = gamma * i[k]
        di1 = delta * e[k] - gamma * i[k] - mu * i[k]
        dd1 = alpha * e[k] + mu * i[k]

        s_half = s[k] + ds1 * dt / 2
        e_half = e[k] + de1 * dt / 2
        r_half = r[k] + dr1 * dt / 2
        i_half = i[k] + di1 * dt / 2
        d_half = d[k] + dd1 * dt / 2

        ds2 = -beta * s_half * i_half
        de2 = beta * s_half * i_half - delta * e_half - alpha * e_half
        dr2 = gamma * i_half
        di2 = delta * e_half - gamma * i_half - mu * i_half
        dd2 = alpha * e_half + mu * i_half

        s[k+1] = s[k] + ds2 * dt
        e[k+1] = e[k] + de2 * dt
        r[k+1] = r[k] + dr2 * dt
        i[k+1] = i[k] + di2 * dt
        d[k+1] = d[k] + dd2 * dt

    return s, e, r, i, d


def plot_serid(s, e, r, i, d, t):
    plt.plot(t, s, label='Susceptible')
    plt.plot(t, e, label='Exposed')
    plt.plot(t, r, label='Recovered')
    plt.plot(t, i, label='Infected')
    plt.plot(t, d, label='Dead')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SERID Model')
    plt.legend()
    plt.show()


s0 = 1000
e0 = 0
r0 = 0
i0 = 10
beta = 0.2
gamma = 0.1
delta = 0.05
alpha = 0.01
mu = 0.03
t_max = 100
dt = 0.1

s, e, r, i, d = serid_rk2(s0, e0, r0, i0, d0, beta, gamma, delta, alpha, mu, t_max, dt)

plot_serid(s, e, r, i, d, np.arange(0, t_max, dt))
