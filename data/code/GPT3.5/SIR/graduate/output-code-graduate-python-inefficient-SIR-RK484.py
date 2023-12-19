import numpy as np
import matplotlib.pyplot as plt

def SIR_RK4(beta, gamma, n, I0, R0, t_max, dt):
    def SIR_deriv(y, t, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / n
        dIdt = beta * S * I / n - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    t = np.linspace(0, t_max, int(t_max / dt) + 1)
    y0 = n-I0-R0, I0, R0
    ret = np.empty((len(t), 3))
    ret[0] = y0
    for i, ti in enumerate(t[:-1]):
        yi = ret[i]
        k1 = dt * SIR_deriv(yi, ti, beta, gamma)
        k2 = dt * SIR_deriv(yi + 0.5 * k1, ti + 0.5 * dt, beta, gamma)
        k3 = dt * SIR_deriv(yi + 0.5 * k2, ti + 0.5 * dt, beta, gamma)
        k4 = dt * SIR_deriv(yi + k3, ti + dt, beta, gamma)
        ret[i + 1] = yi + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return t, ret[:, 0], ret[:, 1], ret[:, 2]


n = 1000
I0 = 1
R0 = 0
beta = 0.2
gamma = 0.1
t_max = 100
dt = 0.1

t, S, I, R = SIR_RK4(beta, gamma, n, I0, R0, t_max, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.title('SIR Model')
plt.legend()
plt.show()

