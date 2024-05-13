import numpy as np
import matplotlib.pyplot as plt


def rk4(derivs, y0, t0, dt, N):
    y0 = np.array(y0)
    t = np.linspace(t0, t0 + N*dt, N+1)
    y = np.zeros((N+1, len(y0)))
    y[0] = y0
    for i in range(N):
        k1 = dt * derivs(y[i], t[i])
        k2 = dt * derivs(y[i] + 0.5*k1, t[i] + 0.5*dt)
        k3 = dt * derivs(y[i] + 0.5*k2, t[i] + 0.5*dt)
        k4 = dt * derivs(y[i] + k3, t[i] + dt)
        y[i+1] = y[i] + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    return t, y


def sird_model(y, t):
    beta = 0.2
    gamma = 0.1
    delta = 0.05
    S, I, R, D = y
    N = S + I + R + D
    dSdt = -beta * S * I / N
    dIdt = (beta * S * I / N) - (gamma * I) - (delta * I)
    dRdt = gamma * I
    dDdt = delta * I
    return [dSdt, dIdt, dRdt, dDdt]


t0 = 0
tf = 100
N = 1000
y0 = [999, 1, 0, 0]
dt = (tf - t0) / N


t, y = rk4(sird_model, y0, t0, dt, N)

plt.plot(t, y[:, 0], label='Susceptible')
plt.plot(t, y[:, 1], label='Infected')
plt.plot(t, y[:, 2], label='Recovered')
plt.plot(t, y[:, 3], label='Deceased')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIRD Model')
plt.legend()
plt.show()
