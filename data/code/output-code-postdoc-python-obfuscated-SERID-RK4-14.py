import numpy as np


def serid_model(t, y, beta, gamma, delta):
    S, E, R, I, D = y
    N = S + E + R + I + D
    dSdt = -beta * S * (I + E) / N
    dEdt = beta * S * (I + E) / N - delta * E
    dRdt = gamma * I
    dIdt = delta * E - gamma * I
    dDdt = gamma * I
    return [dSdt, dEdt, dRdt, dIdt, dDdt]


def rk4_integrate(func, y0, t, args=()):
    dt = t[1] - t[0]
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t) - 1):
        k1 = func(t[i], y[i], *args)
        k2 = func(t[i] + 0.5 * dt, y[i] + 0.5 * dt * k1, *args)
        k3 = func(t[i] + 0.5 * dt, y[i] + 0.5 * dt * k2, *args)
        k4 = func(t[i] + dt, y[i] + dt * k3, *args)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
    return y


def serid_simulation(S0, E0, R0, I0, D0, beta, gamma, delta, t):
    y0 = [S0, E0, R0, I0, D0]
    return rk4_integrate(serid_model, y0, t, args=(beta, gamma, delta))


# Example usage

# Model parameters
S0 = 100000
E0 = 100
R0 = 0
I0 = 10
D0 = 0
beta = 0.5
gamma = 0.2
delta = 0.1

# Simulation time
t = np.linspace(0, 100, 1000)

# Run simulation
y = serid_simulation(S0, E0, R0, I0, D0, beta, gamma, delta, t)

# Plot results
import matplotlib.pyplot as plt

plt.plot(t, y[:, 0], label='Susceptible')
plt.plot(t, y[:, 1], label='Exposed')
plt.plot(t, y[:, 2], label='Recovered')
plt.plot(t, y[:, 3], label='Infected')
plt.plot(t, y[:, 4], label='Deaths')

plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SERID Model Simulation')
plt.show()
