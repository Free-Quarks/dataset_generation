import numpy as np
import matplotlib.pyplot as plt
import json

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def rk4(y, t, dt, model, beta, gamma):
    k1 = model(y, t, beta, gamma)
    k2 = model([y[i] + dt / 2 * k1[i] for i in range(3)], t + dt / 2, beta, gamma)
    k3 = model([y[i] + dt / 2 * k2[i] for i in range(3)], t + dt / 2, beta, gamma)
    k4 = model([y[i] + dt * k3[i] for i in range(3)], t + dt, beta, gamma)
    return [y[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(3)]

def simulate_sir(S0, I0, R0, beta, gamma, dt, N):
    S, I, R = [S0], [I0], [R0]
    t = np.linspace(0, N, int(N/dt) + 1)
    for _ in t[1:]:
        next_S, next_I, next_R = rk4([S[-1], I[-1], R[-1]], _, dt, sir_model, beta, gamma)
        S.append(next_S)
        I.append(next_I)
        R.append(next_R)
    return t, S, I, R

def plot_sir(t, S, I, R):
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)
    ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number')
    ax.legend()
    plt.show()

t, S, I, R = simulate_sir(999, 1, 0, 0.2, 0.1, 0.1, 150)
plot_sir(t, S, I, R)
