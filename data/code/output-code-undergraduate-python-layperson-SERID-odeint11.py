import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def serid_model(y, t, beta, gamma, N):
    S, E, R, I, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - gamma * E
    dRdt = gamma * E
    dIdt = (1 - D) * gamma * E - D * I
    dDdt = D * I
    return dSdt, dEdt, dRdt, dIdt, dDdt


def serid_simulation(initial_conditions, params, days):
    S0, E0, R0, I0, D0 = initial_conditions
    N = S0 + E0 + R0 + I0 + D0
    beta, gamma, D = params
    t = np.linspace(0, days, days)

    solution = odeint(serid_model, [S0, E0, R0, I0, D0], t, args=(beta, gamma, N))
    S, E, R, I, D = solution.T

    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, R, label='Recovered')
    plt.plot(t, I, label='Infected')
    plt.plot(t, D, label='Dead')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SEIR Model Simulation')
    plt.legend()
    plt.grid(True)
    plt.show()


initial_conditions = [1000, 1, 0, 0, 0]
params = [0.2, 1/14, 0.02]
days = 100

serid_simulation(initial_conditions, params, days)
