import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def simulate_sir_model(N, I0, R0, beta, gamma, days):
    def sir_model(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    S0 = N - I0 - R0
    y0 = S0, I0, R0
    t = np.linspace(0, days, days)
    sol = odeint(sir_model, y0, t, args=(N, beta, gamma))

    S, I, R = sol[:, 0], sol[:, 1], sol[:, 2]

    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage
N = 1000  # total population
I0 = 1  # initial number of infected individuals
R0 = 0  # initial number of recovered individuals
beta = 0.2  # infection rate
gamma = 0.1  # recovery rate
days = 100  # number of days

simulate_sir_model(N, I0, R0, beta, gamma, days)
