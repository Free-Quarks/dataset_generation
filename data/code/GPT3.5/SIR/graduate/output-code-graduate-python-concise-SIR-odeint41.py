import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def SIR_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def simulate_sir_model(N, I0, R0, beta, gamma, t):
    S0 = N - I0 - R0
    y0 = S0, I0, R0

    sol = odeint(SIR_model, y0, t, args=(N, beta, gamma))
    S, I, R = sol.T

    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of individuals')
    plt.legend()
    plt.title('SIR Model Simulation')
    plt.show()
}

