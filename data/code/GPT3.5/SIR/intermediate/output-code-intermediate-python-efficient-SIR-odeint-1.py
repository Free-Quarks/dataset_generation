import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def simulate_SIR_model(S0, I0, R0, beta, gamma, t):
    y0 = S0, I0, R0
    args = (beta, gamma)
    solution = odeint(SIR_model, y0, t, args)
    S, I, R = solution.T
    return S, I, R


def plot_SIR_model(S, I, R, t):
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.grid(True)
    plt.show()

