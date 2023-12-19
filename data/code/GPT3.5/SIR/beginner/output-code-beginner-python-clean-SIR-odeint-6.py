import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def plot_model(S0, I0, R0, beta, gamma, days):
    t = np.linspace(0, days, days)
    y0 = [S0, I0, R0]
    solution = odeint(SIR_model, y0, t, args=(beta, gamma))
    S = solution[:, 0]
    I = solution[:, 1]
    R = solution[:, 2]
    
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


plot_model(999, 1, 0, 0.2, 0.1, 100)
