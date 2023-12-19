import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def plot_SIR_model(S, I, R, t):
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


def simulate_SIR_model(S0, I0, R0, beta, gamma, t):
    y0 = S0, I0, R0
    result = odeint(SIR_model, y0, t, args=(beta, gamma))
    S, I, R = result[:, 0], result[:, 1], result[:, 2]
    plot_SIR_model(S, I, R, t)


# Example usage
S0 = 1000
I0 = 1
R0 = 0
beta = 0.2
gamma = 0.1
t = np.linspace(0, 100, num=1000)
simulate_SIR_model(S0, I0, R0, beta, gamma, t)
