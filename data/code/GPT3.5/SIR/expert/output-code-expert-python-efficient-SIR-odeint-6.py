import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def plot_SIR_model(S, I, R):
    plt.plot(S, 'b', label='Susceptible')
    plt.plot(I, 'r', label='Infected')
    plt.plot(R, 'g', label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Number of individuals')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


def simulate_SIR_model(S0, I0, R0, beta, gamma, t):
    y0 = [S0, I0, R0]
    solution = odeint(SIR_model, y0, t, args=(beta, gamma))
    S = solution[:, 0]
    I = solution[:, 1]
    R = solution[:, 2]
    plot_SIR_model(S, I, R)


S0 = 1000
I0 = 1
R0 = 0
t = np.linspace(0, 49, 50)
beta = 0.2
gamma = 0.1

simulate_SIR_model(S0, I0, R0, beta, gamma, t)
