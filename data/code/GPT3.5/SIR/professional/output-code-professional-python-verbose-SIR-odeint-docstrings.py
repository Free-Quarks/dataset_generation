import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def SIR_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def plot_epidemic(t, S, I, R):
    fig, ax = plt.subplots()
    ax.plot(t, S, 'b', label='Susceptible')
    ax.plot(t, I, 'r', label='Infected')
    ax.plot(t, R, 'g', label='Recovered')
    ax.set(xlabel='Time', ylabel='Population', title='Epidemic Simulation')
    ax.legend()
    ax.grid()
    plt.show()


N = 1000
I0 = 1
R0 = 0
S0 = N - I0 - R0
beta = 0.2
gamma = 0.1
t = np.linspace(0, 100, 100)
y0 = S0, I0, R0
sol = odeint(SIR_model, y0, t, args=(N, beta, gamma))
S, I, R = sol.T
plot_epidemic(t, S, I, R)
