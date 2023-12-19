import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def plot_sir_model(t, S, I, R):
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


# Parameters
beta = 0.2
# transmission rate

gamma = 0.1
# recovery rate

N = 1000
# total population

I0 = 1
# initial number of infected individuals

S0 = N - I0
# initial number of susceptible individuals

R0 = 0
# initial number of recovered individuals

# Time vector
t = np.linspace(0, 100, 100)

# Initial conditions vector
y0 = [S0, I0, R0]

# Integrate the SIR equations over the time grid
ty = odeint(sir_model, y0, t, args=(beta, gamma))

S, I, R = ty.T

plot_sir_model(t, S, I, R)
