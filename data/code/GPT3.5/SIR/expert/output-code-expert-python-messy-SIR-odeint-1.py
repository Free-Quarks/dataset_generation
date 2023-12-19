import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def SIR_model(y, t, beta, gamma):
    S, I, R = y
    N = S + I + R
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def plot_SIR(S, I, R):
    plt.plot(S, 'b-', label='Susceptible')
    plt.plot(I, 'r-', label='Infected')
    plt.plot(R, 'g-', label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.legend()
    plt.show()


def main(N, I0, R0, beta, gamma, days):
    S0 = N - I0 - R0
    t = np.linspace(0, days, days)
    y0 = S0, I0, R0
    args = (beta, gamma)
    result = odeint(SIR_model, y0, t, args)
    S, I, R = result.T
    plot_SIR(S, I, R)


if __name__ == '__main__':
    N = 1000  # population size
    I0 = 1  # initial number of infected individuals
    R0 = 0  # initial number of recovered individuals
    beta = 0.2  # contact rate
    gamma = 0.1  # recovery rate
    days = 160  # number of days
    main(N, I0, R0, beta, gamma, days)
