import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def SIR_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def simulate_sir_model(N, I0, R0, beta, gamma, days):
    S0 = N - I0 - R0
    t = np.linspace(0, days, days)
    y0 = S0, I0, R0
    result = odeint(SIR_model, y0, t, args=(N, beta, gamma))
    S, I, R = result.T

    plt.figure(figsize=(12, 8))
    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.grid()
    plt.show()
}

