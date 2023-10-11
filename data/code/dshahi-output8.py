import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def serid_model(y, t, beta, gamma, delta):
    S, E, I, R, D = y
    dSdt = -beta * S * I
    dEdt = beta * S * I - delta * E
    dIdt = delta * E - gamma * I
    dRdt = gamma * I
    dDdt = delta * E
    return [dSdt, dEdt, dIdt, dRdt, dDdt]


def run_serid_model(beta, gamma, delta, S0, E0, I0, R0, D0, days):
    y0 = [S0, E0, I0, R0, D0]
    t = np.linspace(0, days, days)
    sol = odeint(serid_model, y0, t, args=(beta, gamma, delta))
    S, E, I, R, D = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4]
    
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.plot(t, D, label='Dead')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SEIR Model')
    plt.legend()
    plt.grid(True)
    plt.show()


run_serid_model(0.2, 0.1, 0.05, 1000, 1, 1, 0, 0, 100)
