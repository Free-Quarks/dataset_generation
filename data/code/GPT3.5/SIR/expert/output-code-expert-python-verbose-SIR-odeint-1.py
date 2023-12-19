import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def simulate_SIR_model(S0, I0, R0, beta, gamma, T):
    y0 = [S0, I0, R0]
    t = np.linspace(0, T, T+1)
    sol = odeint(SIR_model, y0, t, args=(beta, gamma))
    S, I, R = sol[:, 0], sol[:, 1], sol[:, 2]
    
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model Simulation')
    plt.legend()
    plt.show()
}

