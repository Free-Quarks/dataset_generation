import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def seir_model(y, t, beta, gamma, sigma):
    S, E, I, R = y
    dSdt = -beta * S * I
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]


def simulate_seir_model(beta, gamma, sigma, S0, E0, I0, R0, t_start, t_end, t_step):
    t = np.arange(t_start, t_end, t_step)
    y0 = [S0, E0, I0, R0]
    result = odeint(seir_model, y0, t, args=(beta, gamma, sigma))
    S, E, I, R = result.T
    
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SEIR Model')
    plt.legend()
    plt.grid(True)
    plt.show()
}

