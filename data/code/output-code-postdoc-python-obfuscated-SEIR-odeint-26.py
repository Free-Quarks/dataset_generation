import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def seir_model(y, t, beta, gamma, sigma):
    S, E, I, R = y
    dS_dt = -beta * S * I
    dE_dt = beta * S * I - sigma * E
    dI_dt = sigma * E - gamma * I
    dR_dt = gamma * I
    return [dS_dt, dE_dt, dI_dt, dR_dt]


def run_seir_model(N, E0, I0, R0, beta, gamma, sigma, days):
    S0 = N - E0 - I0 - R0
    t = np.linspace(0, days, days)
    y0 = [S0, E0, I0, R0]
    params = (beta, gamma, sigma)
    solution = odeint(seir_model, y0, t, args=params)
    S, E, I, R = solution.T
    
    plt.figure(figsize=(10,6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of individuals')
    plt.title('SEIR Model')
    plt.legend()
    plt.show()
}

