from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

def seir_model(y, t, beta, gamma, sigma):
    S, E, I, R = y
    dSdt = -beta * S * I
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]


def simulate_seir_model(N, beta, gamma, sigma, E0, I0, R0, days):
    S0 = N - E0 - I0 - R0
    y0 = [S0, E0, I0, R0]
    t = np.linspace(0, days, days)
    result = odeint(seir_model, y0, t, args=(beta, gamma, sigma))
    S, E, I, R = result.T
    
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered/Removed')
    plt.xlabel('Days')
    plt.ylabel('Population')
    plt.title('SEIR Model Simulation')
    plt.legend()
    plt.show()
}

