import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def sidarthe_model(y, t, N, beta, sigma, gamma, mu, eta, theta):
    S, I, D, A, R, T, H, E = y
    dSdt = -beta * S * (I + theta * A) / N
    dIdt = (beta * S * (I + theta * A) / N) - (sigma * I) - (gamma * I) - (mu * I)
    dDdt = mu * I
    dAdt = (1 - eta) * sigma * I - (gamma * A)
    dRdt = gamma * I + gamma * A
    dTdt = theta * eta * sigma * I - (mu * T)
    dHdt = theta * (1 - eta) * sigma * I - (mu * H)
    dEdt = eta * sigma * I - (gamma * E)
    return dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt


def sidarthe_simulation(N, initial_conditions, parameters, days):
    S0, I0, D0, A0, R0, T0, H0, E0 = initial_conditions
    beta, sigma, gamma, mu, eta, theta = parameters
    t = np.arange(days)

    y0 = S0, I0, D0, A0, R0, T0, H0, E0
    result = odeint(sidarthe_model, y0, t, args=(N, beta, sigma, gamma, mu, eta, theta))

    S, I, D, A, R, T, H, E = result.T

    fig, ax = plt.subplots()
    ax.plot(t, I, 'r', label='Infected')
    ax.plot(t, D, 'k', label='Deceased')
    ax.plot(t, A, 'b', label='Asymptomatic')
    ax.plot(t, R, 'g', label='Recovered')
    ax.plot(t, T, 'm', label='Treated')
    ax.plot(t, H, 'y', label='Hospitalized')
    ax.plot(t, E, 'c', label='Exposed')
    ax.set_xlabel('Days')
    ax.set_ylabel('Population')
    ax.set_title('SIDARTHE Model Simulation')
    ax.legend()
    plt.show()


initial_conditions = 60000000, 100, 0, 100, 0, 0, 0, 0
parameters = 0.9, 0.2, 0.25, 0.01, 0.04, 0.5
sidarthe_simulation(60000000, initial_conditions, parameters, 200)
