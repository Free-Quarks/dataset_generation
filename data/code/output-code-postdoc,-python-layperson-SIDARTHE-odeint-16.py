import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def sidarthe_model(y, t, N, beta, gamma, delta, alpha, epsilon, rho):
    S, I, D, A, R, T, H, E = y
    dSdt = -beta * S * (I + alpha * A) / N
    dIdt = beta * S * (I + alpha * A) / N - gamma * I - delta * I - epsilon * I
    dDdt = rho * epsilon * I
    dAdt = gamma * I - alpha * A
    dRdt = delta * I
    dTdt = rho * (1 - epsilon) * I
    dHdt = delta * I
    dEdt = delta * I
    return dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt


def run_simulation(N, beta, gamma, delta, alpha, epsilon, rho, days, S0, I0, D0, A0, R0, T0, H0, E0):
    y0 = S0, I0, D0, A0, R0, T0, H0, E0
    t = np.linspace(0, days, days)
    result = odeint(sidarthe_model, y0, t, args=(N, beta, gamma, delta, alpha, epsilon, rho))
    S, I, D, A, R, T, H, E = result.T

    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, D, label='Deceased')
    plt.plot(t, A, label='Asymptomatic')
    plt.plot(t, R, label='Recovered')
    plt.plot(t, T, label='Tested')
    plt.plot(t, H, label='Hospitalized')
    plt.plot(t, E, label='Exposed')

    plt.xlabel('Days')
    plt.ylabel('Population')
    plt.title('SIDARTHE Model Simulation')
    plt.legend()
    plt.show()
}

