import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def seirid_model(y, t, N, beta, sigma, gamma, delta, alpha):
    S, E, I, R, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - (gamma + delta + alpha) * I
    dRdt = gamma * I
    dDdt = delta * I
    dAdt = alpha * I
    return dSdt, dEdt, dIdt, dRdt, dDdt, dAdt


def run_seirid_model(N, E0, I0, R0, D0, beta, sigma, gamma, delta, alpha, days):
    # Initial conditions vector
    y0 = N - (E0 + I0 + R0 + D0), E0, I0, R0, D0
    
    # Time vector
    t = np.linspace(0, days, days)
    
    # Integrate the SEIRID equations over the time grid t
    result = odeint(seirid_model, y0, t, args=(N, beta, sigma, gamma, delta, alpha))
    
    # Plotting
    plt.plot(t, result[:, 0], label='Susceptible')
    plt.plot(t, result[:, 1], label='Exposed')
    plt.plot(t, result[:, 2], label='Infected')
    plt.plot(t, result[:, 3], label='Recovered')
    plt.plot(t, result[:, 4], label='Deaths')
    plt.plot(t, result[:, 5], label='Asymptomatic')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.legend()
    plt.show()
}

