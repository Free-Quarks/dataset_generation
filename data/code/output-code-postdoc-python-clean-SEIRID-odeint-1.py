import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Function that defines the SEIRID model

def seirid_model(y, t, beta, gamma, delta, alpha, N):
    S, E, I, R, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - delta * E
    dIdt = delta * E - (1 - alpha) * gamma * I - alpha * delta * I
    dRdt = (1 - alpha) * gamma * I
    dDdt = alpha * delta * I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]


def run_seirid_model(N, E0, I0, R0, D0, beta, gamma, delta, alpha, t_max):
    # Initial conditions
    S0 = N - E0 - I0 - R0 - D0
    y0 = [S0, E0, I0, R0, D0]

    # Time points
    t = np.linspace(0, t_max, t_max+1)

    # Integrate the SEIRID equations over the time grid, t
    result = odeint(seirid_model, y0, t, args=(beta, gamma, delta, alpha, N))
    S, E, I, R, D = result.T

    # Plotting
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.plot(t, D, label='Deceased')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.legend()
    plt.title('SEIRID Model')
    plt.show()


run_seirid_model(N=10000, E0=10, I0=1, R0=0, D0=0, beta=0.2, gamma=0.1, delta=0.5, alpha=0.01, t_max=365)
