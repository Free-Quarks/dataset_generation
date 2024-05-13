import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def sidarthe_model(y, t, beta, alpha, gamma, delta, epsilon, theta):
    S, I, D, A, R, T, H, E = y
    N = S + I + D + A + R + T + H + E
    dSdt = -beta * S * (I + delta * A) / N
    dIdt = beta * S * (I + delta * A) / N - alpha * I - gamma * I
    dDdt = gamma * theta * I - epsilon * D
    dAdt = gamma * (1 - theta) * I - epsilon * A
    dRdt = alpha * I
    dTdt = delta * alpha * A
    dHdt = epsilon * (D + A)
    dEdt = epsilon * (D + A)
    return dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt


def run_simulation(N, infected, duration, beta, alpha, gamma, delta, epsilon, theta):
    S0 = N - infected
    I0 = infected
    D0 = 0
    A0 = 0
    R0 = 0
    T0 = 0
    H0 = 0
    E0 = 0
    y0 = S0, I0, D0, A0, R0, T0, H0, E0
    t = np.linspace(0, duration, duration)
    solution = odeint(sidarthe_model, y0, t, args=(beta, alpha, gamma, delta, epsilon, theta))
    S, I, D, A, R, T, H, E = solution.T
    
    plt.plot(t, I, label='Infected')
    plt.plot(t, D, label='Deaths')
    plt.plot(t, A, label='Asymptomatic')
    plt.plot(t, R, label='Recovered')
    plt.plot(t, T, label='Transferred')
    plt.plot(t, H, label='Hospitalized')
    plt.plot(t, E, label='Exits')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of People')
    plt.title('SIDARTHE Model')
    plt.legend()
    plt.show()
}

