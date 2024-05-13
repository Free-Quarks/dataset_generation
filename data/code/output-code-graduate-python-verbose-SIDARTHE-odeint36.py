import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def sidarthe_model(y, t, N, beta, sigma, delta, alpha, rho, theta, kappa):
    S, I, D, A, R, T, H, E = y
    dSdt = -beta * S * (I + D + A + R + T + H + E) / N
    dIdt = beta * S * (I + D + A + R + T + H + E) / N - (sigma + delta + alpha) * I
    dDdt = delta * I
    dAdt = alpha * I - (rho + theta) * A
    dRdt = rho * A
    dTdt = theta * A
    dHdt = sigma * I
    dEdt = kappa * I
    return dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt


def run_sidarthe_model(N, days, beta, sigma, delta, alpha, rho, theta, kappa):
    S0, I0, D0, A0, R0, T0, H0, E0 = N-1, 1, 0, 0, 0, 0, 0, 0
    y0 = S0, I0, D0, A0, R0, T0, H0, E0
    t = np.linspace(0, days, days)
    result = odeint(sidarthe_model, y0, t, args=(N, beta, sigma, delta, alpha, rho, theta, kappa))
    S, I, D, A, R, T, H, E = result.T

    plt.figure(figsize=(10,6))
    plt.plot(t, I, 'b-', label='Infected')
    plt.plot(t, D, 'r-', label='Deceased')
    plt.plot(t, A, 'g-', label='Asymptomatic')
    plt.plot(t, R, 'c-', label='Recovered')
    plt.plot(t, T, 'm-', label='Transferred')
    plt.plot(t, H, 'y-', label='Hospitalized')
    plt.plot(t, E, 'k-', label='Exposed')
    plt.xlabel('Days')
    plt.ylabel('Number of Individuals')
    plt.grid(True)
    plt.legend()
    plt.show()
}

