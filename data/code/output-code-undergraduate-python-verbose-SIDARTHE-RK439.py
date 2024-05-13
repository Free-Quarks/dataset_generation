import numpy as np
import matplotlib.pyplot as plt


def sidarthe_model(y, t, beta, sigma, gamma, tau, lambd, kappa):
    S, I, D, A, R, T, H, E = y
    N = S + I + D + A + R + T + H + E
    dSdt = -beta * S * I / N
    dIdt = (beta * S * I / N) - (sigma * I) - (gamma * I) - (tau * I)
    dDdt = tau * I
    dAdt = (sigma * I) - (lambd * A) - (kappa * A)
    dRdt = gamma * I + lambd * A
    dTdt = kappa * A
    dHdt = sigma * I
    dEdt = beta * S * I / N
    return [dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt]


def run_sidarthe_model(S0, I0, D0, A0, R0, T0, H0, E0, beta, sigma, gamma, tau, lambd, kappa, t_end, t_step):
    t = np.arange(0, t_end, t_step)
    y0 = [S0, I0, D0, A0, R0, T0, H0, E0]
    result = odeint(sidarthe_model, y0, t, args=(beta, sigma, gamma, tau, lambd, kappa))
    S, I, D, A, R, T, H, E = result.T

    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, D, label='Deceased')
    plt.plot(t, A, label='Asymptomatic')
    plt.plot(t, R, label='Recovered')
    plt.plot(t, T, label='Terminal')
    plt.plot(t, H, label='Hospitalized')
    plt.plot(t, E, label='Exposed')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SIDARTHE Model')
    plt.legend()
    plt.show()
}

