import numpy as np
import matplotlib.pyplot as plt

def sidarthe_model(t, y, beta, kappa, sigma, alpha, rho, gamma, delta):
    S, I, D, A, R, T, H, E = y
    N = S + I + D + A + R + T + H + E
    dSdt = -beta * S * (I + alpha * A + rho * T) / N
    dIdt = beta * S * (I + alpha * A + rho * T) / N - (kappa + sigma + gamma) * I
    dDdt = kappa * I
    dAdt = sigma * I - (delta + alpha) * A
    dRdt = gamma * I + delta * A
    dTdt = rho * T
    dHdt = alpha * A
    dEdt = beta * S * (I + alpha * A + rho * T) / N
    return [dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt]


def sidarthe_simulation(initial_conditions, parameters, t_start, t_end, delta_t):
    t = np.arange(t_start, t_end, delta_t)
    y = initial_conditions
    beta, kappa, sigma, alpha, rho, gamma, delta = unpack_parameters(parameters)
    result = odeint(sidarthe_model, y, t, args=(beta, kappa, sigma, alpha, rho, gamma, delta))
    S, I, D, A, R, T, H, E = result.T
    return t, S, I, D, A, R, T, H, E


def plot_sidarthe(t, S, I, D, A, R, T, H, E):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, S, label='Susceptible')
    ax.plot(t, I, label='Infected')
    ax.plot(t, D, label='Deceased')
    ax.plot(t, A, label='Asymptomatic')
    ax.plot(t, R, label='Recovered')
    ax.plot(t, T, label='Tested')
    ax.plot(t, H, label='Hospitalized')
    ax.plot(t, E, label='Exposed')
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of individuals')
    ax.set_title('SIDARTHE Model')
    ax.legend()
    plt.show()
