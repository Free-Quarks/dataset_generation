import numpy as np
import matplotlib.pyplot as plt


def seir_model(beta, sigma, gamma, initial_conditions, t_max):
    S_0, E_0, I_0, R_0 = initial_conditions
    N = S_0 + E_0 + I_0 + R_0
    dt = 0.01
    t = np.arange(0, t_max, dt)

    S = np.zeros_like(t)
    E = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)

    S[0] = S_0
    E[0] = E_0
    I[0] = I_0
    R[0] = R_0

    for i in range(1, len(t)):
        dSdt = -beta * S[i-1] * I[i-1] / N
        dEdt = beta * S[i-1] * I[i-1] / N - sigma * E[i-1]
        dIdt = sigma * E[i-1] - gamma * I[i-1]
        dRdt = gamma * I[i-1]

        S[i] = S[i-1] + dt * dSdt
        E[i] = E[i-1] + dt * dEdt
        I[i] = I[i-1] + dt * dIdt
        R[i] = R[i-1] + dt * dRdt

    return S, E, I, R


def plot_seir(S, E, I, R):
    t = np.arange(len(S))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, S, label='Susceptible')
    ax.plot(t, E, label='Exposed')
    ax.plot(t, I, label='Infected')
    ax.plot(t, R, label='Recovered')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Number of individuals')
    ax.set_title('SEIR Model')
    ax.legend()

    plt.show()


# Example usage
initial_conditions = (1000, 10, 1, 0)  # S, E, I, R
beta = 0.2
sigma = 0.1
gamma = 0.05
t_max = 100

S, E, I, R = seir_model(beta, sigma, gamma, initial_conditions, t_max)
plot_seir(S, E, I, R)
