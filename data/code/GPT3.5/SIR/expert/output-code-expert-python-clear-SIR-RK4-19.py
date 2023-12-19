import numpy as np
import matplotlib.pyplot as plt


def sir_model(beta, gamma, S0, I0, R0, t_max):
    dt = 0.01
    N = S0 + I0 + R0
    num_steps = int(t_max/dt)
    t = np.linspace(0, t_max, num_steps)
    S = np.zeros(num_steps)
    I = np.zeros(num_steps)
    R = np.zeros(num_steps)
    S[0] = S0
    I[0] = I0
    R[0] = R0

    for i in range(1, num_steps):
        dS_dt = -beta * S[i-1] * I[i-1] / N
        dI_dt = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dR_dt = gamma * I[i-1]
        S[i] = S[i-1] + dt * dS_dt
        I[i] = I[i-1] + dt * dI_dt
        R[i] = R[i-1] + dt * dR_dt

    return t, S, I, R


def plot_sir(t, S, I, R):
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


# Example usage
beta = 0.2
gamma = 0.1
S0 = 1000
I0 = 10
R0 = 0
t_max = 100

t, S, I, R = sir_model(beta, gamma, S0, I0, R0, t_max)
plot_sir(t, S, I, R)
