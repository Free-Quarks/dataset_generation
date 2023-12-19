import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, S0, I0, R0, t0, tf, dt):
    N = S0 + I0 + R0
    t = np.arange(t0, tf, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = S0
    I[0] = I0
    R[0] = R0
    for i in range(1, len(t)):
        dSdt = -beta * S[i-1] * I[i-1] / N
        dIdt = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dRdt = gamma * I[i-1]
        S[i] = S[i-1] + dt * dSdt
        I[i] = I[i-1] + dt * dIdt
        R[i] = R[i-1] + dt * dRdt
    return t, S, I, R


def plot_SIR(t, S, I, R):
    fig, ax = plt.subplots()
    ax.plot(t, S, label='Susceptible')
    ax.plot(t, I, label='Infected')
    ax.plot(t, R, label='Recovered')
    ax.set(xlabel='Time', ylabel='Population', title='SIR Model')
    ax.legend()
    plt.show()


# Example usage
beta = 0.2
gamma = 0.1
S0 = 1000
I0 = 1
R0 = 0
t0 = 0
tf = 100
dt = 0.1

t, S, I, R = SIR_model(beta, gamma, S0, I0, R0, t0, tf, dt)
plot_SIR(t, S, I, R)
