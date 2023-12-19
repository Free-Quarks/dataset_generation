import numpy as np
import matplotlib.pyplot as plt


def sir_model(beta, gamma, N, I0, T):
    dt = 0.1
    num_steps = int(T/dt)
    t = np.linspace(0, T, num_steps+1)
    S = np.zeros(num_steps+1)
    I = np.zeros(num_steps+1)
    R = np.zeros(num_steps+1)
    S[0] = N - I0
    I[0] = I0
    R[0] = 0

    for i in range(num_steps):
        dSdt = -beta * S[i] * I[i]/N
        dIdt = beta * S[i] * I[i]/N - gamma * I[i]
        dRdt = gamma * I[i]
        S[i+1] = S[i] + dt * dSdt
        I[i+1] = I[i] + dt * dIdt
        R[i+1] = R[i] + dt * dRdt

    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


sir_model(0.3, 0.1, 1000, 10, 10)
