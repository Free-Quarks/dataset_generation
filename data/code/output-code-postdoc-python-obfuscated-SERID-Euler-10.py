import numpy as np
import matplotlib.pyplot as plt

def serid_euler(N, beta, gamma, mu, rho, S0, I0, R0, D0, T):
    h = 0.01
    t = np.linspace(0, T, int(T/h)+1)
    S = np.zeros(int(T/h)+1)
    I = np.zeros(int(T/h)+1)
    R = np.zeros(int(T/h)+1)
    D = np.zeros(int(T/h)+1)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    D[0] = D0
    for i in range(1, int(T/h)+1):
        S[i] = S[i-1] + h * (mu * N - beta * S[i-1] * I[i-1] / N - rho * S[i-1])
        I[i] = I[i-1] + h * (beta * S[i-1] * I[i-1] / N - gamma * I[i-1] - rho * I[i-1])
        R[i] = R[i-1] + h * (gamma * I[i-1] - rho * R[i-1])
        D[i] = D[i-1] + h * (rho * S[i-1] + rho * I[i-1] + rho * R[i-1])
    return t, S, I, R, D

def plot_serid(t, S, I, R, D):
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.plot(t, D, label='Dead')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SERID Model')
    plt.legend()
    plt.show()

