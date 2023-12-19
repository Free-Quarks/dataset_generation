import numpy as np
import matplotlib.pyplot as plt

def SIR_Euler(N, I0, R0, beta, gamma, t_end, dt):
    t = np.arange(0, t_end, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = N - I0
    I[0] = I0
    R[0] = R0

    for i in range(1, len(t)):
        dS = -beta * S[i-1] * I[i-1] / N
        dI = (beta * S[i-1] * I[i-1] / N) - (gamma * I[i-1])
        dR = gamma * I[i-1]
        S[i] = S[i-1] + dt * dS
        I[i] = I[i-1] + dt * dI
        R[i] = R[i-1] + dt * dR

    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.title('SIR Model using Euler Method')
    plt.show()

SIR_Euler(1000, 1, 0, 0.3, 0.1, 100, 0.1)

