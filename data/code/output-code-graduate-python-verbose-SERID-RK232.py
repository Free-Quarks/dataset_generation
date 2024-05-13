import numpy as np
import matplotlib.pyplot as plt

def SERID(beta, gamma, sigma, N, I0, E0, R0, D0, t_max):
    def dSdt(S, I, E, R, D, beta, gamma, sigma):
        dSdt = -beta * S * I / N
        return dSdt

    def dEdt(S, I, E, R, D, beta, gamma, sigma):
        dEdt = beta * S * I / N - sigma * E
        return dEdt

    def dIdt(S, I, E, R, D, beta, gamma, sigma):
        dIdt = sigma * E - gamma * I
        return dIdt

    def dRdt(S, I, E, R, D, beta, gamma, sigma):
        dRdt = gamma * I
        return dRdt

    def dDdt(S, I, E, R, D, beta, gamma, sigma):
        dDdt = sigma * E
        return dDdt

    S = np.zeros(t_max)
    E = np.zeros(t_max)
    I = np.zeros(t_max)
    R = np.zeros(t_max)
    D = np.zeros(t_max)

    S[0] = N - I0 - E0 - R0 - D0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    D[0] = D0

    for t in range(1, t_max):
        h = 1
        S[t] = S[t-1] + h * dSdt(S[t-1], I[t-1], E[t-1], R[t-1], D[t-1], beta, gamma, sigma)
        E[t] = E[t-1] + h * dEdt(S[t-1], I[t-1], E[t-1], R[t-1], D[t-1], beta, gamma, sigma)
        I[t] = I[t-1] + h * dIdt(S[t-1], I[t-1], E[t-1], R[t-1], D[t-1], beta, gamma, sigma)
        R[t] = R[t-1] + h * dRdt(S[t-1], I[t-1], E[t-1], R[t-1], D[t-1], beta, gamma, sigma)
        D[t] = D[t-1] + h * dDdt(S[t-1], I[t-1], E[t-1], R[t-1], D[t-1], beta, gamma, sigma)

    plt.plot(range(t_max), S, label='Susceptible')
    plt.plot(range(t_max), E, label='Exposed')
    plt.plot(range(t_max), I, label='Infected')
    plt.plot(range(t_max), R, label='Recovered')
    plt.plot(range(t_max), D, label='Deceased')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.show()


SERID(0.4, 0.1, 0.2, 1000, 10, 5, 2, 1, 100)
