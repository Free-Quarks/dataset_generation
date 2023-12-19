import numpy as np
import matplotlib.pyplot as plt


def SIR_model(S, I, R, beta, gamma, t): 
    N = S + I + R
    h = t[1] - t[0]
    S_n = []
    I_n = []
    R_n = []
    for i in range(len(t)):
        S_n.append(S)
        I_n.append(I)
        R_n.append(R)
        S_prime = -beta * S * I / N
        I_prime = beta * S * I / N - gamma * I
        R_prime = gamma * I
        S += h * S_prime
        I += h * I_prime
        R += h * R_prime
    return S_n, I_n, R_n


def plot_SIR(S, I, R, t):
    S_n, I_n, R_n = SIR_model(S, I, R, beta, gamma, t)
    plt.plot(t, S_n, label='S(t)')
    plt.plot(t, I_n, label='I(t)')
    plt.plot(t, R_n, label='R(t)')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()
