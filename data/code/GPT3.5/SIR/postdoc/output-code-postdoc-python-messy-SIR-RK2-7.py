import numpy as np
import matplotlib.pyplot as plt


def SIR_RK2(beta, gamma, N, I0, T):
    # Define the differential equations
    def dSdt(S, I):
        return -beta * S * I / N

    def dIdt(S, I):
        return beta * S * I / N - gamma * I

    def dRdt(I):
        return gamma * I

    # Initial conditions
    S0 = N - I0
    I = np.zeros(T)
    R = np.zeros(T)
    S = np.zeros(T)
    S[0] = S0
    I[0] = I0

    # Runge-Kutta 2nd order method
    for t in range(T - 1):
        h = 1
        k1 = h * dSdt(S[t], I[t])
        k2 = h * dSdt(S[t] + 0.5 * k1, I[t] + 0.5 * k1)
        S[t + 1] = S[t] + k2

        k1 = h * dIdt(S[t], I[t])
        k2 = h * dIdt(S[t] + 0.5 * k1, I[t] + 0.5 * k1)
        I[t + 1] = I[t] + k2

        k1 = h * dRdt(I[t])
        k2 = h * dRdt(I[t] + 0.5 * k1)
        R[t + 1] = R[t] + k2

    # Plotting
    plt.plot(range(T), S, label='Susceptible')
    plt.plot(range(T), I, label='Infected')
    plt.plot(range(T), R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.title('SIR Model using RK2')
    plt.show()
}

