import numpy as np
import matplotlib.pyplot as plt


def SIR_RK2(beta, gamma, N, I0, T):
    # Initialize arrays
    t = np.linspace(0, T, T+1)
    S = np.zeros(T+1)
    I = np.zeros(T+1)
    R = np.zeros(T+1)
    S[0] = N - I0
    I[0] = I0

    # Runge-Kutta 2nd order method
    for i in range(T):
        k1 = beta * S[i] * I[i] / N
        l1 = gamma * I[i]
        k2 = beta * (S[i] - k1/2) * (I[i] - l1/2) / N
        l2 = gamma * (I[i] - l1/2)
        S[i+1] = S[i] - k2
        I[i+1] = I[i] + k2 - l2
        R[i+1] = R[i] + l2

    # Plot results
    plt.figure()
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SIR Model using RK2')
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage
SIR_RK2(0.3, 0.1, 1000, 10, 100)
