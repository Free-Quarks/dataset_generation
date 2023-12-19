import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(beta, gamma, N, I0, R0, t_end):
    # Initialize arrays
    t = np.linspace(0, t_end, t_end + 1)
    dt = t[1] - t[0]
    S = np.zeros(t_end + 1)
    I = np.zeros(t_end + 1)
    R = np.zeros(t_end + 1)
    S[0] = N - I0 - R0
    I[0] = I0
    R[0] = R0

    # Runge-Kutta 3rd order integration
    for i in range(t_end):
        k1 = dt * (-beta * S[i] * I[i] / N)
        k2 = dt * (-beta * (S[i] + 0.5 * k1) * (I[i] + 0.5 * k1) / N)
        k3 = dt * (-beta * (S[i] - k1 + 2 * k2) * (I[i] - k1 + 2 * k2) / N)
        S[i + 1] = S[i] + (1 / 6) * (k1 + 4 * k2 + k3)
        I[i + 1] = I[i] + gamma * (k1 + 4 * k2 + k3)
        R[i + 1] = R[i] + gamma * (k1 + 4 * k2 + k3)

    # Plot the results
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model using RK3')
    plt.legend()
    plt.show()
}

