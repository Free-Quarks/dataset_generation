import numpy as np
import matplotlib.pyplot as plt


def SIR_RK4(beta, gamma, N, I0, R0, t_end, dt):
    def SIR_model(t, y):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    # Initialize arrays to store the values
    t = np.arange(0, t_end, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))

    # Set initial conditions
    S[0] = N - I0
    I[0] = I0
    R[0] = R0

    # Runge-Kutta 4th order method
    for i in range(1, len(t)):
        h = dt
        k1 = SIR_model(t[i-1], [S[i-1], I[i-1], R[i-1]])
        k2 = SIR_model(t[i-1] + 0.5 * h, [S[i-1] + 0.5 * h * k1[0], I[i-1] + 0.5 * h * k1[1], R[i-1] + 0.5 * h * k1[2]])
        k3 = SIR_model(t[i-1] + 0.5 * h, [S[i-1] + 0.5 * h * k2[0], I[i-1] + 0.5 * h * k2[1], R[i-1] + 0.5 * h * k2[2]])
        k4 = SIR_model(t[i-1] + h, [S[i-1] + h * k3[0], I[i-1] + h * k3[1], R[i-1] + h * k3[2]])

        S[i] = S[i-1] + (h / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        I[i] = I[i-1] + (h / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        R[i] = R[i-1] + (h / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])

    # Plot the results
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model using RK4')
    plt.legend()
    plt.show()


# Example usage
beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
t_end = 100
dt = 0.1

SIR_RK4(beta, gamma, N, I0, R0, t_end, dt)
