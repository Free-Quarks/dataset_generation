import numpy as np
import matplotlib.pyplot as plt


def SIR_RK3(beta, gamma, N, I0, t_max, dt):
    # Define the SIR model
    def SIR_model(y, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    # Initialize arrays
    t = np.arange(0, t_max, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    S[0] = N - I0
    I[0] = I0

    # Runge-Kutta 3rd order method
    for i in range(1, len(t)):
        y = np.array([S[i-1], I[i-1], R[i-1]])
        k1 = dt * SIR_model(y, beta, gamma)
        k2 = dt * SIR_model(y + 0.5 * k1, beta, gamma)
        k3 = dt * SIR_model(y - k1 + 2 * k2, beta, gamma)
        y += (k1 + 4 * k2 + k3) / 6
        S[i] = y[0]
        I[i] = y[1]
        R[i] = y[2]

    # Plotting
    plt.figure()
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model using RK3')
    plt.legend()
    plt.show()


SIR_RK3(0.3, 0.1, 1000, 10, 100, 0.1)
