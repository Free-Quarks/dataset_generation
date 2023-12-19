import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, N, I0, R0, T):
    def derivatives(y, t):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]


    t = np.linspace(0, T, T + 1)
    y0 = [N - I0 - R0, I0, R0]  # initial conditions
    result = integrate.odeint(derivatives, y0, t)
    S, I, R = result.T

    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Number of Individuals')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


# Example usage
beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
T = 100

SIR_model(beta, gamma, N, I0, R0, T)
