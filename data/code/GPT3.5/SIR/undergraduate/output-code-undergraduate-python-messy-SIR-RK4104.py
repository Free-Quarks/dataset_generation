import numpy as np
import matplotlib.pyplot as plt


def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def RK4_SIR_model(y0, t, beta, gamma):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        k1 = h * SIR_model(y[i], t[i], beta, gamma)
        k2 = h * SIR_model(y[i] + 0.5 * k1, t[i] + 0.5 * h, beta, gamma)
        k3 = h * SIR_model(y[i] + 0.5 * k2, t[i] + 0.5 * h, beta, gamma)
        k4 = h * SIR_model(y[i] + k3, t[i] + h, beta, gamma)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y


def plot_SIR_model(t, y):
    S = y[:, 0]
    I = y[:, 1]
    R = y[:, 2]
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.grid()
    plt.show()


# Example usage
beta = 0.2
gamma = 0.1
y0 = [0.99, 0.01, 0.0]
t = np.linspace(0, 100, 1000)
y = RK4_SIR_model(y0, t, beta, gamma)
plot_SIR_model(t, y)
