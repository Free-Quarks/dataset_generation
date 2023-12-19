import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dS_dt = -beta * S * I
    dI_dt = beta * S * I - gamma * I
    dR_dt = gamma * I
    return [dS_dt, dI_dt, dR_dt]


def plot_results(t, S, I, R):
    plt.figure(figsize=(10,6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.grid()
    plt.show()


def main():
    # Initial conditions
    N = 1000
    I0, R0 = 1, 0
    S0 = N - I0 - R0
    y0 = [S0, I0, R0]

    # Time vector
    t = np.linspace(0, 100, 1000)

    # Parameters
    beta = 0.2
    gamma = 0.1

    # Solve the SIR model
    result = odeint(SIR_model, y0, t, args=(beta, gamma))
    S, I, R = result.T

    # Plot the results
    plot_results(t, S, I, R)


if __name__ == '__main__':
    main()
