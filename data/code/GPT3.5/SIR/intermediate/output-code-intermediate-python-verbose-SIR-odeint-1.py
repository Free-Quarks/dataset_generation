import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / (S + I + R)
    dIdt = beta * S * I / (S + I + R) - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def plot_SIR(S, I, R, t):
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of Individuals')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Initial conditions
    S0 = 1000
    I0 = 1
    R0 = 0
    y0 = [S0, I0, R0]
    
    # Parameters
    beta = 0.2
    gamma = 0.1
    
    # Time vector
    t = np.linspace(0, 100, 1000)
    
    # Solve the ODEs
    solution = odeint(SIR_model, y0, t, args=(beta, gamma))
    S, I, R = solution.T
    
    # Plot the results
    plot_SIR(S, I, R, t)
