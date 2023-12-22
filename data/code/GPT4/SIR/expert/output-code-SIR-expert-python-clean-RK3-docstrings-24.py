import numpy as np
import matplotlib.pyplot as plt
import json

def sir_model_dyn(beta, gamma, S0, I0, R0, t):
    """
    This function computes the SIR model dynamics using 3rd order Runge-Kutta method.

    Parameters:
    beta: float
        Effective contact rate
    gamma: float
        Mean recovery rate
    S0: float
        Initial percentage of susceptible individuals
    I0: float
        Initial percentage of infected individuals
    R0: float
        Initial percentage of recovered individuals
    t: float
        Time

    Returns:
    S, I, R: tuple
        Tuple containing the percentage of susceptible, infected, and recovered individuals.
    """
    
    N = S0 + I0 + R0
    dt = t[1] - t[0]

    S, I, R = np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))
    S[0], I[0], R[0] = S0, I0, R0

    for i in range(len(t) - 1):
        k1_S = -beta * S[i] * I[i] / N
        k1_I = beta * S[i] * I[i] / N - gamma * I[i]
        k1_R = gamma * I[i]

        k2_S = -beta * (S[i] + 0.5 * dt * k1_S) * (I[i] + 0.5 * dt * k1_I) / N
        k2_I = beta * (S[i] + 0.5 * dt * k1_S) * (I[i] + 0.5 * dt * k1_I) / N - gamma * (I[i] + 0.5 * dt * k1_I)
        k2_R = gamma * (I[i] + 0.5 * dt * k1_I)

        S[i+1] = S[i] + dt * k2_S
        I[i+1] = I[i] + dt * k2_I
        R[i+1] = R[i] + dt * k2_R

    return S, I, R

def main():
    """
    The main function plots the percentage of susceptible, infected, and recovered individuals over time.
    """
    beta, gamma = 0.2, 1./10
    S0, I0, R0 = 99, 1, 0
    t = np.linspace(0, 160, 160)

    S, I, R = sir_model_dyn(beta, gamma, S0, I0, R0, t)

    plt.figure(figsize=[6, 4])
    plt.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
    plt.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
    plt.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered')
    plt.xlabel('Time /days')
    plt.ylabel('Number')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()

