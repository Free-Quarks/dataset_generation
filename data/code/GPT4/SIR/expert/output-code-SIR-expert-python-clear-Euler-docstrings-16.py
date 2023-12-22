import numpy as np
import matplotlib.pyplot as plt

def run_sir_model(S0, I0, R0, beta, gamma, t):
    """
    This function runs the SIR model using Euler's method.
    
    Parameters:
    S0 (float): Initial susceptible population.
    I0 (float): Initial infected population.
    R0 (float): Initial recovered population.
    beta (float): Contact rate.
    gamma (float): Recovery rate.
    t (int): Time range.
    
    Returns:
    S (list): Susceptible population over time.
    I (list): Infected population over time.
    R (list): Recovered population over time.
    """

    dt = 1.0
    N = S0 + I0 + R0

    S = [S0]
    I = [I0]
    R = [R0]

    for _ in range(t):
        next_S = S[-1] - (beta*S[-1]*I[-1]/N)*dt
        next_I = I[-1] + (beta*S[-1]*I[-1]/N - gamma*I[-1])*dt
        next_R = R[-1] + (gamma*I[-1])*dt

        S.append(next_S)
        I.append(next_I)
        R.append(next_R)

    return S, I, R
    
def plot_sir(S, I, R):
    """
    This function plots the SIR model.
    
    Parameters:
    S (list): Susceptible population over time.
    I (list): Infected population over time.
    R (list): Recovered population over time.
    """
    plt.figure(figsize=[6,4])
    plt.plot(S, label='Susceptible')
    plt.plot(I, label='Infected')
    plt.plot(R, label='Recovered')
    plt.legend()
    plt.grid()
    plt.show()

# example usage
S, I, R = run_sir_model(999, 1, 0, 0.5, 0.1, 200)
plot_sir(S, I, R)
