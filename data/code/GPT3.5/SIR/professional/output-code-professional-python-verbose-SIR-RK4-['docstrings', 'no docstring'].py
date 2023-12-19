import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, S0, I0, R0, num_days):
    '''
    Function to simulate and plot SIR model using Runge-Kutta 4th order method.
    Parameters:
        beta (float): the transmission rate
        gamma (float): the recovery rate
        S0 (int): the initial number of susceptible individuals
        I0 (int): the initial number of infected individuals
        R0 (int): the initial number of recovered individuals
        num_days (int): the number of days to simulate
    Returns:
        S (array): array of susceptible individuals over time
        I (array): array of infected individuals over time
        R (array): array of recovered individuals over time
    '''
    def deriv(SIR, t):
        S, I, R = SIR
        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
    
    t = np.linspace(0, num_days, num_days)
    y0 = S0, I0, R0
    S, I, R = odeint(deriv, y0, t).T
    
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of Individuals')
    plt.title('SIR Model Simulation')
    plt.legend()
    plt.show()
}
