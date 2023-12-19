import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, S0, I0, R0, N, t)
    '''
    Simulate and plot the SIR model
    
    Parameters:
        - beta: float, the infection rate
        - gamma: float, the recovery rate
        - S0:  float, the initial number of susceptible individuals
        - I0:  float, the initial number of infected individuals
        - R0:  float, the initial number of recovered individuals
        - N:   float, the total population size
        - t:   array, the time points to simulate the model
    '''
    
    def SIR_model(beta, gamma, S, I, R):
        '''
        The differential equations for the SIR model
        '''
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
    
    S = S0
    I = I0
    R = R0
    S_vals = [S]
    I_vals = [I]
    R_vals = [R]
    
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        dS, dI, dR = SIR_model(beta, gamma, S, I, R)
        S += dt * dS
        I += dt * dI
        R += dt * dR
        S_vals.append(S)
        I_vals.append(I)
        R_vals.append(R)
    
    plt.figure()
    plt.plot(t, S_vals, label='Susceptible')
    plt.plot(t, I_vals, label='Infected')
    plt.plot(t, R_vals, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Number of Individuals')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


beta = 0.2
gamma = 0.1
S0 = 1000
I0 = 1
R0 = 0
N = S0 + I0 + R0
t = np.linspace(0, 100, 1000)

SIR_model(beta, gamma, S0, I0, R0, N, t)
