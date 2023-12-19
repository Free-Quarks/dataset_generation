import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, N, I0, R0, t_end):
    # Initial conditions
    S0 = N - I0 - R0
    y0 = [S0, I0, R0]
    
    def derivs(y, t):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]
    
    t = np.linspace(0, t_end, t_end+1)
    
    # Solve the differential equations
    y = odeint(derivs, y0, t)
    S, I, R = y.T
    
    # Plot the results
    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


# Example usage
beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
t_end = 100

SIR_model(beta, gamma, N, I0, R0, t_end)
