import numpy as np
import matplotlib.pyplot as plt


# Define the SIR model

def SIR(initial_conditions, parameters, t_range):
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
    
    N = sum(initial_conditions)
    S0, I0, R0 = initial_conditions
    beta, gamma = parameters
    t = np.linspace(0, t_range, t_range)
    
    y0 = S0, I0, R0
    solution = odeint(deriv, y0, t, args=(N, beta, gamma))
    
    S, I, R = solution.T
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.title('SIR Model')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.show()


# Example usage

initial_conditions = [999, 1, 0]  # S, I, R
parameters = [0.3, 0.1]  # beta, gamma

SIR(initial_conditions, parameters, 100)

