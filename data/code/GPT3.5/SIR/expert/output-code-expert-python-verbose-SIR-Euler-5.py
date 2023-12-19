import numpy as np
import matplotlib.pyplot as plt

def SIR_model(S, I, R, beta, gamma, t): 
    # Initialize arrays to store the values
    S_values = [S]
    I_values = [I]
    R_values = [R]
    t_values = [t]
    
    # Define the differential equations
    def dS_dt(S, I, beta):
        return -beta * S * I
    
    def dI_dt(S, I, beta, gamma):
        return beta * S * I - gamma * I
    
    def dR_dt(I, gamma):
        return gamma * I
    
    # Euler method to solve the differential equations
    dt = 0.01
    while t < 100:
        next_S = S + dt * dS_dt(S, I, beta)
        next_I = I + dt * dI_dt(S, I, beta, gamma)
        next_R = R + dt * dR_dt(I, gamma)
        
        S_values.append(next_S)
        I_values.append(next_I)
        R_values.append(next_R)
        t += dt
    
    # Plot the results
    plt.plot(t_values, S_values, label='Susceptible')
    plt.plot(t_values, I_values, label='Infected')
    plt.plot(t_values, R_values, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.show()
}
