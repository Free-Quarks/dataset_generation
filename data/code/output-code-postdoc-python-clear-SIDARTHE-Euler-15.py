import numpy as np
import matplotlib.pyplot as plt

def SIR_model(S, I, R, beta, gamma, N, t_max):
    # Initialize arrays to store values
    S_values = []
    I_values = []
    R_values = []
    t_values = []
    
    # Append initial values to arrays
    S_values.append(S)
    I_values.append(I)
    R_values.append(R)
    t_values.append(0)
    
    # Euler's method
    dt = 0.01
    for t in np.arange(dt, t_max, dt):
        # Calculate new values
        dS = -beta * S * I / N
        dI = beta * S * I / N - gamma * I
        dR = gamma * I
        
        S += dS * dt
        I += dI * dt
        R += dR * dt
        
        # Append new values to arrays
        S_values.append(S)
        I_values.append(I)
        R_values.append(R)
        t_values.append(t)
    
    # Plot results
    plt.plot(t_values, S_values, label='Susceptible')
    plt.plot(t_values, I_values, label='Infected')
    plt.plot(t_values, R_values, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()
}

