import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, N, I0, R0, T):
    # Define the initial conditions
    S0 = N - I0 - R0
    
    # Set the time step and create an array to store the results
    dt = 1
    num_steps = int(T / dt)
    t = np.linspace(0, T, num_steps)
    S = np.zeros(num_steps)
    I = np.zeros(num_steps)
    R = np.zeros(num_steps)
    
    # Set the initial values
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Implement the Euler's method
    for i in range(1, num_steps):
        dS = -beta * S[i-1] * I[i-1] / N
        dI = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dR = gamma * I[i-1]
        
        S[i] = S[i-1] + dt * dS
        I[i] = I[i-1] + dt * dI
        R[i] = R[i-1] + dt * dR
    
    # Plot the results
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Number of individuals')
    plt.legend()
    plt.show()
}

