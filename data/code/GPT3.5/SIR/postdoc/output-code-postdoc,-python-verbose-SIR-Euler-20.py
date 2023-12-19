import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, N, I0, t_end):
    # Initialize arrays
    t = np.linspace(0, t_end, t_end+1)
    S = np.zeros(t_end+1)
    I = np.zeros(t_end+1)
    R = np.zeros(t_end+1)
    
    # Set initial conditions
    S[0] = N - I0
    I[0] = I0
    
    # Euler's method
    for i in range(t_end):
        S[i+1] = S[i] - beta/N * S[i] * I[i]
        I[i+1] = I[i] + beta/N * S[i] * I[i] - gamma * I[i]
        R[i+1] = R[i] + gamma * I[i]
    
    # Plotting
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()
}

