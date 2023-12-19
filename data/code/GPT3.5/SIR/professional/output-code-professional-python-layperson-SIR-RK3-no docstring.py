import numpy as np
import matplotlib.pyplot as plt


def SIR_model(N, I0, R0, beta, gamma, t_max):
    # SIR model
    
    # Initial conditions
    S0 = N - I0 - R0
    
    # Step size
    dt = 0.01
    
    # Number of steps
    num_steps = int(t_max / dt)
    
    # Arrays to store results
    t = np.zeros(num_steps)
    S = np.zeros(num_steps)
    I = np.zeros(num_steps)
    R = np.zeros(num_steps)
    
    # Initialize values
    t[0] = 0
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    for i in range(1, num_steps):
        # Calculate derivatives
        dSdt = -beta * S[i-1] * I[i-1] / N
        dIdt = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dRdt = gamma * I[i-1]
        
        # Update values using RK3
        S[i] = S[i-1] + dt * dSdt
        I[i] = I[i-1] + dt * dIdt
        R[i] = R[i-1] + dt * dRdt
        
        # Update time
        t[i] = t[i-1] + dt
    
    # Plot results
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Number of individuals')
    plt.title('SIR Model')
    plt.legend()
    plt.show()
}

