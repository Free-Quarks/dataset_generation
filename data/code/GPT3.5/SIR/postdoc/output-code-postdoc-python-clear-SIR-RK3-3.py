import numpy as np
import matplotlib.pyplot as plt


def SIR_RK3(N, beta, gamma, num_days):
    # Initial conditions
    S0 = N - 1
    I0 = 1
    R0 = 0
    
    # Time step
    dt = 0.1
    
    # Number of steps
    num_steps = int(num_days / dt)
    
    # Arrays to store results
    S = np.zeros(num_steps)
    I = np.zeros(num_steps)
    R = np.zeros(num_steps)
    
    # Set initial conditions
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Run simulation
    for i in range(1, num_steps):
        # Calculate derivatives
        dSdt = -beta * S[i-1] * I[i-1] / N
        dIdt = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dRdt = gamma * I[i-1]
        
        # Update state using RK3
        S[i] = S[i-1] + dt * dSdt
        I[i] = I[i-1] + dt * dIdt
        R[i] = R[i-1] + dt * dRdt
        
    # Plot results
    t = np.linspace(0, num_days, num_steps)
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of individuals')
    plt.legend()
    plt.title('SIR Model')
    plt.show()
}

