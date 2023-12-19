import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(beta, gamma, S0, I0, R0, t_end, num_steps):
    # Initialize arrays
    S = np.zeros(num_steps)
    I = np.zeros(num_steps)
    R = np.zeros(num_steps)
    t = np.linspace(0, t_end, num_steps)
    
    # Set initial conditions
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Calculate delta_t
    delta_t = t[1] - t[0]
    
    # Runge-Kutta 2nd order method
    for i in range(1, num_steps):
        k1 = delta_t * (beta * S[i-1] * I[i-1], -gamma * I[i-1], gamma * I[i-1])
        k2 = delta_t * (beta * (S[i-1] - 0.5 * k1[0]) * (I[i-1] - 0.5 * k1[1]) - gamma * (I[i-1] - 0.5 * k1[2]), 
                        -gamma * (I[i-1] - 0.5 * k1[1] + 0.5 * k1[2]), 
                        gamma * (I[i-1] - 0.5 * k1[2]))
        
        S[i] = S[i-1] - k2[0]
        I[i] = I[i-1] - k2[1]
        R[i] = R[i-1] + k2[2]
    
    # Plot the results
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model using RK2')
    plt.legend()
    plt.show()
}
