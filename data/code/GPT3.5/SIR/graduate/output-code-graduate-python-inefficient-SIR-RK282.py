import numpy as np
import matplotlib.pyplot as plt


def sir_model(beta, gamma, initial_conditions, t_max, dt):
    # Initialize arrays
    t = np.arange(0, t_max, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    
    # Set initial conditions
    S[0], I[0], R[0] = initial_conditions
    
    # Runge-Kutta 2nd order method
    for i in range(1, len(t)):
        k1 = dt * (-beta * S[i-1] * I[i-1], beta * S[i-1] * I[i-1] - gamma * I[i-1], gamma * I[i-1])
        k2 = dt * (-beta * (S[i-1] + k1[0]/2) * (I[i-1] + k1[1]/2), beta * (S[i-1] + k1[0]/2) * (I[i-1] + k1[1]/2) - gamma * (I[i-1] + k1[1]/2), gamma * (I[i-1] + k1[1]/2))
        S[i] = S[i-1] + k2[0]
        I[i] = I[i-1] + k2[1]
        R[i] = R[i-1] + k2[2]
        
    return t, S, I, R


# Example usage
beta = 0.3
gamma = 0.1
initial_conditions = (0.99, 0.01, 0.0)
t_max = 100
dt = 0.1

# Call the SIR model function
t, S, I, R = sir_model(beta, gamma, initial_conditions, t_max, dt)

# Plot the results
plt.plot(t, S, label='S')
plt.plot(t, I, label='I')
plt.plot(t, R, label='R')
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.title('SIR Model')
plt.legend()
plt.show()
