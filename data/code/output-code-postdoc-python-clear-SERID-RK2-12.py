import numpy as np
import matplotlib.pyplot as plt


def serid_rk2(beta, gamma, N, I0, R0, T):
    # Initial conditions
    S0 = N - I0 - R0
    
    # Step size
    dt = 0.01
    
    # Number of steps
    num_steps = int(T / dt)
    
    # Initialize arrays
    S = np.zeros(num_steps)
    I = np.zeros(num_steps)
    R = np.zeros(num_steps)
    
    # Set initial values
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Runge-Kutta 2nd order method
    for i in range(1, num_steps):
        k1 = dt * (-beta * S[i-1] * I[i-1] / N)
        k2 = dt * (-beta * (S[i-1] + 0.5 * k1) * (I[i-1] + 0.5 * k1) / N)
        
        S[i] = S[i-1] + k2
        
        k1 = dt * (beta * S[i-1] * I[i-1] / N - gamma * I[i-1])
        k2 = dt * (beta * (S[i-1] + 0.5 * k1) * (I[i-1] + 0.5 * k1) / N - gamma * (I[i-1] + 0.5 * k1))
        
        I[i] = I[i-1] + k2
        
        k1 = dt * (gamma * I[i-1])
        k2 = dt * (gamma * (I[i-1] + 0.5 * k1))
        
        R[i] = R[i-1] + k2
    
    return S, I, R


# Example usage
beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
T = 10

S, I, R = serid_rk2(beta, gamma, N, I0, R0, T)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SERID Model using RK2')
plt.legend()
plt.show()
