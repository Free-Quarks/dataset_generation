import numpy as np
import matplotlib.pyplot as plt


def serid_euler(beta, gamma, N, I0, R0, T):
    # Set up the time grid
    t = np.linspace(0, T, T + 1)
    dt = t[1] - t[0]
    
    # Initialize arrays to store the values
    S = np.zeros(T + 1)
    I = np.zeros(T + 1)
    R = np.zeros(T + 1)
    
    # Set initial conditions
    S[0] = N - I0 - R0
    I[0] = I0
    R[0] = R0
    
    # Iterate over time steps
    for i in range(T):
        # Calculate the derivatives
        dSdt = -beta * S[i] * I[i] / N
        dIdt = beta * S[i] * I[i] / N - gamma * I[i]
        dRdt = gamma * I[i]
        
        # Update the values using Euler's method
        S[i+1] = S[i] + dt * dSdt
        I[i+1] = I[i] + dt * dIdt
        R[i+1] = R[i] + dt * dRdt
        
    return S, I, R


beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
T = 100

S, I, R = serid_euler(beta, gamma, N, I0, R0, T)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SERID Model with Euler Method')
plt.show()
