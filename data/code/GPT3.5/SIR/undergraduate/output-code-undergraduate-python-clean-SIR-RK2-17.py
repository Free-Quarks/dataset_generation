import numpy as np
import matplotlib.pyplot as plt

def sir_model(beta, gamma, N, I0, R0, t_end, dt):
    # Initialize arrays
    t = np.arange(0, t_end, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    
    S[0] = N - I0 - R0
    I[0] = I0
    R[0] = R0
    
    # Runge-Kutta Method
    for i in range(1, len(t)):
        # Calculate increments
        dS = -beta * S[i-1] * I[i-1] / N
        dI = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dR = gamma * I[i-1]
        
        # Update values
        S[i] = S[i-1] + dt * dS
        I[i] = I[i-1] + dt * dI
        R[i] = R[i-1] + dt * dR
    
    return t, S, I, R

# Example usage
beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
t_end = 100
dt = 0.1

t, S, I, R = sir_model(beta, gamma, N, I0, R0, t_end, dt)

plt.plot(t, S, label='S')
plt.plot(t, I, label='I')
plt.plot(t, R, label='R')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SIR Model')
plt.show()
