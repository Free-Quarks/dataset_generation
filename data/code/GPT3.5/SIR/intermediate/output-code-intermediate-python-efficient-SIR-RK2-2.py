import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(beta, gamma, S0, I0, R0, t0, t_end, dt):
    # Initialize arrays
    t = np.arange(t0, t_end, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    
    # Set initial conditions
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Runge-Kutta 2nd order method
    for i in range(1, len(t)):
        k1_S = -beta * S[i-1] * I[i-1] / (S[i-1] + I[i-1] + R[i-1])
        k1_I = beta * S[i-1] * I[i-1] / (S[i-1] + I[i-1] + R[i-1]) - gamma * I[i-1]
        k2_S = -beta * (S[i-1] + dt * k1_S) * (I[i-1] + dt * k1_I) / (S[i-1] + I[i-1] + R[i-1])
        k2_I = beta * (S[i-1] + dt * k1_S) * (I[i-1] + dt * k1_I) / (S[i-1] + I[i-1] + R[i-1]) - gamma * (I[i-1] + dt * k1_I)
        
        S[i] = S[i-1] + dt * (0.5 * k1_S + 0.5 * k2_S)
        I[i] = I[i-1] + dt * (0.5 * k1_I + 0.5 * k2_I)
        R[i] = R[i-1] + dt * gamma * (I[i-1] + 0.5 * k1_I)
    
    # Return arrays
    return t, S, I, R

# Example usage
beta = 0.3
gamma = 0.1
S0 = 950
I0 = 50
R0 = 0
t0 = 0
t_end = 100
dt = 1

t, S, I, R = SIR_RK2(beta, gamma, S0, I0, R0, t0, t_end, dt)

plt.figure()
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model with RK2')
plt.legend()
plt.show()
