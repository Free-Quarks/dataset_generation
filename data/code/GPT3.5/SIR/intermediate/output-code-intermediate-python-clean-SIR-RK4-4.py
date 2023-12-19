import numpy as np
import matplotlib.pyplot as plt


def SIR_RK4(beta, gamma, S0, I0, R0, t_max, dt):
    # Initialize arrays
    t = np.arange(0, t_max, dt)
    N = S0 + I0 + R0
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    
    # Set initial conditions
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Runge-Kutta 4th order method
    for i in range(1, len(t)):
        k1 = dt * (-beta * S[i-1] * I[i-1] / N)
        k2 = dt * (-beta * (S[i-1] + k1/2) * (I[i-1] + k1/2) / N)
        k3 = dt * (-beta * (S[i-1] + k2/2) * (I[i-1] + k2/2) / N)
        k4 = dt * (-beta * (S[i-1] + k3) * (I[i-1] + k3) / N)
        
        l1 = dt * (beta * S[i-1] * I[i-1] / N - gamma * I[i-1])
        l2 = dt * (beta * (S[i-1] + k1/2) * (I[i-1] + l1/2) / N - gamma * (I[i-1] + l1/2))
        l3 = dt * (beta * (S[i-1] + k2/2) * (I[i-1] + l2/2) / N - gamma * (I[i-1] + l2/2))
        l4 = dt * (beta * (S[i-1] + k3) * (I[i-1] + l3) / N - gamma * (I[i-1] + l3))
        
        S[i] = S[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
        I[i] = I[i-1] + (l1 + 2*l2 + 2*l3 + l4) / 6
        R[i] = R[i-1] + dt * gamma * (I[i-1] + l1/2)
    
    return S, I, R


# Parameters
beta = 0.2
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
t_max = 100
dt = 0.1

# Run model
S, I, R = SIR_RK4(beta, gamma, S0, I0, R0, t_max, dt)

# Plot results
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK4')
plt.legend()
plt.show()
