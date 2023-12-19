import numpy as np
import matplotlib.pyplot as plt


# Function to simulate SIR model using Runge-Kutta 2nd order method


def SIR_RK2(S0, I0, R0, beta, gamma, tmax, dt):
    # Initialize arrays
    S = np.zeros(int(tmax/dt) + 1)
    I = np.zeros(int(tmax/dt) + 1)
    R = np.zeros(int(tmax/dt) + 1)
    t = np.linspace(0, tmax, int(tmax/dt) + 1)
    
    # Set initial conditions
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Simulate the model using RK2 method
    for i in range(1, len(t)):
        k1_S = -beta * S[i-1] * I[i-1]
        k1_I = beta * S[i-1] * I[i-1] - gamma * I[i-1]
        k2_S = -beta * (S[i-1] + dt * k1_S/2) * (I[i-1] + dt * k1_I/2)
        k2_I = beta * (S[i-1] + dt * k1_S/2) * (I[i-1] + dt * k1_I/2) - gamma * (I[i-1] + dt * k1_I/2)
        
        S[i] = S[i-1] + dt * k2_S
        I[i] = I[i-1] + dt * k2_I
        R[i] = R[i-1] + dt * gamma * (I[i-1] + dt * k1_I/2)
        
    return S, I, R


# Parameters
S0 = 0.99  # Initial susceptible population
I0 = 0.01  # Initial infected population
R0 = 0  # Initial recovered population
beta = 0.4  # Infection rate
gamma = 0.1  # Recovery rate

# Simulation
tmax = 100  # Maximum time
dt = 0.1  # Time step

S, I, R = SIR_RK2(S0, I0, R0, beta, gamma, tmax, dt)

# Plotting
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK2')
plt.legend()
plt.show()

