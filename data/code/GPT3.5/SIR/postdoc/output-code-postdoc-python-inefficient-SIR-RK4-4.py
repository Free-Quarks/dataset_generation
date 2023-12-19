import numpy as np
import matplotlib.pyplot as plt


def SIR_RK4(initial_conditions, parameters, t_span, dt):
    # Extracting initial conditions
    S_0, I_0, R_0 = initial_conditions
    # Extracting parameters
    beta, gamma = parameters
    
    # Calculating number of time steps
    num_steps = int((t_span[1] - t_span[0]) / dt)
    
    # Initializing arrays for S, I, R
    S = np.zeros(num_steps + 1)
    I = np.zeros(num_steps + 1)
    R = np.zeros(num_steps + 1)
    
    # Setting initial values
    S[0] = S_0
    I[0] = I_0
    R[0] = R_0
    
    # Running RK4
    for i in range(num_steps):
        t = t_span[0] + i * dt
        
        S_k1 = -beta * S[i] * I[i]
        I_k1 = beta * S[i] * I[i] - gamma * I[i]
        R_k1 = gamma * I[i]
        
        S_k2 = -beta * (S[i] + dt * S_k1/2) * (I[i] + dt * I_k1/2)
        I_k2 = beta * (S[i] + dt * S_k1/2) * (I[i] + dt * I_k1/2) - gamma * (I[i] + dt * I_k1/2)
        R_k2 = gamma * (I[i] + dt * I_k1/2)
        
        S_k3 = -beta * (S[i] + dt * S_k2/2) * (I[i] + dt * I_k2/2)
        I_k3 = beta * (S[i] + dt * S_k2/2) * (I[i] + dt * I_k2/2) - gamma * (I[i] + dt * I_k2/2)
        R_k3 = gamma * (I[i] + dt * I_k2/2)
        
        S_k4 = -beta * (S[i] + dt * S_k3) * (I[i] + dt * I_k3)
        I_k4 = beta * (S[i] + dt * S_k3) * (I[i] + dt * I_k3) - gamma * (I[i] + dt * I_k3)
        R_k4 = gamma * (I[i] + dt * I_k3)
        
        S[i+1] = S[i] + dt/6 * (S_k1 + 2*S_k2 + 2*S_k3 + S_k4)
        I[i+1] = I[i] + dt/6 * (I_k1 + 2*I_k2 + 2*I_k3 + I_k4)
        R[i+1] = R[i] + dt/6 * (R_k1 + 2*R_k2 + 2*R_k3 + R_k4)
        
    return S, I, R


# Example usage

# Set initial conditions
initial_conditions = [999, 1, 0]

# Set parameters
parameters = [0.3, 0.1]

# Set time span
t_span = [0, 100]

# Set time step size
dt = 0.1

# Simulate SIR model using RK4
S, I, R = SIR_RK4(initial_conditions, parameters, t_span, dt)

# Plotting
plt.plot(np.arange(t_span[0], t_span[1]+dt, dt), S, label='Susceptible')
plt.plot(np.arange(t_span[0], t_span[1]+dt, dt), I, label='Infected')
plt.plot(np.arange(t_span[0], t_span[1]+dt, dt), R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK4')
plt.legend()
plt.show()


