import numpy as np
import matplotlib.pyplot as plt


def SIR_RK3(beta, gamma, S0, I0, R0, t0, tf, dt):
    # Initialize arrays
    t = np.arange(t0, tf+dt, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    
    # Set initial conditions
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Runge-Kutta 3rd order method
    for i in range(1, len(t)):
        k1 = -beta*S[i-1]*I[i-1]
        k2 = -beta*(S[i-1] + (dt/2)*k1)*(I[i-1] + (dt/2)*k1)
        k3 = -beta*(S[i-1] - dt*k1 + 2*dt*k2)*(I[i-1] - dt*k1 + 2*dt*k2)
        
        S[i] = S[i-1] + (dt/6)*(k1 + 4*k2 + k3)
        I[i] = I[i-1] + (dt/6)*(k1 + 4*k2 + k3)
        R[i] = R[i-1] + gamma*dt*I[i-1]
    
    return S, I, R


# Parameters
beta = 0.2
gamma = 0.1
S0 = 1000
I0 = 1
R0 = 0
t0 = 0
tf = 100
dt = 0.1

# Run the model
S, I, R = SIR_RK3(beta, gamma, S0, I0, R0, t0, tf, dt)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK3')
plt.legend()
plt.grid(True)
plt.show()
