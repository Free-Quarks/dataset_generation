import numpy as np
import matplotlib.pyplot as plt


def seir_model(beta, sigma, gamma, N, E0, I0, R0, t_end, dt):
    # Define initial conditions
    S0 = N - E0 - I0 - R0
    
    # Initialize arrays to store the values of each compartment
    S = np.zeros(int(t_end/dt)+1)
    E = np.zeros(int(t_end/dt)+1)
    I = np.zeros(int(t_end/dt)+1)
    R = np.zeros(int(t_end/dt)+1)
    
    # Set initial conditions
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    
    # Calculate the number of time steps
    Nt = int(t_end / dt)
    
    for t in range(Nt):
        # Calculate the derivatives
        dSdt = -beta * S[t] * I[t] / N
        dEdt = beta * S[t] * I[t] / N - sigma * E[t]
        dIdt = sigma * E[t] - gamma * I[t]
        dRdt = gamma * I[t]
        
        # Update the values of each compartment
        S[t+1] = S[t] + dt * dSdt
        E[t+1] = E[t] + dt * dEdt
        I[t+1] = I[t] + dt * dIdt
        R[t+1] = R[t] + dt * dRdt
    
    return S, E, I, R


# Example usage
beta = 0.2
sigma = 0.1
gamma = 0.05
N = 1000
E0 = 10
I0 = 1
R0 = 0
t_end = 100
dt = 0.1

S, E, I, R = seir_model(beta, sigma, gamma, N, E0, I0, R0, t_end, dt)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
