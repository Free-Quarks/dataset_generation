import numpy as np
import matplotlib.pyplot as plt

def seir_model(beta, gamma, sigma, N, E0, I0, R0, t_max):
    # Total population
    S0 = N - E0 - I0 - R0
    
    # Time points
    t = np.linspace(0, t_max, t_max)
    
    # Step size
    h = t[1] - t[0]
    
    # Initialize arrays
    S = np.zeros(t.shape)
    E = np.zeros(t.shape)
    I = np.zeros(t.shape)
    R = np.zeros(t.shape)
    
    # Set initial conditions
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    
    # Runge-Kutta 4th order method
    for i in range(1, len(t)):
        # Calculate derivatives
        dSdt = -beta * I[i-1] * S[i-1] / N
        dEdt = beta * I[i-1] * S[i-1] / N - sigma * E[i-1]
        dIdt = sigma * E[i-1] - gamma * I[i-1]
        dRdt = gamma * I[i-1]
        
        # Update values using RK4 method
        S[i] = S[i-1] + h * dSdt
        E[i] = E[i-1] + h * dEdt
        I[i] = I[i-1] + h * dIdt
        R[i] = R[i-1] + h * dRdt
    
    return S, E, I, R

# Example usage
beta = 0.5
gamma = 0.2
sigma = 0.1
N = 1000
E0 = 10
I0 = 1
R0 = 0
t_max = 100

S, E, I, R = seir_model(beta, gamma, sigma, N, E0, I0, R0, t_max)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SEIR Model Simulation')
plt.show()
