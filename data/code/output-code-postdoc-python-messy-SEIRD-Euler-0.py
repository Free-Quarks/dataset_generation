import numpy as np
import matplotlib.pyplot as plt

def seird_model(N, beta, sigma, gamma, mu, S0, E0, I0, R0, D0, t_end, dt):
    # Initialize arrays
    t = np.arange(0, t_end+dt, dt)
    S = np.zeros(len(t))
    E = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    D = np.zeros(len(t))
    
    # Set initial conditions
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    D[0] = D0
    
    # Euler's method
    for i in range(1, len(t)):
        dS = -beta * I[i-1] * S[i-1] / N
        dE = beta * I[i-1] * S[i-1] / N - sigma * E[i-1]
        dI = sigma * E[i-1] - (gamma + mu) * I[i-1]
        dR = gamma * I[i-1]
        dD = mu * I[i-1]
        
        S[i] = S[i-1] + dt * dS
        E[i] = E[i-1] + dt * dE
        I[i] = I[i-1] + dt * dI
        R[i] = R[i-1] + dt * dR
        D[i] = D[i-1] + dt * dD
    
    return t, S, E, I, R, D

# Example usage
N = 100000
beta = 0.2
sigma = 1/3
gamma = 1/7
mu = 1/14
S0 = N-1
E0 = 1
I0 = 0
R0 = 0
D0 = 0
T = 180
dt = 0.1

t, S, E, I, R, D = seird_model(N, beta, sigma, gamma, mu, S0, E0, I0, R0, D0, T, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.plot(t, D, label='Deceased')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SEIRD Model')
plt.show()

