import numpy as np
import matplotlib.pyplot as plt


def serid_rk2(N, beta, gamma, tau, dt, T):
    # Create arrays to store the values
    S = np.zeros(T)
    E = np.zeros(T)
    I = np.zeros(T)
    R = np.zeros(T)
    D = np.zeros(T)
    t = np.arange(0, T, dt)
    
    # Set initial conditions
    S[0] = N-1
    E[0] = 1
    I[0] = 0
    R[0] = 0
    D[0] = 0
    
    # Euler's method
    for n in range(1, T):
        k1 = -beta * S[n-1] * I[n-1] / N
        l1 = beta * S[n-1] * I[n-1] / N - tau * E[n-1]
        m1 = tau * E[n-1] - gamma * I[n-1]
        n1 = gamma * I[n-1]
        
        k2 = -beta * (S[n-1] + dt * k1/2) * (I[n-1] + dt * m1/2) / N
        l2 = beta * (S[n-1] + dt * k1/2) * (I[n-1] + dt * m1/2) / N - tau * (E[n-1] + dt * l1/2)
        m2 = tau * (E[n-1] + dt * l1/2) - gamma * (I[n-1] + dt * m1/2)
        n2 = gamma * (I[n-1] + dt * m1/2)
        
        S[n] = S[n-1] + dt * (k1 + k2)/2
        E[n] = E[n-1] + dt * (l1 + l2)/2
        I[n] = I[n-1] + dt * (m1 + m2)/2
        R[n] = R[n-1] + dt * (n1 + n2)/2
        D[n] = D[n-1] + dt * (m1 + m2)/2
    
    # Return the arrays
    return S, E, I, R, D


# Example usage
N = 1000
beta = 0.3
gamma = 0.1
tau = 0.2
dt = 0.1
T = 100

S, E, I, R, D = serid_rk2(N, beta, gamma, tau, dt, T)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.plot(D, label='Dead')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SERID Model Simulation')
plt.show()
