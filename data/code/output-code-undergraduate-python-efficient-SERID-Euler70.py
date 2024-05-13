import numpy as np
import matplotlib.pyplot as plt

def SERID(beta, gamma, rho, delta, N, I0, R0, D0, T):
    # Set up arrays to store the results
    S = np.zeros(T)
    E = np.zeros(T)
    R = np.zeros(T)
    I = np.zeros(T)
    D = np.zeros(T)
    
    # Set initial conditions
    S[0] = N - I0 - R0 - D0
    E[0] = 0
    I[0] = I0
    R[0] = R0
    D[0] = D0
    
    # Euler's method to update the state variables
    for t in range(1, T):
        S[t] = S[t-1] - beta*S[t-1]*I[t-1]/N
        E[t] = E[t-1] + beta*S[t-1]*I[t-1]/N - gamma*E[t-1] - rho*E[t-1]
        I[t] = I[t-1] + rho*E[t-1] - delta*I[t-1]
        R[t] = R[t-1] + gamma*E[t-1]
        D[t] = D[t-1] + delta*I[t-1]
    
    return S, E, I, R, D

# Example usage
beta = 0.5
gamma = 0.1
rho = 0.2
N = 1000
I0 = 10
R0 = 0
D0 = 0
T = 100

S, E, I, R, D = SERID(beta, gamma, rho, N, I0, R0, D0, T)

plt.plot(S, label='Susceptible')
plt.plot(E, label='Exposed')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.plot(D, label='Dead')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.legend()
plt.show()
