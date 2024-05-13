import numpy as np
import matplotlib.pyplot as plt


def sidarthe_model(N, alpha, beta, gamma, delta, epsilon, theta, timesteps, initial_conditions):
    
    S_0, I_0, D_0, A_0, R_0, T_0, H_0, E_0 = initial_conditions
    S = np.zeros(timesteps)
    I = np.zeros(timesteps)
    D = np.zeros(timesteps)
    A = np.zeros(timesteps)
    R = np.zeros(timesteps)
    T = np.zeros(timesteps)
    H = np.zeros(timesteps)
    E = np.zeros(timesteps)
    
    S[0] = S_0
    I[0] = I_0
    D[0] = D_0
    A[0] = A_0
    R[0] = R_0
    T[0] = T_0
    H[0] = H_0
    E[0] = E_0
    
    dt = 1
    for t in range(timesteps-1):
        S[t+1] = S[t] - alpha*S[t]*I[t]*dt/N
        I[t+1] = I[t] + alpha*S[t]*I[t]*dt/N - beta*I[t]*dt - theta*I[t]*dt
        D[t+1] = D[t] + beta*I[t]*dt
        A[t+1] = A[t] + theta*I[t]*dt - gamma*A[t]*dt - delta*A[t]*dt
        R[t+1] = R[t] + gamma*A[t]*dt
        T[t+1] = T[t] + delta*A[t]*dt
        H[t+1] = H[t] + epsilon*A[t]*dt
        E[t+1] = E[t] + epsilon*A[t]*dt
    
    return S, I, D, A, R, T, H, E


N = 1000000  # Population size
alpha = 0.5  # Contact rate
beta = 0.1   # Infection rate
gamma = 0.1  # Recovery rate
theta = 0.1  # Hospitalization rate
epsilon = 0.2  # Intensive care rate
delta = 0.05  # Death rate
timesteps = 100
initial_conditions = (N-1, 1, 0, 0, 0, 0, 0, 0)  # (S, I, D, A, R, T, H, E) at t=0

S, I, D, A, R, T, H, E = sidarthe_model(N, alpha, beta, gamma, delta, epsilon, theta, timesteps, initial_conditions)

plt.plot(range(timesteps), S, label='Susceptible')
plt.plot(range(timesteps), I, label='Infected')
plt.plot(range(timesteps), D, label='Deceased')
plt.plot(range(timesteps), A, label='Asymptomatic')
plt.plot(range(timesteps), R, label='Recovered')
plt.plot(range(timesteps), T, label='Tested')
plt.plot(range(timesteps), H, label='Hospitalized')
plt.plot(range(timesteps), E, label='Intensive Care')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.legend()
plt.show()
