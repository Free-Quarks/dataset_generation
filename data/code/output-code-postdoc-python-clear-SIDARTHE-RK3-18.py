import numpy as np
import matplotlib.pyplot as plt


def sidarthe_model(beta=0.2, sigma=0.2, gamma=0.1, delta=0.1, rho=0.02, theta=0.02, N=1000, I0=1, D0=0, A0=0, R0=0, T=200):
    # Function to simulate SIDARTHE model
    
    # Initialize arrays
    t = np.linspace(0, T, T+1)
    S = np.zeros(T+1)
    I = np.zeros(T+1)
    D = np.zeros(T+1)
    A = np.zeros(T+1)
    R = np.zeros(T+1)
    T = np.zeros(T+1)
    
    # Set initial conditions
    S[0] = N - I0 - D0 - A0 - R0
    I[0] = I0
    D[0] = D0
    A[0] = A0
    R[0] = R0
    
    # Run simulation
    for i in range(T):
        dS = -beta*S[i]*I[i]/N - sigma*S[i]*A[i]/N
        dI = beta*S[i]*I[i]/N - gamma*I[i] - (1 - rho)*delta*I[i]
        dD = (1 - theta)*delta*I[i]
        dA = sigma*S[i]*A[i]/N - theta*A[i]
        dR = gamma*I[i] + (1 - theta)*theta*A[i]
        
        S[i+1] = S[i] + dS
        I[i+1] = I[i] + dI
        D[i+1] = D[i] + dD
        A[i+1] = A[i] + dA
        R[i+1] = R[i] + dR
        T[i+1] = T[i] + dD + dR
    
    return t, S, I, D, A, R, T


# Parameters
beta = 0.2
sigma = 0.2
gamma = 0.1
delta = 0.1
rho = 0.02
theta = 0.02
N = 1000
I0 = 1
D0 = 0
A0 = 0
R0 = 0
T = 200


# Run simulation
t, S, I, D, A, R, T = sidarthe_model(beta, sigma, gamma, delta, rho, theta, N, I0, D0, A0, R0, T)


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, D, label='Deceased')
plt.plot(t, A, label='Asymptomatic')
plt.plot(t, R, label='Recovered')
plt.plot(t, T, label='Total')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIDARTHE Model')
plt.legend()
plt.grid(True)
plt.show()
