import numpy as np
import matplotlib.pyplot as plt


# Function to implement the SIDARTHE model

def sidarthe_model(N, beta, gamma, alpha, rho, theta, epsilon, t_max):
    # Initial conditions
    S0 = N - 1
    I0 = 1
    D0 = 0
    A0 = 0
    R0 = 0
    T0 = 0
    H0 = 0
    E0 = 0
    
    # Arrays to store the results
    S = np.zeros(t_max)
    I = np.zeros(t_max)
    D = np.zeros(t_max)
    A = np.zeros(t_max)
    R = np.zeros(t_max)
    T = np.zeros(t_max)
    H = np.zeros(t_max)
    E = np.zeros(t_max)
    
    # Assign initial conditions
    S[0] = S0
    I[0] = I0
    D[0] = D0
    A[0] = A0
    R[0] = R0
    T[0] = T0
    H[0] = H0
    E[0] = E0
    
    # Time step
    dt = 0.1
    
    # RK2 method
    for t in range(1, t_max):
        # Compute intermediate values
        S_half = S[t-1] - dt*beta*S[t-1]*(I[t-1] + rho*A[t-1]) / N
        I_half = I[t-1] + dt*(beta*S[t-1]*(I[t-1] + rho*A[t-1])/N - gamma*I[t-1] - alpha*I[t-1])
        D_half = D[t-1] + dt*theta*alpha*I[t-1]
        A_half = A[t-1] + dt*(gamma*I[t-1] - rho*beta*S[t-1]*A[t-1]/N)
        R_half = R[t-1] + dt*(1-theta)*alpha*I[t-1]
        T_half = T[t-1] + dt*rho*beta*S[t-1]*A[t-1]/N
        H_half = H[t-1] + dt*epsilon*D[t-1]
        E_half = E[t-1] + dt*(1-epsilon)*D[t-1]
        
        # Compute next step values
        S[t] = S[t-1] - dt*beta*S_half*(I_half + rho*A_half) / N
        I[t] = I[t-1] + dt*(beta*S_half*(I_half + rho*A_half)/N - gamma*I_half - alpha*I_half)
        D[t] = D[t-1] + dt*theta*alpha*I_half
        A[t] = A[t-1] + dt*(gamma*I_half - rho*beta*S_half*A_half/N)
        R[t] = R[t-1] + dt*(1-theta)*alpha*I_half
        T[t] = T[t-1] + dt*rho*beta*S_half*A_half/N
        H[t] = H[t-1] + dt*epsilon*D_half
        E[t] = E[t-1] + dt*(1-epsilon)*D_half
        
    return S, I, D, A, R, T, H, E


# Parameters
N = 100000  # Total population
beta = 0.3  # Infection rate
gamma = 0.1  # Recovery rate
alpha = 0.01  # Fatality rate
rho = 0.5  # Asymptomatic ratio
theta = 0.1  # Hospitalization rate
epsilon = 0.1  # ICU admission rate
t_max = 100  # Simulation duration

# Run the model
S, I, D, A, R, T, H, E = sidarthe_model(N, beta, gamma, alpha, rho, theta, epsilon, t_max)

# Plot the results
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(D, label='Deceased')
plt.plot(A, label='Asymptomatic')
plt.plot(R, label='Recovered')
plt.plot(T, label='Asymptomatic Transmission')
plt.plot(H, label='Hospitalized')
plt.plot(E, label='ICU admitted')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.legend()
plt.show()

