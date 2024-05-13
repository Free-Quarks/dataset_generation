import numpy as np
import matplotlib.pyplot as plt

def seir_model(beta, gamma, sigma, N, I0, E0, R0, T):
    # Total population
    S0 = N - I0 - E0 - R0
    
    # Set initial conditions
    S = [S0]
    E = [E0]
    I = [I0]
    R = [R0]
    
    # Set time step
    dt = T[1] - T[0]
    
    # Run the model
    for t in range(1, len(T)):
        dSdt = -beta * S[t-1] * I[t-1] / N
        dEdt = beta * S[t-1] * I[t-1] / N - sigma * E[t-1]
        dIdt = sigma * E[t-1] - gamma * I[t-1]
        dRdt = gamma * I[t-1]
        
        S.append(S[t-1] + dt * dSdt)
        E.append(E[t-1] + dt * dEdt)
        I.append(I[t-1] + dt * dIdt)
        R.append(R[t-1] + dt * dRdt)
        
    return S, E, I, R

# Parameters
beta = 0.2
gamma = 0.1
sigma = 0.05
N = 1000
I0 = 1
E0 = 0
R0 = 0
T = np.linspace(0, 100, 100)

# Run the model
S, E, I, R = seir_model(beta, gamma, sigma, N, I0, E0, R0, T)

# Plot the results
plt.plot(T, S, label='Susceptible')
plt.plot(T, E, label='Exposed')
plt.plot(T, I, label='Infected')
plt.plot(T, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SEIR Model')
plt.legend()
plt.show()
