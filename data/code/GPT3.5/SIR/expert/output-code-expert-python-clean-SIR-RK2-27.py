import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, t_max, N, I0, R0):
    # Create arrays to store the values
    S = np.zeros(t_max+1)
    I = np.zeros(t_max+1)
    R = np.zeros(t_max+1)
    
    S[0] = N - I0 - R0
    I[0] = I0
    R[0] = R0
    
    dt = 0.1
    
    for t in range(t_max):
        dS = -beta * S[t] * I[t] / N
        dI = beta * S[t] * I[t] / N - gamma * I[t]
        dR = gamma * I[t]
        
        S[t+1] = S[t] + dt * dS
        I[t+1] = I[t] + dt * dI
        R[t+1] = R[t] + dt * dR
    
    return S, I, R


# Parameters
beta = 0.2
gamma = 0.1
t_max = 100
N = 1000
I0 = 1
R0 = 0

# Run the model
S, I, R = SIR_model(beta, gamma, t_max, N, I0, R0)

# Plot the results
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time step')
plt.ylabel('Population')
plt.title('SIR Model Simulation')
plt.legend()
plt.show()
