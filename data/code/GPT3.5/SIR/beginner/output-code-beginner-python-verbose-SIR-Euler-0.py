import numpy as np
import matplotlib.pyplot as plt


def SIR_model(beta, gamma, N, I0, R0, T):
    # Set the initial conditions
    S0 = N - I0 - R0
    
    # Create arrays to store the results
    S = np.zeros(T)
    I = np.zeros(T)
    R = np.zeros(T)
    
    # Set the initial values
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Iterate over time steps
    for t in range(T-1):
        # Calculate the new values of S, I, and R
        dS = -beta*S[t]*I[t]/N
        dI = beta*S[t]*I[t]/N - gamma*I[t]
        dR = gamma*I[t]
        
        S[t+1] = S[t] + dS
        I[t+1] = I[t] + dI
        R[t+1] = R[t] + dR
    
    return S, I, R


# Set the parameters
beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
T = 100

# Run the model
S, I, R = SIR_model(beta, gamma, N, I0, R0, T)

# Plot the results
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
